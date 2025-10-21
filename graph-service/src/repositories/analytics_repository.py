"""
Analytics repository for reporting and system insights
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from ..utils.neo4j_utils import Neo4jClient

logger = logging.getLogger(__name__)


class AnalyticsRepository:
    """
    Repository for analytics, reporting, and system insights
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
    
    # ==================== System Overview Analytics ====================
    
    async def get_system_overview(self, vehicle_signature: str) -> Dict[str, Any]:
        """Get comprehensive system overview"""
        
        query = """
        MATCH (v:Vehicle {signature: $vehicle_signature})
        
        // Count all entities
        OPTIONAL MATCH (c:Component {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (circuit:Circuit {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (z:Zone {vehicle_signature: $vehicle_signature})
        
        // Count relationships
        OPTIONAL MATCH ()-[conn:CONNECTS_TO]->()
        WHERE (startNode(conn).vehicle_signature = $vehicle_signature AND 
               endNode(conn).vehicle_signature = $vehicle_signature)
        
        OPTIONAL MATCH ()-[power:POWERED_BY]->()
        WHERE (startNode(power).vehicle_signature = $vehicle_signature AND 
               endNode(power).vehicle_signature = $vehicle_signature)
        
        OPTIONAL MATCH ()-[control:CONTROLS]->()
        WHERE (startNode(control).vehicle_signature = $vehicle_signature AND 
               endNode(control).vehicle_signature = $vehicle_signature)
        
        // Data completeness metrics
        WITH v, 
             collect(DISTINCT c) as components,
             collect(DISTINCT circuit) as circuits,
             collect(DISTINCT z) as zones,
             collect(DISTINCT conn) as connections,
             collect(DISTINCT power) as power_rels,
             collect(DISTINCT control) as control_rels
        
        RETURN 
            v,
            size(components) as component_count,
            size(circuits) as circuit_count,
            size(zones) as zone_count,
            size(connections) as connection_count,
            size(power_rels) as power_relationship_count,
            size(control_rels) as control_relationship_count,
            size([c in components WHERE c.position IS NOT NULL]) as positioned_components,
            size([c in components WHERE c.voltage_rating IS NOT NULL]) as components_with_voltage,
            size([c in components WHERE c.current_rating IS NOT NULL]) as components_with_current
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        if not result.records:
            return {"error": "Vehicle not found"}
        
        record = result.records[0]
        vehicle_data = dict(record["v"])
        
        total_components = record["component_count"]
        
        # Calculate completeness percentages
        position_completeness = (record["positioned_components"] / total_components * 100) if total_components > 0 else 0
        voltage_completeness = (record["components_with_voltage"] / total_components * 100) if total_components > 0 else 0
        current_completeness = (record["components_with_current"] / total_components * 100) if total_components > 0 else 0
        
        return {
            "vehicle": vehicle_data,
            "counts": {
                "components": total_components,
                "circuits": record["circuit_count"],
                "zones": record["zone_count"],
                "connections": record["connection_count"],
                "power_relationships": record["power_relationship_count"],
                "control_relationships": record["control_relationship_count"]
            },
            "data_completeness": {
                "position_data_percent": round(position_completeness, 1),
                "voltage_data_percent": round(voltage_completeness, 1),
                "current_data_percent": round(current_completeness, 1)
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def get_component_type_analytics(self, vehicle_signature: str) -> Dict[str, Any]:
        """Get detailed analytics by component type"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        RETURN 
            c.type as component_type,
            count(c) as count,
            avg(c.voltage_rating) as avg_voltage,
            avg(c.current_rating) as avg_current,
            sum(c.voltage_rating * c.current_rating) as total_power,
            size([comp in collect(c) WHERE comp.position IS NOT NULL]) as positioned_count,
            collect(DISTINCT c.zone) as zones_used,
            collect(DISTINCT c.manufacturer) as manufacturers
        ORDER BY count DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        type_analytics = []
        total_components = 0
        
        for record in result.records:
            component_type = record["component_type"]
            count = record["count"]
            total_components += count
            
            type_analytics.append({
                "type": component_type,
                "count": count,
                "avg_voltage": record["avg_voltage"],
                "avg_current": record["avg_current"],
                "total_power": record["total_power"] or 0,
                "positioned_count": record["positioned_count"],
                "position_coverage_percent": (record["positioned_count"] / count * 100) if count > 0 else 0,
                "zones_used": [z for z in record["zones_used"] if z],
                "manufacturers": [m for m in record["manufacturers"] if m]
            })
        
        # Add percentage of total for each type
        for analytics in type_analytics:
            analytics["percentage_of_total"] = (analytics["count"] / total_components * 100) if total_components > 0 else 0
        
        return {
            "vehicle_signature": vehicle_signature,
            "total_components": total_components,
            "type_analytics": type_analytics,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    async def get_zone_analytics(self, vehicle_signature: str) -> Dict[str, Any]:
        """Get analytics by physical zone"""
        
        query = """
        // Zone-based analytics
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.zone IS NOT NULL
        
        RETURN 
            c.zone as zone,
            count(c) as component_count,
            collect(DISTINCT c.type) as component_types,
            avg(c.voltage_rating) as avg_voltage,
            avg(c.current_rating) as avg_current,
            sum(c.voltage_rating * c.current_rating) as total_power,
            size([comp in collect(c) WHERE comp.position IS NOT NULL]) as positioned_count
        ORDER BY component_count DESC
        
        UNION ALL
        
        // Components without zone assignment
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.zone IS NULL
        
        RETURN 
            'unassigned' as zone,
            count(c) as component_count,
            collect(DISTINCT c.type) as component_types,
            avg(c.voltage_rating) as avg_voltage,
            avg(c.current_rating) as avg_current,
            sum(c.voltage_rating * c.current_rating) as total_power,
            size([comp in collect(c) WHERE comp.position IS NOT NULL]) as positioned_count
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        zone_analytics = []
        total_components = 0
        
        for record in result.records:
            zone = record["zone"]
            count = record["component_count"]
            total_components += count
            
            zone_analytics.append({
                "zone": zone,
                "component_count": count,
                "component_types": record["component_types"],
                "type_diversity": len(record["component_types"]),
                "avg_voltage": record["avg_voltage"],
                "avg_current": record["avg_current"],
                "total_power": record["total_power"] or 0,
                "positioned_count": record["positioned_count"],
                "position_coverage_percent": (record["positioned_count"] / count * 100) if count > 0 else 0
            })
        
        # Add percentage of total for each zone
        for analytics in zone_analytics:
            analytics["percentage_of_total"] = (analytics["component_count"] / total_components * 100) if total_components > 0 else 0
        
        return {
            "vehicle_signature": vehicle_signature,
            "total_components": total_components,
            "zone_analytics": zone_analytics,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # ==================== Circuit Analytics ====================
    
    async def get_circuit_analytics(self, vehicle_signature: str) -> Dict[str, Any]:
        """Get comprehensive circuit analytics"""
        
        query = """
        MATCH (circuit:Circuit {vehicle_signature: $vehicle_signature})
        
        // Get components in each circuit
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(circuit)
        
        WITH circuit, collect(comp) as components,
             sum(coalesce(comp.current_rating, 0)) as total_load_current,
             sum(coalesce(comp.voltage_rating * comp.current_rating, 0)) as total_power
        
        RETURN 
            circuit.id as circuit_id,
            circuit.name as circuit_name,
            circuit.circuit_type as circuit_type,
            circuit.voltage as circuit_voltage,
            circuit.max_current as max_capacity,
            circuit.protection as protection_type,
            circuit.safety_critical as is_safety_critical,
            size(components) as component_count,
            total_load_current,
            total_power,
            CASE 
                WHEN circuit.max_current > 0 THEN (total_load_current / circuit.max_current) * 100
                ELSE 0
            END as load_percentage,
            CASE
                WHEN circuit.max_current > 0 THEN circuit.max_current - total_load_current
                ELSE 0
            END as available_capacity
        ORDER BY load_percentage DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        circuit_analytics = []
        overloaded_circuits = 0
        heavily_loaded_circuits = 0
        safety_critical_circuits = 0
        
        for record in result.records:
            load_percentage = record["load_percentage"]
            is_safety_critical = record["is_safety_critical"]
            
            if load_percentage > 100:
                overloaded_circuits += 1
            elif load_percentage > 80:
                heavily_loaded_circuits += 1
            
            if is_safety_critical:
                safety_critical_circuits += 1
            
            circuit_analytics.append({
                "circuit_id": record["circuit_id"],
                "circuit_name": record["circuit_name"],
                "circuit_type": record["circuit_type"],
                "voltage": record["circuit_voltage"],
                "max_capacity": record["max_capacity"],
                "protection_type": record["protection_type"],
                "is_safety_critical": is_safety_critical,
                "component_count": record["component_count"],
                "total_load_current": record["total_load_current"],
                "total_power": record["total_power"],
                "load_percentage": round(load_percentage, 1),
                "available_capacity": record["available_capacity"],
                "status": self._get_circuit_status(load_percentage)
            })
        
        return {
            "vehicle_signature": vehicle_signature,
            "total_circuits": len(circuit_analytics),
            "summary": {
                "overloaded_circuits": overloaded_circuits,
                "heavily_loaded_circuits": heavily_loaded_circuits,
                "safety_critical_circuits": safety_critical_circuits,
                "circuits_without_protection": len([c for c in circuit_analytics if not c["protection_type"] or c["protection_type"] == "none"])
            },
            "circuit_details": circuit_analytics,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _get_circuit_status(self, load_percentage: float) -> str:
        """Determine circuit status based on load percentage"""
        if load_percentage > 100:
            return "overloaded"
        elif load_percentage > 90:
            return "critical"
        elif load_percentage > 80:
            return "heavily_loaded"
        elif load_percentage > 60:
            return "moderately_loaded"
        else:
            return "normal"
    
    # ==================== Power Flow Analytics ====================
    
    async def get_power_flow_analytics(self, vehicle_signature: str) -> Dict[str, Any]:
        """Analyze power flow throughout the system"""
        
        # Get power sources
        power_sources_query = """
        MATCH (ps:Component {vehicle_signature: $vehicle_signature})
        WHERE ps.type IN ['battery', 'alternator', 'power_supply']
        
        // Find what each power source powers
        OPTIONAL MATCH (load:Component)-[:POWERED_BY*1..5]->(ps)
        WHERE load.vehicle_signature = $vehicle_signature
        
        RETURN 
            ps.id as source_id,
            ps.name as source_name,
            ps.type as source_type,
            ps.voltage_rating as source_voltage,
            ps.current_rating as source_capacity,
            count(DISTINCT load) as loads_powered,
            sum(coalesce(load.current_rating, 0)) as total_load_current
        ORDER BY total_load_current DESC
        """
        
        sources_result = await self.neo4j.run(power_sources_query, vehicle_signature=vehicle_signature)
        
        power_sources = []
        total_system_capacity = 0
        total_system_load = 0
        
        for record in sources_result.records:
            source_capacity = record["source_capacity"] or 0
            source_load = record["total_load_current"] or 0
            
            total_system_capacity += source_capacity
            total_system_load += source_load
            
            utilization = (source_load / source_capacity * 100) if source_capacity > 0 else 0
            
            power_sources.append({
                "source_id": record["source_id"],
                "source_name": record["source_name"],
                "source_type": record["source_type"],
                "voltage": record["source_voltage"],
                "capacity": source_capacity,
                "current_load": source_load,
                "loads_powered": record["loads_powered"],
                "utilization_percent": round(utilization, 1),
                "status": "overloaded" if utilization > 100 else "high" if utilization > 80 else "normal"
            })
        
        # Analyze power distribution paths
        path_analysis_query = """
        MATCH (source:Component {vehicle_signature: $vehicle_signature})
        WHERE source.type IN ['battery', 'alternator']
        
        MATCH path = (source)-[:POWERED_BY|CONNECTS_TO*1..5]->(load:Component)
        WHERE load.vehicle_signature = $vehicle_signature
          AND load.type NOT IN ['battery', 'alternator']
        
        RETURN 
            length(path) as path_length,
            count(*) as path_count
        ORDER BY path_length
        """
        
        path_result = await self.neo4j.run(path_analysis_query, vehicle_signature=vehicle_signature)
        
        path_distribution = {}
        for record in path_result.records:
            path_length = record["path_length"]
            path_count = record["path_count"]
            path_distribution[f"{path_length}_hops"] = path_count
        
        # Calculate system efficiency
        system_efficiency = (total_system_load / total_system_capacity * 100) if total_system_capacity > 0 else 0
        
        return {
            "vehicle_signature": vehicle_signature,
            "power_sources": power_sources,
            "system_summary": {
                "total_capacity": total_system_capacity,
                "total_load": total_system_load,
                "system_utilization_percent": round(system_efficiency, 1),
                "power_source_count": len(power_sources),
                "redundancy_available": len(power_sources) > 1
            },
            "path_distribution": path_distribution,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # ==================== Data Quality Analytics ====================
    
    async def get_data_quality_report(self, vehicle_signature: str) -> Dict[str, Any]:
        """Generate comprehensive data quality report"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        RETURN 
            count(c) as total_components,
            
            // Position data quality
            count(CASE WHEN c.position IS NOT NULL THEN 1 END) as with_position,
            count(CASE WHEN c.zone IS NOT NULL THEN 1 END) as with_zone,
            
            // Electrical data quality
            count(CASE WHEN c.voltage_rating IS NOT NULL THEN 1 END) as with_voltage,
            count(CASE WHEN c.current_rating IS NOT NULL THEN 1 END) as with_current,
            count(CASE WHEN c.voltage_rating IS NOT NULL AND c.current_rating IS NOT NULL THEN 1 END) as with_complete_electrical,
            
            // Identification data quality
            count(CASE WHEN c.name IS NOT NULL AND c.name <> '' THEN 1 END) as with_name,
            count(CASE WHEN c.part_number IS NOT NULL AND c.part_number <> '' THEN 1 END) as with_part_number,
            count(CASE WHEN c.manufacturer IS NOT NULL AND c.manufacturer <> '' THEN 1 END) as with_manufacturer,
            
            // Relationship connectivity
            count(CASE WHEN NOT (c)-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]-() 
                      AND NOT ()-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]->(c) THEN 1 END) as orphaned_components
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        if not result.records:
            return {"error": "No data found"}
        
        record = result.records[0]
        total = record["total_components"]
        
        # Calculate quality percentages
        position_quality = (record["with_position"] / total * 100) if total > 0 else 0
        zone_quality = (record["with_zone"] / total * 100) if total > 0 else 0
        voltage_quality = (record["with_voltage"] / total * 100) if total > 0 else 0
        current_quality = (record["with_current"] / total * 100) if total > 0 else 0
        electrical_quality = (record["with_complete_electrical"] / total * 100) if total > 0 else 0
        name_quality = (record["with_name"] / total * 100) if total > 0 else 0
        part_number_quality = (record["with_part_number"] / total * 100) if total > 0 else 0
        manufacturer_quality = (record["with_manufacturer"] / total * 100) if total > 0 else 0
        
        # Calculate overall quality score
        quality_metrics = [
            position_quality, zone_quality, voltage_quality, current_quality,
            name_quality, part_number_quality, manufacturer_quality
        ]
        overall_quality = sum(quality_metrics) / len(quality_metrics)
        
        # Identify issues
        issues = []
        if position_quality < 80:
            issues.append(f"Only {position_quality:.1f}% of components have position data")
        if electrical_quality < 90:
            issues.append(f"Only {electrical_quality:.1f}% of components have complete electrical ratings")
        if record["orphaned_components"] > 0:
            issues.append(f"{record['orphaned_components']} components have no relationships")
        
        # Generate recommendations
        recommendations = []
        if position_quality < 50:
            recommendations.append("Prioritize spatial data mapping for 3D visualization")
        if electrical_quality < 70:
            recommendations.append("Complete electrical specifications for better analysis")
        if name_quality < 95:
            recommendations.append("Improve component naming for better identification")
        
        return {
            "vehicle_signature": vehicle_signature,
            "total_components": total,
            "overall_quality_score": round(overall_quality, 1),
            "quality_metrics": {
                "spatial_data": {
                    "position_coverage_percent": round(position_quality, 1),
                    "zone_coverage_percent": round(zone_quality, 1)
                },
                "electrical_data": {
                    "voltage_coverage_percent": round(voltage_quality, 1),
                    "current_coverage_percent": round(current_quality, 1),
                    "complete_electrical_percent": round(electrical_quality, 1)
                },
                "identification_data": {
                    "name_coverage_percent": round(name_quality, 1),
                    "part_number_coverage_percent": round(part_number_quality, 1),
                    "manufacturer_coverage_percent": round(manufacturer_quality, 1)
                },
                "connectivity": {
                    "orphaned_components": record["orphaned_components"],
                    "connected_components": total - record["orphaned_components"]
                }
            },
            "issues": issues,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # ==================== Trend Analytics ====================
    
    async def get_system_trends(
        self, 
        vehicle_signature: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Get system trends over time (requires timestamp tracking)"""
        
        start_date = datetime.utcnow() - timedelta(days=days_back)
        start_date_str = start_date.isoformat()
        
        # This is a placeholder implementation
        # In a real system, you'd track changes over time
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.created_at >= $start_date
        
        RETURN 
            date(c.created_at) as creation_date,
            count(c) as components_added,
            collect(DISTINCT c.type) as types_added
        ORDER BY creation_date
        """
        
        result = await self.neo4j.run(
            query, 
            vehicle_signature=vehicle_signature,
            start_date=start_date_str
        )
        
        daily_trends = []
        for record in result.records:
            daily_trends.append({
                "date": str(record["creation_date"]),
                "components_added": record["components_added"],
                "types_added": record["types_added"]
            })
        
        return {
            "vehicle_signature": vehicle_signature,
            "analysis_period_days": days_back,
            "daily_trends": daily_trends,
            "total_additions": sum(trend["components_added"] for trend in daily_trends),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # ==================== Custom Analytics ====================
    
    async def run_custom_analytics_query(
        self, 
        vehicle_signature: str,
        custom_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a custom analytics query"""
        
        # Security: Only allow read operations
        forbidden_keywords = ["CREATE", "DELETE", "SET", "REMOVE", "MERGE", "DROP"]
        query_upper = custom_query.upper()
        
        for keyword in forbidden_keywords:
            if keyword in query_upper:
                raise ValueError(f"Forbidden operation: {keyword}")
        
        # Ensure vehicle signature is included
        if "$vehicle_signature" not in custom_query:
            raise ValueError("Custom query must include $vehicle_signature parameter")
        
        params = parameters or {}
        params["vehicle_signature"] = vehicle_signature
        
        try:
            result = await self.neo4j.run(custom_query, **params)
            
            # Convert result to serializable format
            records = []
            for record in result.records:
                record_dict = {}
                for key in record.keys():
                    value = record[key]
                    # Convert Neo4j types to serializable types
                    if hasattr(value, '__dict__'):
                        record_dict[key] = dict(value)
                    else:
                        record_dict[key] = value
                records.append(record_dict)
            
            return {
                "vehicle_signature": vehicle_signature,
                "query": custom_query,
                "record_count": len(records),
                "records": records,
                "executed_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Custom query failed: {e}")
            return {
                "error": f"Query execution failed: {str(e)}",
                "vehicle_signature": vehicle_signature,
                "executed_at": datetime.utcnow().isoformat()
            }