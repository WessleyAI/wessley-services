"""
Advanced query service for complex graph analysis and traversal operations
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from dataclasses import dataclass

from ..utils.neo4j_utils import Neo4jClient
from ..utils.cypher_builder import CypherQueryBuilder

logger = logging.getLogger(__name__)


@dataclass
class PathAnalysisResult:
    """Result of path analysis between components"""
    source_id: str
    target_id: str
    path_found: bool
    path_length: Optional[int]
    path_nodes: List[str]
    path_relationships: List[str]
    total_resistance: Optional[float]
    voltage_drop: Optional[float]
    analysis_metadata: Dict[str, Any]


@dataclass
class CircuitAnalysisResult:
    """Result of circuit analysis"""
    circuit_id: str
    circuit_name: str
    total_components: int
    power_sources: List[str]
    loads: List[str]
    protection_devices: List[str]
    total_current_draw: float
    voltage_drop_percentage: float
    load_factor: float
    reliability_score: float
    potential_issues: List[str]


@dataclass
class PowerDistributionAnalysis:
    """Power distribution analysis result"""
    vehicle_signature: str
    power_sources: List[Dict[str, Any]]
    distribution_tree: Dict[str, Any]
    load_analysis: Dict[str, Any]
    efficiency_metrics: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]


class QueryService:
    """
    Advanced query service for complex electrical system analysis
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.query_builder = CypherQueryBuilder()
    
    # ==================== Path Analysis ====================
    
    async def find_shortest_electrical_path(
        self,
        source_id: str,
        target_id: str,
        vehicle_signature: str,
        max_depth: int = 15,
        relationship_types: Optional[List[str]] = None
    ) -> PathAnalysisResult:
        """Find the shortest electrical path between two components"""
        
        rel_types = relationship_types or ["CONNECTS_TO", "POWERED_BY"]
        rel_filter = "|".join(rel_types)
        
        query = f"""
        MATCH (source:Component {{id: $source_id, vehicle_signature: $vehicle_signature}})
        MATCH (target:Component {{id: $target_id, vehicle_signature: $vehicle_signature}})
        
        CALL {{
            WITH source, target
            MATCH path = shortestPath((source)-[:{rel_filter}*1..{max_depth}]-(target))
            RETURN path
            ORDER BY length(path)
            LIMIT 1
        }}
        
        WITH path, 
             [node in nodes(path) | node.id] as node_ids,
             [rel in relationships(path) | type(rel)] as rel_types,
             [rel in relationships(path) WHERE rel.wire_gauge IS NOT NULL | rel.wire_gauge] as wire_gauges
        
        // Calculate total resistance (simplified)
        WITH path, node_ids, rel_types, 
             reduce(resistance = 0.0, rel in relationships(path) | 
                 resistance + coalesce(rel.resistance, 0.1)) as total_resistance
        
        RETURN 
            path,
            length(path) as path_length,
            node_ids,
            rel_types,
            total_resistance,
            nodes(path) as path_nodes,
            relationships(path) as path_relationships
        """
        
        result = await self.neo4j.run(
            query,
            source_id=source_id,
            target_id=target_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return PathAnalysisResult(
                source_id=source_id,
                target_id=target_id,
                path_found=False,
                path_length=None,
                path_nodes=[],
                path_relationships=[],
                total_resistance=None,
                voltage_drop=None,
                analysis_metadata={"no_path_found": True}
            )
        
        record = result.records[0]
        
        return PathAnalysisResult(
            source_id=source_id,
            target_id=target_id,
            path_found=True,
            path_length=record["path_length"],
            path_nodes=record["node_ids"],
            path_relationships=record["rel_types"],
            total_resistance=record["total_resistance"],
            voltage_drop=self._calculate_voltage_drop(record["total_resistance"], 12.0, 10.0),
            analysis_metadata={
                "path_complexity": record["path_length"],
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def find_all_paths_between_components(
        self,
        source_id: str,
        target_id: str,
        vehicle_signature: str,
        max_paths: int = 5,
        max_depth: int = 10
    ) -> List[PathAnalysisResult]:
        """Find multiple paths between two components"""
        
        query = f"""
        MATCH (source:Component {{id: $source_id, vehicle_signature: $vehicle_signature}})
        MATCH (target:Component {{id: $target_id, vehicle_signature: $vehicle_signature}})
        
        CALL {{
            WITH source, target
            MATCH path = (source)-[:CONNECTS_TO|POWERED_BY*1..{max_depth}]-(target)
            WHERE length(path) <= {max_depth}
            RETURN path
            ORDER BY length(path)
            LIMIT {max_paths}
        }}
        
        WITH path,
             [node in nodes(path) | node.id] as node_ids,
             [rel in relationships(path) | type(rel)] as rel_types
        
        RETURN 
            path,
            length(path) as path_length,
            node_ids,
            rel_types
        ORDER BY path_length
        """
        
        result = await self.neo4j.run(
            query,
            source_id=source_id,
            target_id=target_id,
            vehicle_signature=vehicle_signature
        )
        
        paths = []
        for record in result.records:
            paths.append(PathAnalysisResult(
                source_id=source_id,
                target_id=target_id,
                path_found=True,
                path_length=record["path_length"],
                path_nodes=record["node_ids"],
                path_relationships=record["rel_types"],
                total_resistance=None,
                voltage_drop=None,
                analysis_metadata={"path_index": len(paths)}
            ))
        
        return paths
    
    # ==================== Circuit Analysis ====================
    
    async def analyze_circuit_comprehensive(
        self,
        circuit_id: str,
        vehicle_signature: str
    ) -> CircuitAnalysisResult:
        """Perform comprehensive circuit analysis"""
        
        query = """
        MATCH (circuit:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        
        // Get all components in the circuit
        OPTIONAL MATCH (circuit)<-[:PART_OF]-(component:Component)
        
        // Categorize components by role
        WITH circuit, collect(component) as all_components,
             [c IN collect(component) WHERE c.type IN ['battery', 'alternator', 'power_supply'] | c.id] as power_sources,
             [c IN collect(component) WHERE c.type NOT IN ['battery', 'alternator', 'power_supply', 'fuse', 'relay'] | c.id] as loads,
             [c IN collect(component) WHERE c.type IN ['fuse', 'circuit_breaker'] | c.id] as protection_devices
        
        // Calculate total current draw
        WITH circuit, all_components, power_sources, loads, protection_devices,
             reduce(total_current = 0.0, c in all_components | 
                 total_current + coalesce(c.current_rating, 0.0)) as total_current_draw
        
        // Calculate load factor
        WITH circuit, all_components, power_sources, loads, protection_devices, total_current_draw,
             CASE 
                 WHEN circuit.max_current > 0 THEN (total_current_draw / circuit.max_current) * 100
                 ELSE 0 
             END as load_factor
        
        // Calculate reliability score (simplified)
        WITH circuit, all_components, power_sources, loads, protection_devices, 
             total_current_draw, load_factor,
             CASE 
                 WHEN size(protection_devices) > 0 AND load_factor < 80 THEN 0.9
                 WHEN size(protection_devices) > 0 THEN 0.7
                 WHEN load_factor < 80 THEN 0.6
                 ELSE 0.4
             END as reliability_score
        
        RETURN 
            circuit.id as circuit_id,
            circuit.name as circuit_name,
            size(all_components) as total_components,
            power_sources,
            loads,
            protection_devices,
            total_current_draw,
            5.0 as voltage_drop_percentage,  // Placeholder calculation
            load_factor,
            reliability_score
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            raise ValueError(f"Circuit {circuit_id} not found")
        
        record = result.records[0]
        
        # Identify potential issues
        potential_issues = []
        load_factor = record["load_factor"]
        
        if load_factor > 100:
            potential_issues.append("Circuit overloaded - load exceeds capacity")
        elif load_factor > 80:
            potential_issues.append("Circuit heavily loaded - consider capacity upgrade")
        
        if not record["protection_devices"]:
            potential_issues.append("No protection devices found in circuit")
        
        if record["voltage_drop_percentage"] > 5:
            potential_issues.append("Excessive voltage drop detected")
        
        return CircuitAnalysisResult(
            circuit_id=record["circuit_id"],
            circuit_name=record["circuit_name"],
            total_components=record["total_components"],
            power_sources=record["power_sources"],
            loads=record["loads"],
            protection_devices=record["protection_devices"],
            total_current_draw=record["total_current_draw"],
            voltage_drop_percentage=record["voltage_drop_percentage"],
            load_factor=load_factor,
            reliability_score=record["reliability_score"],
            potential_issues=potential_issues
        )
    
    async def get_circuit_topology(
        self,
        circuit_id: str,
        vehicle_signature: str
    ) -> Dict[str, Any]:
        """Get circuit topology with component relationships"""
        
        query = """
        MATCH (circuit:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        MATCH (circuit)<-[:PART_OF]-(component:Component)
        
        // Get connections between components in the circuit
        OPTIONAL MATCH (component)-[rel:CONNECTS_TO]->(connected:Component)-[:PART_OF]->(circuit)
        
        // Get power relationships
        OPTIONAL MATCH (component)-[power_rel:POWERED_BY]->(power_source:Component)
        
        // Get control relationships  
        OPTIONAL MATCH (component)-[control_rel:CONTROLS]->(controlled:Component)-[:PART_OF]->(circuit)
        
        RETURN 
            circuit,
            collect(DISTINCT component) as components,
            collect(DISTINCT {
                from: component.id,
                to: connected.id,
                type: 'CONNECTS_TO',
                properties: rel
            }) as connections,
            collect(DISTINCT {
                from: power_source.id,
                to: component.id, 
                type: 'POWERED_BY',
                properties: power_rel
            }) as power_relationships,
            collect(DISTINCT {
                from: component.id,
                to: controlled.id,
                type: 'CONTROLS', 
                properties: control_rel
            }) as control_relationships
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return {}
        
        record = result.records[0]
        
        return {
            "circuit": dict(record["circuit"]),
            "components": [dict(comp) for comp in record["components"]],
            "topology": {
                "connections": [conn for conn in record["connections"] if conn["to"] is not None],
                "power_relationships": [rel for rel in record["power_relationships"] if rel["from"] is not None],
                "control_relationships": [rel for rel in record["control_relationships"] if rel["to"] is not None]
            },
            "analysis_metadata": {
                "component_count": len(record["components"]),
                "connection_count": len([c for c in record["connections"] if c["to"] is not None]),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    # ==================== Power Distribution Analysis ====================
    
    async def analyze_power_distribution(
        self,
        vehicle_signature: str,
        include_load_analysis: bool = True
    ) -> PowerDistributionAnalysis:
        """Analyze complete power distribution system"""
        
        # Get power sources
        power_sources_query = """
        MATCH (ps:Component {vehicle_signature: $vehicle_signature})
        WHERE ps.type IN ['battery', 'alternator', 'power_supply']
        RETURN collect({
            id: ps.id,
            type: ps.type,
            voltage_rating: ps.voltage_rating,
            current_rating: ps.current_rating,
            specifications: ps.specifications
        }) as power_sources
        """
        
        power_result = await self.neo4j.run(power_sources_query, vehicle_signature=vehicle_signature)
        power_sources = power_result.records[0]["power_sources"] if power_result.records else []
        
        # Build distribution tree
        distribution_tree = await self._build_power_distribution_tree(vehicle_signature)
        
        # Analyze loads if requested
        load_analysis = {}
        if include_load_analysis:
            load_analysis = await self._analyze_system_loads(vehicle_signature)
        
        # Calculate efficiency metrics
        efficiency_metrics = await self._calculate_power_efficiency(vehicle_signature)
        
        # Identify bottlenecks
        bottlenecks = await self._identify_power_bottlenecks(vehicle_signature)
        
        # Generate recommendations
        recommendations = self._generate_power_recommendations(
            power_sources, load_analysis, efficiency_metrics, bottlenecks
        )
        
        return PowerDistributionAnalysis(
            vehicle_signature=vehicle_signature,
            power_sources=power_sources,
            distribution_tree=distribution_tree,
            load_analysis=load_analysis,
            efficiency_metrics=efficiency_metrics,
            bottlenecks=bottlenecks,
            recommendations=recommendations
        )
    
    # ==================== System Health Analysis ====================
    
    async def analyze_system_health(
        self,
        vehicle_signature: str
    ) -> Dict[str, Any]:
        """Comprehensive system health analysis"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Component health metrics
        WITH collect(c) as all_components,
             [comp IN collect(c) WHERE comp.position IS NULL | comp.id] as components_without_position,
             [comp IN collect(c) WHERE comp.voltage_rating IS NULL | comp.id] as components_without_voltage,
             [comp IN collect(c) WHERE comp.current_rating IS NULL | comp.id] as components_without_current
        
        // Connection analysis
        OPTIONAL MATCH ()-[r:CONNECTS_TO {vehicle_signature: $vehicle_signature}]->()
        WITH all_components, components_without_position, components_without_voltage, components_without_current,
             collect(r) as all_connections
        
        // Orphaned components (no connections)
        OPTIONAL MATCH (orphan:Component {vehicle_signature: $vehicle_signature})
        WHERE NOT (orphan)-[:CONNECTS_TO|POWERED_BY|CONTROLS]-() 
          AND NOT ()-[:CONNECTS_TO|POWERED_BY|CONTROLS]->(orphan)
        
        RETURN 
            size(all_components) as total_components,
            size(components_without_position) as missing_position_count,
            size(components_without_voltage) as missing_voltage_count,
            size(components_without_current) as missing_current_count,
            size(all_connections) as total_connections,
            collect(orphan.id) as orphaned_components,
            components_without_position,
            components_without_voltage,
            components_without_current
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        if not result.records:
            return {"status": "no_data", "vehicle_signature": vehicle_signature}
        
        record = result.records[0]
        
        # Calculate health score
        total_components = record["total_components"]
        issues_count = (
            record["missing_position_count"] +
            record["missing_voltage_count"] + 
            record["missing_current_count"] +
            len(record["orphaned_components"])
        )
        
        health_score = max(0, (total_components - issues_count) / total_components * 100) if total_components > 0 else 0
        
        # Determine health status
        if health_score >= 90:
            health_status = "excellent"
        elif health_score >= 75:
            health_status = "good"
        elif health_score >= 50:
            health_status = "fair"
        else:
            health_status = "poor"
        
        return {
            "vehicle_signature": vehicle_signature,
            "health_score": health_score,
            "health_status": health_status,
            "total_components": total_components,
            "total_connections": record["total_connections"],
            "data_quality": {
                "missing_position": {
                    "count": record["missing_position_count"],
                    "components": record["components_without_position"]
                },
                "missing_voltage": {
                    "count": record["missing_voltage_count"],
                    "components": record["components_without_voltage"]
                },
                "missing_current": {
                    "count": record["missing_current_count"],
                    "components": record["components_without_current"]
                }
            },
            "connectivity_issues": {
                "orphaned_components": record["orphaned_components"]
            },
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    # ==================== Helper Methods ====================
    
    def _calculate_voltage_drop(self, resistance: float, voltage: float, current: float) -> float:
        """Calculate voltage drop using Ohm's law"""
        if resistance and current:
            return (resistance * current / 1000)  # Convert to volts
        return 0.0
    
    async def _build_power_distribution_tree(self, vehicle_signature: str) -> Dict[str, Any]:
        """Build hierarchical power distribution tree"""
        
        query = """
        MATCH (source:Component {vehicle_signature: $vehicle_signature})
        WHERE source.type IN ['battery', 'alternator']
        
        OPTIONAL MATCH path = (source)-[:POWERED_BY|CONNECTS_TO*0..5]->(load:Component)
        WHERE load.vehicle_signature = $vehicle_signature
        
        RETURN source.id as source_id, collect(DISTINCT load.id) as powered_components
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        tree = {}
        for record in result.records:
            tree[record["source_id"]] = record["powered_components"]
        
        return tree
    
    async def _analyze_system_loads(self, vehicle_signature: str) -> Dict[str, Any]:
        """Analyze system electrical loads"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.current_rating IS NOT NULL AND c.type NOT IN ['battery', 'alternator']
        
        RETURN 
            sum(c.current_rating) as total_load_current,
            avg(c.current_rating) as average_load_current,
            max(c.current_rating) as peak_load_current,
            count(c) as load_component_count,
            collect({type: c.type, current: c.current_rating}) as load_breakdown
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        if not result.records:
            return {}
        
        record = result.records[0]
        return {
            "total_current": record["total_load_current"],
            "average_current": record["average_load_current"], 
            "peak_current": record["peak_load_current"],
            "component_count": record["load_component_count"],
            "breakdown_by_type": record["load_breakdown"]
        }
    
    async def _calculate_power_efficiency(self, vehicle_signature: str) -> Dict[str, Any]:
        """Calculate power system efficiency metrics"""
        
        # Simplified efficiency calculation
        return {
            "overall_efficiency": 0.85,  # Placeholder
            "distribution_losses": 0.05,
            "conversion_losses": 0.10,
            "estimated_by": "simplified_model"
        }
    
    async def _identify_power_bottlenecks(self, vehicle_signature: str) -> List[Dict[str, Any]]:
        """Identify potential power distribution bottlenecks"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.current_rating IS NOT NULL
        
        // Find heavily loaded components (>80% of rating)
        OPTIONAL MATCH (heavily_loaded:Component)-[r:POWERED_BY]->(c)
        WHERE heavily_loaded.current_rating > c.current_rating * 0.8
        
        RETURN c.id as component_id, c.type as component_type, 
               c.current_rating as rating,
               collect(heavily_loaded.id) as heavy_loads
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        bottlenecks = []
        for record in result.records:
            if record["heavy_loads"]:
                bottlenecks.append({
                    "component_id": record["component_id"],
                    "type": "high_load",
                    "description": f"Component {record['component_id']} heavily loaded",
                    "affected_loads": record["heavy_loads"]
                })
        
        return bottlenecks
    
    def _generate_power_recommendations(
        self,
        power_sources: List[Dict],
        load_analysis: Dict,
        efficiency_metrics: Dict,
        bottlenecks: List[Dict]
    ) -> List[str]:
        """Generate power system recommendations"""
        
        recommendations = []
        
        if not power_sources:
            recommendations.append("Add primary power source (battery/alternator)")
        
        if load_analysis.get("total_current", 0) > 100:
            recommendations.append("Consider power distribution optimization for high current loads")
        
        if bottlenecks:
            recommendations.append("Address identified power bottlenecks to improve reliability")
        
        if efficiency_metrics.get("overall_efficiency", 1) < 0.8:
            recommendations.append("Improve power system efficiency through component upgrades")
        
        return recommendations