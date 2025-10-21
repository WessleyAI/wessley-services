"""
Vehicle-specific operations and metadata management service
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.nodes.vehicle import VehicleNode
from ..models.nodes.zone import ZoneNode
from ..utils.neo4j_utils import Neo4jClient

logger = logging.getLogger(__name__)


class VehicleService:
    """
    Service for vehicle-specific operations and metadata management
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
    
    # ==================== Vehicle Management ====================
    
    async def create_vehicle(
        self,
        signature: str,
        make: str,
        model: str,
        year: int,
        engine: Optional[str] = None,
        market: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VehicleNode:
        """Create a new vehicle record"""
        
        vehicle_data = {
            "signature": signature,
            "make": make,
            "model": model,
            "year": year,
            "engine": engine,
            "market": market,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        query = """
        CREATE (v:Vehicle $props)
        RETURN v
        """
        
        result = await self.neo4j.run(query, props=vehicle_data)
        
        if not result.records:
            raise Exception("Failed to create vehicle")
        
        vehicle_node_data = dict(result.records[0]["v"])
        vehicle = VehicleNode.from_neo4j_dict(vehicle_node_data)
        
        logger.info(f"Created vehicle: {signature}")
        return vehicle
    
    async def get_vehicle(self, signature: str) -> Optional[VehicleNode]:
        """Get vehicle by signature"""
        
        query = """
        MATCH (v:Vehicle {signature: $signature})
        RETURN v
        """
        
        result = await self.neo4j.run(query, signature=signature)
        
        if not result.records:
            return None
        
        vehicle_data = dict(result.records[0]["v"])
        return VehicleNode.from_neo4j_dict(vehicle_data)
    
    async def update_vehicle(
        self,
        signature: str,
        updates: Dict[str, Any]
    ) -> VehicleNode:
        """Update vehicle metadata"""
        
        # Add update timestamp
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        # Build SET clause dynamically
        set_clauses = [f"v.{key} = ${key}" for key in updates.keys()]
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (v:Vehicle {{signature: $signature}})
        SET {set_clause}
        RETURN v
        """
        
        params = {"signature": signature, **updates}
        
        result = await self.neo4j.run(query, **params)
        
        if not result.records:
            raise Exception(f"Vehicle {signature} not found")
        
        vehicle_data = dict(result.records[0]["v"])
        return VehicleNode.from_neo4j_dict(vehicle_data)
    
    async def delete_vehicle(self, signature: str) -> bool:
        """Delete vehicle and all associated data"""
        
        query = """
        MATCH (v:Vehicle {signature: $signature})
        
        // Delete all components for this vehicle
        OPTIONAL MATCH (c:Component {vehicle_signature: $signature})
        DETACH DELETE c
        
        // Delete all circuits for this vehicle  
        OPTIONAL MATCH (circuit:Circuit {vehicle_signature: $signature})
        DETACH DELETE circuit
        
        // Delete all zones for this vehicle
        OPTIONAL MATCH (z:Zone {vehicle_signature: $signature})
        DETACH DELETE z
        
        // Delete the vehicle itself
        DETACH DELETE v
        
        RETURN count(v) as deleted_count
        """
        
        result = await self.neo4j.run(query, signature=signature)
        
        deleted_count = result.records[0]["deleted_count"] if result.records else 0
        
        if deleted_count > 0:
            logger.info(f"Deleted vehicle and all data: {signature}")
            return True
        
        return False
    
    async def list_vehicles(
        self,
        make: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 100
    ) -> List[VehicleNode]:
        """List vehicles with optional filtering"""
        
        where_clauses = []
        params = {}
        
        if make:
            where_clauses.append("v.make = $make")
            params["make"] = make
        
        if year:
            where_clauses.append("v.year = $year")
            params["year"] = year
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
        MATCH (v:Vehicle)
        WHERE {where_clause}
        RETURN v
        ORDER BY v.make, v.model, v.year
        LIMIT {limit}
        """
        
        result = await self.neo4j.run(query, **params)
        
        vehicles = []
        for record in result.records:
            vehicle_data = dict(record["v"])
            vehicles.append(VehicleNode.from_neo4j_dict(vehicle_data))
        
        return vehicles
    
    # ==================== Zone Management ====================
    
    async def create_zone(
        self,
        zone_id: str,
        vehicle_signature: str,
        name: str,
        bounds: Optional[Dict[str, float]] = None,
        access_level: str = "moderate",
        environmental_conditions: Optional[Dict[str, Any]] = None
    ) -> ZoneNode:
        """Create a new zone for a vehicle"""
        
        zone_data = {
            "id": zone_id,
            "vehicle_signature": vehicle_signature,
            "name": name,
            "bounds": bounds or {},
            "access_level": access_level,
            "environmental_conditions": environmental_conditions or {},
            "created_at": datetime.utcnow().isoformat()
        }
        
        query = """
        CREATE (z:Zone $props)
        RETURN z
        """
        
        result = await self.neo4j.run(query, props=zone_data)
        
        if not result.records:
            raise Exception("Failed to create zone")
        
        zone_node_data = dict(result.records[0]["z"])
        zone = ZoneNode.from_neo4j_dict(zone_node_data)
        
        logger.info(f"Created zone: {zone_id} for vehicle {vehicle_signature}")
        return zone
    
    async def get_vehicle_zones(self, vehicle_signature: str) -> List[ZoneNode]:
        """Get all zones for a vehicle"""
        
        query = """
        MATCH (z:Zone {vehicle_signature: $vehicle_signature})
        RETURN z
        ORDER BY z.name
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        zones = []
        for record in result.records:
            zone_data = dict(record["z"])
            zones.append(ZoneNode.from_neo4j_dict(zone_data))
        
        return zones
    
    async def create_standard_zones(self, vehicle_signature: str) -> List[ZoneNode]:
        """Create standard automotive zones for a vehicle"""
        
        standard_zones = [
            {
                "id": "engine_bay",
                "name": "Engine Bay",
                "bounds": {"min_x": -1000, "max_x": 1000, "min_y": -500, "max_y": 500, "min_z": 0, "max_z": 800},
                "access_level": "moderate",
                "environmental_conditions": {
                    "temperature_range": {"min": -20, "max": 120},
                    "vibration_level": "high",
                    "moisture_exposure": "moderate"
                }
            },
            {
                "id": "dashboard",
                "name": "Dashboard",
                "bounds": {"min_x": -800, "max_x": 800, "min_y": -300, "max_y": 100, "min_z": 800, "max_z": 1200},
                "access_level": "easy",
                "environmental_conditions": {
                    "temperature_range": {"min": -10, "max": 70},
                    "vibration_level": "low",
                    "moisture_exposure": "low"
                }
            },
            {
                "id": "cabin",
                "name": "Passenger Cabin",
                "bounds": {"min_x": -800, "max_x": 800, "min_y": -200, "max_y": 800, "min_z": 600, "max_z": 1400},
                "access_level": "easy",
                "environmental_conditions": {
                    "temperature_range": {"min": -10, "max": 50},
                    "vibration_level": "low",
                    "moisture_exposure": "very_low"
                }
            },
            {
                "id": "trunk",
                "name": "Trunk/Cargo Area",
                "bounds": {"min_x": -600, "max_x": 600, "min_y": -800, "max_y": -400, "min_z": 0, "max_z": 600},
                "access_level": "easy",
                "environmental_conditions": {
                    "temperature_range": {"min": -20, "max": 60},
                    "vibration_level": "moderate",
                    "moisture_exposure": "moderate"
                }
            },
            {
                "id": "underhood_fuse_box",
                "name": "Under-hood Fuse Box",
                "bounds": {"min_x": -200, "max_x": 200, "min_y": 200, "max_y": 400, "min_z": 600, "max_z": 700},
                "access_level": "moderate",
                "environmental_conditions": {
                    "temperature_range": {"min": -20, "max": 100},
                    "vibration_level": "moderate",
                    "moisture_exposure": "moderate"
                }
            },
            {
                "id": "interior_fuse_box",
                "name": "Interior Fuse Box",
                "bounds": {"min_x": -300, "max_x": -200, "min_y": 0, "max_y": 100, "min_z": 900, "max_z": 1000},
                "access_level": "easy",
                "environmental_conditions": {
                    "temperature_range": {"min": -10, "max": 60},
                    "vibration_level": "low",
                    "moisture_exposure": "very_low"
                }
            }
        ]
        
        created_zones = []
        
        for zone_spec in standard_zones:
            try:
                zone = await self.create_zone(
                    zone_id=zone_spec["id"],
                    vehicle_signature=vehicle_signature,
                    name=zone_spec["name"],
                    bounds=zone_spec["bounds"],
                    access_level=zone_spec["access_level"],
                    environmental_conditions=zone_spec["environmental_conditions"]
                )
                created_zones.append(zone)
            except Exception as e:
                logger.warning(f"Failed to create zone {zone_spec['id']}: {e}")
        
        return created_zones
    
    # ==================== Vehicle Statistics ====================
    
    async def get_vehicle_statistics(self, vehicle_signature: str) -> Dict[str, Any]:
        """Get comprehensive vehicle statistics"""
        
        query = """
        MATCH (v:Vehicle {signature: $vehicle_signature})
        
        // Count components
        OPTIONAL MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Count circuits
        OPTIONAL MATCH (circuit:Circuit {vehicle_signature: $vehicle_signature})
        
        // Count zones
        OPTIONAL MATCH (z:Zone {vehicle_signature: $vehicle_signature})
        
        // Count connections
        OPTIONAL MATCH ()-[conn:CONNECTS_TO]->()
        WHERE (startNode(conn).vehicle_signature = $vehicle_signature AND 
               endNode(conn).vehicle_signature = $vehicle_signature)
        
        // Component type distribution
        WITH v, collect(DISTINCT c) as components, collect(DISTINCT circuit) as circuits,
             collect(DISTINCT z) as zones, collect(DISTINCT conn) as connections
        
        RETURN 
            v,
            size(components) as component_count,
            size(circuits) as circuit_count,
            size(zones) as zone_count,
            size(connections) as connection_count,
            [comp in components | comp.type] as component_types,
            [comp in components | comp.zone] as component_zones,
            [comp in components WHERE comp.position IS NOT NULL | comp.id] as positioned_components
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        if not result.records:
            return {"error": "Vehicle not found"}
        
        record = result.records[0]
        vehicle_data = dict(record["v"])
        
        # Analyze component types
        component_types = record["component_types"]
        type_distribution = {}
        for comp_type in component_types:
            if comp_type:
                type_distribution[comp_type] = type_distribution.get(comp_type, 0) + 1
        
        # Analyze zones
        component_zones = record["component_zones"]
        zone_distribution = {}
        for zone in component_zones:
            if zone:
                zone_distribution[zone] = zone_distribution.get(zone, 0) + 1
        
        # Calculate data completeness
        total_components = record["component_count"]
        positioned_components = len(record["positioned_components"])
        position_completeness = (positioned_components / total_components * 100) if total_components > 0 else 0
        
        return {
            "vehicle": {
                "signature": vehicle_data["signature"],
                "make": vehicle_data["make"],
                "model": vehicle_data["model"],
                "year": vehicle_data["year"],
                "engine": vehicle_data.get("engine"),
                "market": vehicle_data.get("market")
            },
            "counts": {
                "components": record["component_count"],
                "circuits": record["circuit_count"],
                "zones": record["zone_count"],
                "connections": record["connection_count"]
            },
            "distributions": {
                "component_types": type_distribution,
                "zone_usage": zone_distribution
            },
            "data_quality": {
                "position_completeness_percent": position_completeness,
                "positioned_components": positioned_components,
                "total_components": total_components
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # ==================== Vehicle Comparison ====================
    
    async def compare_vehicles(
        self,
        vehicle_signatures: List[str]
    ) -> Dict[str, Any]:
        """Compare multiple vehicles"""
        
        if len(vehicle_signatures) < 2:
            raise ValueError("At least 2 vehicles required for comparison")
        
        vehicle_stats = {}
        
        # Get statistics for each vehicle
        for signature in vehicle_signatures:
            stats = await self.get_vehicle_statistics(signature)
            if "error" not in stats:
                vehicle_stats[signature] = stats
        
        if len(vehicle_stats) < 2:
            return {"error": "Insufficient valid vehicles for comparison"}
        
        # Perform comparison analysis
        comparison = {
            "vehicles_compared": list(vehicle_stats.keys()),
            "comparison_metrics": {},
            "similarities": [],
            "differences": [],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Compare component counts
        component_counts = {sig: stats["counts"]["components"] for sig, stats in vehicle_stats.items()}
        comparison["comparison_metrics"]["component_counts"] = component_counts
        
        # Compare circuit counts
        circuit_counts = {sig: stats["counts"]["circuits"] for sig, stats in vehicle_stats.items()}
        comparison["comparison_metrics"]["circuit_counts"] = circuit_counts
        
        # Compare data quality
        position_completeness = {sig: stats["data_quality"]["position_completeness_percent"] 
                               for sig, stats in vehicle_stats.items()}
        comparison["comparison_metrics"]["position_completeness"] = position_completeness
        
        # Identify similarities and differences
        vehicles = list(vehicle_stats.keys())
        
        # Check component type similarities
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                
                types1 = set(vehicle_stats[v1]["distributions"]["component_types"].keys())
                types2 = set(vehicle_stats[v2]["distributions"]["component_types"].keys())
                
                common_types = types1.intersection(types2)
                if len(common_types) > 5:
                    comparison["similarities"].append({
                        "vehicles": [v1, v2],
                        "type": "component_types",
                        "description": f"Share {len(common_types)} component types"
                    })
                
                unique_v1 = types1 - types2
                unique_v2 = types2 - types1
                
                if unique_v1:
                    comparison["differences"].append({
                        "vehicle": v1,
                        "type": "unique_component_types",
                        "description": f"Has unique types: {', '.join(unique_v1)}"
                    })
                
                if unique_v2:
                    comparison["differences"].append({
                        "vehicle": v2,
                        "type": "unique_component_types",
                        "description": f"Has unique types: {', '.join(unique_v2)}"
                    })
        
        return comparison
    
    # ==================== Vehicle Data Validation ====================
    
    async def validate_vehicle_data(self, vehicle_signature: str) -> Dict[str, Any]:
        """Validate vehicle data integrity and completeness"""
        
        validation_result = {
            "vehicle_signature": vehicle_signature,
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "validation_timestamp": datetime.utcnow().isoformat()
        }
        
        # Check if vehicle exists
        vehicle = await self.get_vehicle(vehicle_signature)
        if not vehicle:
            validation_result["is_valid"] = False
            validation_result["issues"].append("Vehicle record not found")
            return validation_result
        
        # Check for orphaned components
        orphan_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE NOT (c)-[:CONNECTS_TO|POWERED_BY|CONTROLS]-() 
          AND NOT ()-[:CONNECTS_TO|POWERED_BY|CONTROLS]->(c)
        RETURN count(c) as orphan_count, collect(c.id) as orphan_ids
        """
        
        orphan_result = await self.neo4j.run(orphan_query, vehicle_signature=vehicle_signature)
        if orphan_result.records:
            orphan_count = orphan_result.records[0]["orphan_count"]
            if orphan_count > 0:
                validation_result["warnings"].append(
                    f"Found {orphan_count} orphaned components with no connections"
                )
                validation_result["recommendations"].append(
                    "Review isolated components and establish proper connections"
                )
        
        # Check for missing spatial data
        spatial_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        RETURN 
            count(c) as total_components,
            count(CASE WHEN c.position IS NULL THEN 1 END) as missing_position,
            count(CASE WHEN c.zone IS NULL THEN 1 END) as missing_zone
        """
        
        spatial_result = await self.neo4j.run(spatial_query, vehicle_signature=vehicle_signature)
        if spatial_result.records:
            record = spatial_result.records[0]
            total = record["total_components"]
            missing_pos = record["missing_position"]
            missing_zone = record["missing_zone"]
            
            if missing_pos > 0:
                percentage = (missing_pos / total * 100) if total > 0 else 0
                validation_result["warnings"].append(
                    f"{missing_pos} components ({percentage:.1f}%) missing position data"
                )
                if percentage > 50:
                    validation_result["recommendations"].append(
                        "Complete spatial mapping for better 3D visualization"
                    )
            
            if missing_zone > 0:
                percentage = (missing_zone / total * 100) if total > 0 else 0
                validation_result["warnings"].append(
                    f"{missing_zone} components ({percentage:.1f}%) missing zone assignment"
                )
        
        # Check for circuits without protection
        protection_query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        WHERE NOT exists(c.protection) OR c.protection = 'none'
        RETURN count(c) as unprotected_count
        """
        
        protection_result = await self.neo4j.run(protection_query, vehicle_signature=vehicle_signature)
        if protection_result.records:
            unprotected = protection_result.records[0]["unprotected_count"]
            if unprotected > 0:
                validation_result["issues"].append(
                    f"{unprotected} circuits lack proper protection devices"
                )
                validation_result["is_valid"] = False
        
        # Check for unrealistic electrical values
        electrical_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.voltage_rating > 1000 OR c.current_rating > 1000
        RETURN count(c) as unrealistic_count, collect(c.id) as unrealistic_ids
        """
        
        electrical_result = await self.neo4j.run(electrical_query, vehicle_signature=vehicle_signature)
        if electrical_result.records:
            unrealistic = electrical_result.records[0]["unrealistic_count"]
            if unrealistic > 0:
                validation_result["warnings"].append(
                    f"{unrealistic} components have unrealistic electrical ratings"
                )
                validation_result["recommendations"].append(
                    "Review electrical specifications for accuracy"
                )
        
        return validation_result
    
    # ==================== Vehicle Export ====================
    
    async def export_vehicle_data(
        self,
        vehicle_signature: str,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export complete vehicle data"""
        
        if format not in ["json", "graphml"]:
            raise ValueError("Supported formats: json, graphml")
        
        # Get vehicle metadata
        vehicle = await self.get_vehicle(vehicle_signature)
        if not vehicle:
            raise ValueError("Vehicle not found")
        
        # Get all vehicle data
        export_data = {
            "vehicle": vehicle.to_neo4j_dict(),
            "export_metadata": {
                "format": format,
                "exported_at": datetime.utcnow().isoformat(),
                "exporter": "graph-service"
            }
        }
        
        # Get components
        components_query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        RETURN collect(c) as components
        """
        
        components_result = await self.neo4j.run(components_query, vehicle_signature=vehicle_signature)
        if components_result.records:
            components = [dict(comp) for comp in components_result.records[0]["components"]]
            export_data["components"] = components
        
        # Get circuits
        circuits_query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        RETURN collect(c) as circuits
        """
        
        circuits_result = await self.neo4j.run(circuits_query, vehicle_signature=vehicle_signature)
        if circuits_result.records:
            circuits = [dict(circuit) for circuit in circuits_result.records[0]["circuits"]]
            export_data["circuits"] = circuits
        
        # Get zones
        zones = await self.get_vehicle_zones(vehicle_signature)
        export_data["zones"] = [zone.to_neo4j_dict() for zone in zones]
        
        # Get relationships
        relationships_query = """
        MATCH (c1:Component {vehicle_signature: $vehicle_signature})-[r]->(c2:Component {vehicle_signature: $vehicle_signature})
        RETURN 
            c1.id as from_id,
            c2.id as to_id,
            type(r) as relationship_type,
            properties(r) as relationship_properties
        """
        
        relationships_result = await self.neo4j.run(relationships_query, vehicle_signature=vehicle_signature)
        relationships = []
        for record in relationships_result.records:
            relationships.append({
                "from_id": record["from_id"],
                "to_id": record["to_id"],
                "type": record["relationship_type"],
                "properties": dict(record["relationship_properties"])
            })
        
        export_data["relationships"] = relationships
        
        return export_data