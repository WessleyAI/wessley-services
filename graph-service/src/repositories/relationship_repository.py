"""
Relationship repository for managing electrical system relationships
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..models.relationships import (
    ConnectsToRelationship,
    PoweredByRelationship,
    ControlsRelationship,
    LocatedInRelationship,
    PartOfRelationship
)
from ..utils.neo4j_utils import Neo4jClient

logger = logging.getLogger(__name__)


class RelationshipRepository:
    """
    Repository for managing all types of electrical system relationships
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
    
    # ==================== CONNECTS_TO Relationships ====================
    
    async def create_connection(
        self,
        from_component_id: str,
        to_component_id: str,
        vehicle_signature: str,
        connection_props: ConnectsToRelationship
    ) -> bool:
        """Create a CONNECTS_TO relationship between components"""
        
        query = """
        MATCH (from:Component {id: $from_id, vehicle_signature: $vehicle_signature})
        MATCH (to:Component {id: $to_id, vehicle_signature: $vehicle_signature})
        CREATE (from)-[r:CONNECTS_TO $props]->(to)
        RETURN r
        """
        
        result = await self.neo4j.run(
            query,
            from_id=from_component_id,
            to_id=to_component_id,
            vehicle_signature=vehicle_signature,
            props=connection_props.to_neo4j_dict()
        )
        
        if result.records:
            logger.info(f"Created connection: {from_component_id} -> {to_component_id}")
            return True
        
        return False
    
    async def get_connections(
        self,
        component_id: str,
        vehicle_signature: str,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Dict[str, Any]]:
        """Get all CONNECTS_TO relationships for a component"""
        
        if direction == "outgoing":
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            MATCH (c)-[r:CONNECTS_TO]->(connected:Component)
            WHERE connected.vehicle_signature = $vehicle_signature
            RETURN 'outgoing' as direction, r, connected
            """
        elif direction == "incoming":
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            MATCH (connected:Component)-[r:CONNECTS_TO]->(c)
            WHERE connected.vehicle_signature = $vehicle_signature
            RETURN 'incoming' as direction, r, connected
            """
        else:  # both
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            
            OPTIONAL MATCH (c)-[r_out:CONNECTS_TO]->(connected_out:Component)
            WHERE connected_out.vehicle_signature = $vehicle_signature
            
            OPTIONAL MATCH (connected_in:Component)-[r_in:CONNECTS_TO]->(c)
            WHERE connected_in.vehicle_signature = $vehicle_signature
            
            WITH collect({direction: 'outgoing', r: r_out, connected: connected_out}) as outgoing,
                 collect({direction: 'incoming', r: r_in, connected: connected_in}) as incoming
            
            UNWIND (outgoing + incoming) as connection
            WHERE connection.connected IS NOT NULL
            RETURN connection.direction as direction, connection.r as r, connection.connected as connected
            """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        connections = []
        for record in result.records:
            rel_data = dict(record["r"])
            connection_rel = ConnectsToRelationship.from_neo4j_dict(rel_data)
            
            connections.append({
                "direction": record["direction"],
                "relationship": connection_rel,
                "connected_component": dict(record["connected"])
            })
        
        return connections
    
    async def update_connection(
        self,
        from_component_id: str,
        to_component_id: str,
        vehicle_signature: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a CONNECTS_TO relationship"""
        
        if not updates:
            return False
        
        # Add update timestamp
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        # Build SET clause dynamically
        set_clauses = [f"r.{key} = ${key}" for key in updates.keys()]
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (from:Component {{id: $from_id, vehicle_signature: $vehicle_signature}})
        MATCH (to:Component {{id: $to_id, vehicle_signature: $vehicle_signature}})
        MATCH (from)-[r:CONNECTS_TO]->(to)
        SET {set_clause}
        RETURN r
        """
        
        params = {
            "from_id": from_component_id,
            "to_id": to_component_id,
            "vehicle_signature": vehicle_signature,
            **updates
        }
        
        result = await self.neo4j.run(query, **params)
        
        if result.records:
            logger.info(f"Updated connection: {from_component_id} -> {to_component_id}")
            return True
        
        return False
    
    async def delete_connection(
        self,
        from_component_id: str,
        to_component_id: str,
        vehicle_signature: str
    ) -> bool:
        """Delete a CONNECTS_TO relationship"""
        
        query = """
        MATCH (from:Component {id: $from_id, vehicle_signature: $vehicle_signature})
        MATCH (to:Component {id: $to_id, vehicle_signature: $vehicle_signature})
        MATCH (from)-[r:CONNECTS_TO]->(to)
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await self.neo4j.run(
            query,
            from_id=from_component_id,
            to_id=to_component_id,
            vehicle_signature=vehicle_signature
        )
        
        deleted_count = result.records[0]["deleted_count"] if result.records else 0
        
        if deleted_count > 0:
            logger.info(f"Deleted connection: {from_component_id} -> {to_component_id}")
            return True
        
        return False
    
    # ==================== POWERED_BY Relationships ====================
    
    async def create_power_relationship(
        self,
        source_component_id: str,
        powered_component_id: str,
        vehicle_signature: str,
        power_props: PoweredByRelationship
    ) -> bool:
        """Create a POWERED_BY relationship"""
        
        query = """
        MATCH (source:Component {id: $source_id, vehicle_signature: $vehicle_signature})
        MATCH (powered:Component {id: $powered_id, vehicle_signature: $vehicle_signature})
        CREATE (powered)-[r:POWERED_BY $props]->(source)
        RETURN r
        """
        
        result = await self.neo4j.run(
            query,
            source_id=source_component_id,
            powered_id=powered_component_id,
            vehicle_signature=vehicle_signature,
            props=power_props.to_neo4j_dict()
        )
        
        if result.records:
            logger.info(f"Created power relationship: {powered_component_id} powered by {source_component_id}")
            return True
        
        return False
    
    async def get_power_relationships(
        self,
        component_id: str,
        vehicle_signature: str,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get power relationships for a component"""
        
        if direction == "sources":  # What powers this component
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            MATCH (c)-[r:POWERED_BY]->(source:Component)
            WHERE source.vehicle_signature = $vehicle_signature
            RETURN 'powered_by' as direction, r, source as related_component
            """
        elif direction == "loads":  # What this component powers
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            MATCH (load:Component)-[r:POWERED_BY]->(c)
            WHERE load.vehicle_signature = $vehicle_signature
            RETURN 'powers' as direction, r, load as related_component
            """
        else:  # both
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            
            OPTIONAL MATCH (c)-[r_source:POWERED_BY]->(source:Component)
            WHERE source.vehicle_signature = $vehicle_signature
            
            OPTIONAL MATCH (load:Component)-[r_load:POWERED_BY]->(c)
            WHERE load.vehicle_signature = $vehicle_signature
            
            WITH collect({direction: 'powered_by', r: r_source, related: source}) as sources,
                 collect({direction: 'powers', r: r_load, related: load}) as loads
            
            UNWIND (sources + loads) as power_rel
            WHERE power_rel.related IS NOT NULL
            RETURN power_rel.direction as direction, power_rel.r as r, power_rel.related as related_component
            """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        power_relationships = []
        for record in result.records:
            rel_data = dict(record["r"])
            power_rel = PoweredByRelationship.from_neo4j_dict(rel_data)
            
            power_relationships.append({
                "direction": record["direction"],
                "relationship": power_rel,
                "related_component": dict(record["related_component"])
            })
        
        return power_relationships
    
    # ==================== CONTROLS Relationships ====================
    
    async def create_control_relationship(
        self,
        controller_id: str,
        controlled_id: str,
        vehicle_signature: str,
        control_props: ControlsRelationship
    ) -> bool:
        """Create a CONTROLS relationship"""
        
        query = """
        MATCH (controller:Component {id: $controller_id, vehicle_signature: $vehicle_signature})
        MATCH (controlled:Component {id: $controlled_id, vehicle_signature: $vehicle_signature})
        CREATE (controller)-[r:CONTROLS $props]->(controlled)
        RETURN r
        """
        
        result = await self.neo4j.run(
            query,
            controller_id=controller_id,
            controlled_id=controlled_id,
            vehicle_signature=vehicle_signature,
            props=control_props.to_neo4j_dict()
        )
        
        if result.records:
            logger.info(f"Created control relationship: {controller_id} controls {controlled_id}")
            return True
        
        return False
    
    async def get_control_relationships(
        self,
        component_id: str,
        vehicle_signature: str,
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get control relationships for a component"""
        
        if direction == "controls":  # What this component controls
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            MATCH (c)-[r:CONTROLS]->(controlled:Component)
            WHERE controlled.vehicle_signature = $vehicle_signature
            RETURN 'controls' as direction, r, controlled as related_component
            """
        elif direction == "controlled_by":  # What controls this component
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            MATCH (controller:Component)-[r:CONTROLS]->(c)
            WHERE controller.vehicle_signature = $vehicle_signature
            RETURN 'controlled_by' as direction, r, controller as related_component
            """
        else:  # both
            query = """
            MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
            
            OPTIONAL MATCH (c)-[r_controls:CONTROLS]->(controlled:Component)
            WHERE controlled.vehicle_signature = $vehicle_signature
            
            OPTIONAL MATCH (controller:Component)-[r_controlled:CONTROLS]->(c)
            WHERE controller.vehicle_signature = $vehicle_signature
            
            WITH collect({direction: 'controls', r: r_controls, related: controlled}) as controls,
                 collect({direction: 'controlled_by', r: r_controlled, related: controller}) as controlled_by
            
            UNWIND (controls + controlled_by) as control_rel
            WHERE control_rel.related IS NOT NULL
            RETURN control_rel.direction as direction, control_rel.r as r, control_rel.related as related_component
            """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        control_relationships = []
        for record in result.records:
            rel_data = dict(record["r"])
            control_rel = ControlsRelationship.from_neo4j_dict(rel_data)
            
            control_relationships.append({
                "direction": record["direction"],
                "relationship": control_rel,
                "related_component": dict(record["related_component"])
            })
        
        return control_relationships
    
    # ==================== LOCATED_IN Relationships ====================
    
    async def create_location_relationship(
        self,
        component_id: str,
        zone_id: str,
        vehicle_signature: str,
        location_props: LocatedInRelationship
    ) -> bool:
        """Create a LOCATED_IN relationship"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        MATCH (z:Zone {id: $zone_id, vehicle_signature: $vehicle_signature})
        CREATE (c)-[r:LOCATED_IN $props]->(z)
        RETURN r
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            zone_id=zone_id,
            vehicle_signature=vehicle_signature,
            props=location_props.to_neo4j_dict()
        )
        
        if result.records:
            logger.info(f"Created location relationship: {component_id} located in {zone_id}")
            return True
        
        return False
    
    async def get_component_location(
        self,
        component_id: str,
        vehicle_signature: str
    ) -> Optional[Dict[str, Any]]:
        """Get location relationship for a component"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        MATCH (c)-[r:LOCATED_IN]->(z:Zone)
        WHERE z.vehicle_signature = $vehicle_signature
        RETURN r, z
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return None
        
        record = result.records[0]
        rel_data = dict(record["r"])
        location_rel = LocatedInRelationship.from_neo4j_dict(rel_data)
        
        return {
            "relationship": location_rel,
            "zone": dict(record["z"])
        }
    
    # ==================== PART_OF Relationships ====================
    
    async def create_circuit_membership(
        self,
        component_id: str,
        circuit_id: str,
        vehicle_signature: str,
        membership_props: PartOfRelationship
    ) -> bool:
        """Create a PART_OF relationship for circuit membership"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        MATCH (circuit:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        CREATE (c)-[r:PART_OF $props]->(circuit)
        RETURN r
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature,
            props=membership_props.to_neo4j_dict()
        )
        
        if result.records:
            logger.info(f"Created circuit membership: {component_id} part of {circuit_id}")
            return True
        
        return False
    
    async def get_component_circuits(
        self,
        component_id: str,
        vehicle_signature: str
    ) -> List[Dict[str, Any]]:
        """Get all circuits a component belongs to"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        MATCH (c)-[r:PART_OF]->(circuit:Circuit)
        WHERE circuit.vehicle_signature = $vehicle_signature
        RETURN r, circuit
        ORDER BY circuit.name
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        circuits = []
        for record in result.records:
            rel_data = dict(record["r"])
            part_of_rel = PartOfRelationship.from_neo4j_dict(rel_data)
            
            circuits.append({
                "relationship": part_of_rel,
                "circuit": dict(record["circuit"])
            })
        
        return circuits
    
    # ==================== Generic Relationship Operations ====================
    
    async def get_all_relationships(
        self,
        component_id: str,
        vehicle_signature: str,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all relationships for a component grouped by type"""
        
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        query = f"""
        MATCH (c:Component {{id: $component_id, vehicle_signature: $vehicle_signature}})
        
        // Outgoing relationships
        OPTIONAL MATCH (c)-[r_out{rel_filter}]->(target)
        WHERE target.vehicle_signature = $vehicle_signature OR 
              NOT EXISTS(target.vehicle_signature)
        
        // Incoming relationships  
        OPTIONAL MATCH (source)-[r_in{rel_filter}]->(c)
        WHERE source.vehicle_signature = $vehicle_signature
        
        RETURN 
            collect(DISTINCT {{
                direction: 'outgoing',
                type: type(r_out),
                relationship: r_out,
                related_node: target
            }}) as outgoing,
            collect(DISTINCT {{
                direction: 'incoming',
                type: type(r_in),
                relationship: r_in,
                related_node: source
            }}) as incoming
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return {}
        
        record = result.records[0]
        
        # Group relationships by type
        relationships = {
            "CONNECTS_TO": [],
            "POWERED_BY": [],
            "CONTROLS": [],
            "LOCATED_IN": [],
            "PART_OF": []
        }
        
        # Process outgoing and incoming relationships
        for rel_group in [record["outgoing"], record["incoming"]]:
            for rel_data in rel_group:
                if rel_data["relationship"] is not None:
                    rel_type = rel_data["type"]
                    if rel_type in relationships:
                        relationships[rel_type].append({
                            "direction": rel_data["direction"],
                            "relationship": dict(rel_data["relationship"]),
                            "related_node": dict(rel_data["related_node"]) if rel_data["related_node"] else None
                        })
        
        return relationships
    
    async def delete_all_component_relationships(
        self,
        component_id: str,
        vehicle_signature: str
    ) -> int:
        """Delete all relationships for a component"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (c)-[r]-()
        DELETE r
        RETURN count(r) as deleted_count
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        deleted_count = result.records[0]["deleted_count"] if result.records else 0
        
        if deleted_count > 0:
            logger.info(f"Deleted {deleted_count} relationships for component {component_id}")
        
        return deleted_count
    
    async def find_shortest_path(
        self,
        from_component_id: str,
        to_component_id: str,
        vehicle_signature: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Find shortest path between two components"""
        
        rel_filter = ":CONNECTS_TO|POWERED_BY|CONTROLS"
        if relationship_types:
            rel_filter = ":" + "|".join(relationship_types)
        
        query = f"""
        MATCH (from:Component {{id: $from_id, vehicle_signature: $vehicle_signature}})
        MATCH (to:Component {{id: $to_id, vehicle_signature: $vehicle_signature}})
        MATCH path = shortestPath((from)-[{rel_filter}*1..{max_depth}]-(to))
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT 1
        """
        
        result = await self.neo4j.run(
            query,
            from_id=from_component_id,
            to_id=to_component_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return None
        
        record = result.records[0]
        path = record["path"]
        
        return {
            "path_length": record["path_length"],
            "nodes": [dict(node) for node in path.nodes],
            "relationships": [dict(rel) for rel in path.relationships],
            "path_types": [type(rel).__name__ for rel in path.relationships]
        }
    
    async def get_relationship_statistics(self, vehicle_signature: str) -> Dict[str, Any]:
        """Get statistics about relationships in the system"""
        
        query = """
        MATCH ()-[r]->()
        WHERE (startNode(r).vehicle_signature = $vehicle_signature AND 
               endNode(r).vehicle_signature = $vehicle_signature) OR
              (exists(startNode(r).vehicle_signature) AND 
               startNode(r).vehicle_signature = $vehicle_signature)
        
        RETURN 
            type(r) as relationship_type,
            count(r) as count
        ORDER BY count DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        statistics = {
            "total_relationships": 0,
            "by_type": {},
            "generated_at": datetime.utcnow().isoformat()
        }
        
        for record in result.records:
            rel_type = record["relationship_type"]
            count = record["count"]
            statistics["by_type"][rel_type] = count
            statistics["total_relationships"] += count
        
        return statistics