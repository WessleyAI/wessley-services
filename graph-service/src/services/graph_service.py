"""
Core graph service for electrical system data management
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..models.nodes.component import ComponentNode
from ..models.nodes.circuit import CircuitNode
from ..models.nodes.zone import ZoneNode
from ..models.nodes.connector import ConnectorNode
from ..models.nodes.vehicle import VehicleNode

from ..models.relationships import (
    ConnectsToRelationship,
    PoweredByRelationship, 
    ControlsRelationship,
    LocatedInRelationship,
    PartOfRelationship
)

from ..models.schemas import (
    ComponentValidationSchema,
    CircuitValidationSchema,
    ComponentQuerySchema,
    CircuitQuerySchema
)

from ..utils.neo4j_utils import Neo4jClient
from ..utils.data_validation import DataValidator

logger = logging.getLogger(__name__)


class GraphService:
    """
    Core service for graph operations on electrical system data
    """
    
    def __init__(self, neo4j_client: Neo4jClient, data_validator: DataValidator):
        self.neo4j = neo4j_client
        self.validator = data_validator
    
    # ==================== Component Operations ====================
    
    async def create_component(
        self, 
        component_data: ComponentValidationSchema,
        validate: bool = True
    ) -> ComponentNode:
        """Create a new component in the graph"""
        
        if validate:
            await self.validator.validate_component(component_data)
        
        component = ComponentNode(**component_data.dict())
        
        query = """
        CREATE (c:Component $props)
        RETURN c
        """
        
        result = await self.neo4j.run(query, props=component.to_neo4j_dict())
        
        if not result.records:
            raise Exception("Failed to create component")
        
        logger.info(f"Created component: {component.id}")
        return component
    
    async def get_component(
        self, 
        component_id: str, 
        vehicle_signature: str
    ) -> Optional[ComponentNode]:
        """Retrieve a specific component"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        RETURN c
        """
        
        result = await self.neo4j.run(
            query, 
            component_id=component_id, 
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return None
        
        component_data = dict(result.records[0]["c"])
        return ComponentNode.from_neo4j_dict(component_data)
    
    async def update_component(
        self,
        component_id: str,
        vehicle_signature: str,
        updates: Dict[str, Any],
        validate: bool = True
    ) -> ComponentNode:
        """Update an existing component"""
        
        if validate:
            await self.validator.validate_component_update(updates)
        
        # Add update timestamp
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        # Build SET clause dynamically
        set_clauses = [f"c.{key} = ${key}" for key in updates.keys()]
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (c:Component {{id: $component_id, vehicle_signature: $vehicle_signature}})
        SET {set_clause}
        RETURN c
        """
        
        params = {
            "component_id": component_id,
            "vehicle_signature": vehicle_signature,
            **updates
        }
        
        result = await self.neo4j.run(query, **params)
        
        if not result.records:
            raise Exception(f"Component {component_id} not found")
        
        component_data = dict(result.records[0]["c"])
        return ComponentNode.from_neo4j_dict(component_data)
    
    async def delete_component(
        self, 
        component_id: str, 
        vehicle_signature: str
    ) -> bool:
        """Delete a component and its relationships"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        DETACH DELETE c
        RETURN count(c) as deleted_count
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        deleted_count = result.records[0]["deleted_count"]
        if deleted_count > 0:
            logger.info(f"Deleted component: {component_id}")
            return True
        
        return False
    
    async def query_components(
        self, 
        query_params: ComponentQuerySchema
    ) -> List[ComponentNode]:
        """Query components with filtering"""
        
        where_clauses = ["c.vehicle_signature = $vehicle_signature"]
        params = {"vehicle_signature": query_params.vehicle_signature}
        
        # Build dynamic WHERE clause
        if query_params.component_types:
            where_clauses.append("c.type IN $component_types")
            params["component_types"] = [t.value for t in query_params.component_types]
        
        if query_params.zone:
            where_clauses.append("c.zone = $zone")
            params["zone"] = query_params.zone
        
        if query_params.manufacturer:
            where_clauses.append("c.manufacturer = $manufacturer")
            params["manufacturer"] = query_params.manufacturer
        
        if query_params.has_position is not None:
            if query_params.has_position:
                where_clauses.append("c.position IS NOT NULL")
            else:
                where_clauses.append("c.position IS NULL")
        
        if query_params.voltage_min is not None:
            where_clauses.append("c.voltage_rating >= $voltage_min")
            params["voltage_min"] = query_params.voltage_min
        
        if query_params.voltage_max is not None:
            where_clauses.append("c.voltage_rating <= $voltage_max")
            params["voltage_max"] = query_params.voltage_max
        
        where_clause = " AND ".join(where_clauses)
        
        # Build query
        query = f"""
        MATCH (c:Component)
        WHERE {where_clause}
        RETURN c
        ORDER BY c.type, c.name
        """
        
        # Add pagination
        if query_params.limit:
            query += f" LIMIT {query_params.limit}"
            if query_params.offset:
                query += f" SKIP {query_params.offset}"
        
        result = await self.neo4j.run(query, **params)
        
        components = []
        for record in result.records:
            component_data = dict(record["c"])
            components.append(ComponentNode.from_neo4j_dict(component_data))
        
        return components
    
    # ==================== Circuit Operations ====================
    
    async def create_circuit(
        self, 
        circuit_data: CircuitValidationSchema,
        validate: bool = True
    ) -> CircuitNode:
        """Create a new circuit in the graph"""
        
        if validate:
            await self.validator.validate_circuit(circuit_data)
        
        circuit = CircuitNode(**circuit_data.dict())
        
        query = """
        CREATE (c:Circuit $props)
        RETURN c
        """
        
        result = await self.neo4j.run(query, props=circuit.to_neo4j_dict())
        
        if not result.records:
            raise Exception("Failed to create circuit")
        
        logger.info(f"Created circuit: {circuit.id}")
        return circuit
    
    async def get_circuit(
        self, 
        circuit_id: str, 
        vehicle_signature: str
    ) -> Optional[CircuitNode]:
        """Retrieve a specific circuit"""
        
        query = """
        MATCH (c:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        RETURN c
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return None
        
        circuit_data = dict(result.records[0]["c"])
        return CircuitNode.from_neo4j_dict(circuit_data)
    
    # ==================== Relationship Operations ====================
    
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
    
    # ==================== Analysis Operations ====================
    
    async def get_component_connections(
        self, 
        component_id: str, 
        vehicle_signature: str,
        relationship_types: Optional[List[str]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all connections for a component"""
        
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        query = f"""
        MATCH (c:Component {{id: $component_id, vehicle_signature: $vehicle_signature}})
        
        // Outgoing relationships
        OPTIONAL MATCH (c)-[r_out{rel_filter}]->(connected_out:Component)
        
        // Incoming relationships
        OPTIONAL MATCH (connected_in:Component)-[r_in{rel_filter}]->(c)
        
        RETURN 
            collect(DISTINCT {{
                direction: 'outgoing',
                relationship: r_out,
                component: connected_out
            }}) as outgoing,
            collect(DISTINCT {{
                direction: 'incoming', 
                relationship: r_in,
                component: connected_in
            }}) as incoming
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return {"outgoing": [], "incoming": []}
        
        record = result.records[0]
        return {
            "outgoing": [conn for conn in record["outgoing"] if conn["component"] is not None],
            "incoming": [conn for conn in record["incoming"] if conn["component"] is not None]
        }
    
    async def trace_power_path(
        self,
        from_component_id: str,
        to_component_id: str,
        vehicle_signature: str,
        max_depth: int = 10
    ) -> Optional[Dict[str, Any]]:
        """Trace power path between two components"""
        
        query = """
        MATCH (start:Component {id: $from_id, vehicle_signature: $vehicle_signature})
        MATCH (end:Component {id: $to_id, vehicle_signature: $vehicle_signature})
        MATCH path = shortestPath((start)-[:POWERED_BY|CONNECTS_TO*1..${max_depth}]->(end))
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT 5
        """
        
        result = await self.neo4j.run(
            query,
            from_id=from_component_id,
            to_id=to_component_id,
            vehicle_signature=vehicle_signature,
            max_depth=max_depth
        )
        
        if not result.records:
            return None
        
        record = result.records[0]
        return {
            "path": record["path"],
            "length": record["path_length"],
            "components": [node for node in record["path"].nodes],
            "relationships": [rel for rel in record["path"].relationships]
        }
    
    async def get_circuit_components(
        self,
        circuit_id: str,
        vehicle_signature: str
    ) -> List[ComponentNode]:
        """Get all components in a circuit"""
        
        query = """
        MATCH (circuit:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        MATCH (component:Component)-[:PART_OF]->(circuit)
        RETURN component
        ORDER BY component.type, component.name
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        components = []
        for record in result.records:
            component_data = dict(record["component"])
            components.append(ComponentNode.from_neo4j_dict(component_data))
        
        return components
    
    async def get_system_statistics(
        self, 
        vehicle_signature: str
    ) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        query = """
        MATCH (v:Vehicle {signature: $vehicle_signature})
        
        // Count components by type
        OPTIONAL MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Count circuits  
        OPTIONAL MATCH (circuit:Circuit {vehicle_signature: $vehicle_signature})
        
        // Count zones
        OPTIONAL MATCH (z:Zone {vehicle_signature: $vehicle_signature})
        
        // Count relationships
        OPTIONAL MATCH ()-[r:CONNECTS_TO]->() 
        WHERE r.vehicle_signature = $vehicle_signature OR 
              (startNode(r).vehicle_signature = $vehicle_signature AND 
               endNode(r).vehicle_signature = $vehicle_signature)
        
        RETURN 
            count(DISTINCT c) as total_components,
            count(DISTINCT circuit) as total_circuits, 
            count(DISTINCT z) as total_zones,
            count(DISTINCT r) as total_connections,
            collect(DISTINCT c.type) as component_types,
            collect(DISTINCT c.zone) as zones
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        if not result.records:
            return {}
        
        record = result.records[0]
        return {
            "vehicle_signature": vehicle_signature,
            "total_components": record["total_components"],
            "total_circuits": record["total_circuits"],
            "total_zones": record["total_zones"],
            "total_connections": record["total_connections"],
            "component_types": [t for t in record["component_types"] if t],
            "zones": [z for z in record["zones"] if z],
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # ==================== Health Check ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform service health check"""
        
        try:
            # Test Neo4j connection
            await self.neo4j.run("RETURN 1 as test")
            
            # Get database info
            db_info = await self.neo4j.get_database_info()
            
            return {
                "status": "healthy",
                "neo4j_connected": True,
                "database_info": db_info,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "neo4j_connected": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }