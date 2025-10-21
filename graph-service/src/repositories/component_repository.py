"""
Component repository for CRUD operations on electrical components
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.nodes.component import ComponentNode
from ..models.schemas.component_schema import ComponentQuerySchema
from ..utils.neo4j_utils import Neo4jClient

logger = logging.getLogger(__name__)


class ComponentRepository:
    """
    Repository for component CRUD operations
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
    
    async def create(self, component: ComponentNode) -> ComponentNode:
        """Create a new component"""
        
        query = """
        CREATE (c:Component $props)
        RETURN c
        """
        
        result = await self.neo4j.run(query, props=component.to_neo4j_dict())
        
        if not result.records:
            raise Exception("Failed to create component")
        
        created_data = dict(result.records[0]["c"])
        logger.info(f"Created component: {component.id}")
        return ComponentNode.from_neo4j_dict(created_data)
    
    async def get_by_id(self, component_id: str, vehicle_signature: str) -> Optional[ComponentNode]:
        """Get component by ID and vehicle signature"""
        
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
    
    async def update(self, component_id: str, vehicle_signature: str, updates: Dict[str, Any]) -> Optional[ComponentNode]:
        """Update component with provided fields"""
        
        if not updates:
            existing = await self.get_by_id(component_id, vehicle_signature)
            return existing
        
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
            return None
        
        updated_data = dict(result.records[0]["c"])
        logger.info(f"Updated component: {component_id}")
        return ComponentNode.from_neo4j_dict(updated_data)
    
    async def delete(self, component_id: str, vehicle_signature: str) -> bool:
        """Delete component and its relationships"""
        
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
        
        deleted_count = result.records[0]["deleted_count"] if result.records else 0
        
        if deleted_count > 0:
            logger.info(f"Deleted component: {component_id}")
            return True
        
        return False
    
    async def find_by_criteria(self, query_params: ComponentQuerySchema) -> List[ComponentNode]:
        """Find components matching query criteria"""
        
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
        
        # Voltage range filters
        if query_params.voltage_min is not None:
            where_clauses.append("c.voltage_rating >= $voltage_min")
            params["voltage_min"] = query_params.voltage_min
        
        if query_params.voltage_max is not None:
            where_clauses.append("c.voltage_rating <= $voltage_max")
            params["voltage_max"] = query_params.voltage_max
        
        # Current range filters
        if query_params.current_min is not None:
            where_clauses.append("c.current_rating >= $current_min")
            params["current_min"] = query_params.current_min
        
        if query_params.current_max is not None:
            where_clauses.append("c.current_rating <= $current_max")
            params["current_max"] = query_params.current_max
        
        where_clause = " AND ".join(where_clauses)
        
        # Build complete query
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
    
    async def find_by_zone(self, vehicle_signature: str, zone: str) -> List[ComponentNode]:
        """Find all components in a specific zone"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature, zone: $zone})
        RETURN c
        ORDER BY c.type, c.name
        """
        
        result = await self.neo4j.run(
            query,
            vehicle_signature=vehicle_signature,
            zone=zone
        )
        
        components = []
        for record in result.records:
            component_data = dict(record["c"])
            components.append(ComponentNode.from_neo4j_dict(component_data))
        
        return components
    
    async def find_by_type(self, vehicle_signature: str, component_type: str) -> List[ComponentNode]:
        """Find all components of a specific type"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature, type: $component_type})
        RETURN c
        ORDER BY c.name
        """
        
        result = await self.neo4j.run(
            query,
            vehicle_signature=vehicle_signature,
            component_type=component_type
        )
        
        components = []
        for record in result.records:
            component_data = dict(record["c"])
            components.append(ComponentNode.from_neo4j_dict(component_data))
        
        return components
    
    async def find_without_position(self, vehicle_signature: str) -> List[ComponentNode]:
        """Find components without spatial position data"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.position IS NULL
        RETURN c
        ORDER BY c.type, c.name
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        components = []
        for record in result.records:
            component_data = dict(record["c"])
            components.append(ComponentNode.from_neo4j_dict(component_data))
        
        return components
    
    async def find_orphaned(self, vehicle_signature: str) -> List[ComponentNode]:
        """Find components with no relationships (orphaned)"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE NOT (c)-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]-() 
          AND NOT ()-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]->(c)
        RETURN c
        ORDER BY c.type, c.name
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        components = []
        for record in result.records:
            component_data = dict(record["c"])
            components.append(ComponentNode.from_neo4j_dict(component_data))
        
        return components
    
    async def find_by_electrical_rating(
        self, 
        vehicle_signature: str,
        min_voltage: Optional[float] = None,
        max_voltage: Optional[float] = None,
        min_current: Optional[float] = None,
        max_current: Optional[float] = None
    ) -> List[ComponentNode]:
        """Find components by electrical rating ranges"""
        
        where_clauses = ["c.vehicle_signature = $vehicle_signature"]
        params = {"vehicle_signature": vehicle_signature}
        
        if min_voltage is not None:
            where_clauses.append("c.voltage_rating >= $min_voltage")
            params["min_voltage"] = min_voltage
        
        if max_voltage is not None:
            where_clauses.append("c.voltage_rating <= $max_voltage")
            params["max_voltage"] = max_voltage
        
        if min_current is not None:
            where_clauses.append("c.current_rating >= $min_current")
            params["min_current"] = min_current
        
        if max_current is not None:
            where_clauses.append("c.current_rating <= $max_current")
            params["max_current"] = max_current
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (c:Component)
        WHERE {where_clause}
        RETURN c
        ORDER BY c.voltage_rating DESC, c.current_rating DESC
        """
        
        result = await self.neo4j.run(query, **params)
        
        components = []
        for record in result.records:
            component_data = dict(record["c"])
            components.append(ComponentNode.from_neo4j_dict(component_data))
        
        return components
    
    async def count_by_criteria(self, query_params: ComponentQuerySchema) -> int:
        """Count components matching criteria"""
        
        where_clauses = ["c.vehicle_signature = $vehicle_signature"]
        params = {"vehicle_signature": query_params.vehicle_signature}
        
        # Build same WHERE clause as find_by_criteria but return count
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
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (c:Component)
        WHERE {where_clause}
        RETURN count(c) as component_count
        """
        
        result = await self.neo4j.run(query, **params)
        
        return result.records[0]["component_count"] if result.records else 0
    
    async def get_type_distribution(self, vehicle_signature: str) -> Dict[str, int]:
        """Get distribution of component types"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        RETURN c.type as component_type, count(c) as count
        ORDER BY count DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        distribution = {}
        for record in result.records:
            comp_type = record["component_type"]
            count = record["count"]
            if comp_type:
                distribution[comp_type] = count
        
        return distribution
    
    async def get_zone_distribution(self, vehicle_signature: str) -> Dict[str, int]:
        """Get distribution of components by zone"""
        
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.zone IS NOT NULL
        RETURN c.zone as zone, count(c) as count
        ORDER BY count DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        distribution = {}
        for record in result.records:
            zone = record["zone"]
            count = record["count"]
            distribution[zone] = count
        
        return distribution
    
    async def batch_create(self, components: List[ComponentNode]) -> List[ComponentNode]:
        """Create multiple components in a batch operation"""
        
        if not components:
            return []
        
        # Prepare batch data
        component_props = [comp.to_neo4j_dict() for comp in components]
        
        query = """
        UNWIND $component_props as props
        CREATE (c:Component)
        SET c = props
        RETURN c
        """
        
        result = await self.neo4j.run(query, component_props=component_props)
        
        created_components = []
        for record in result.records:
            component_data = dict(record["c"])
            created_components.append(ComponentNode.from_neo4j_dict(component_data))
        
        logger.info(f"Batch created {len(created_components)} components")
        return created_components
    
    async def batch_update(self, updates: List[Dict[str, Any]]) -> int:
        """Batch update multiple components"""
        
        if not updates:
            return 0
        
        updated_count = 0
        
        for update_data in updates:
            component_id = update_data.pop("component_id")
            vehicle_signature = update_data.pop("vehicle_signature")
            
            updated_component = await self.update(component_id, vehicle_signature, update_data)
            if updated_component:
                updated_count += 1
        
        logger.info(f"Batch updated {updated_count} components")
        return updated_count
    
    async def exists(self, component_id: str, vehicle_signature: str) -> bool:
        """Check if component exists"""
        
        query = """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        RETURN count(c) > 0 as exists
        """
        
        result = await self.neo4j.run(
            query,
            component_id=component_id,
            vehicle_signature=vehicle_signature
        )
        
        return result.records[0]["exists"] if result.records else False