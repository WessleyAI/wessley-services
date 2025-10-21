"""
Circuit repository for CRUD operations on electrical circuits
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.nodes.circuit import CircuitNode
from ..models.schemas.circuit_schema import CircuitQuerySchema
from ..utils.neo4j_utils import Neo4jClient

logger = logging.getLogger(__name__)


class CircuitRepository:
    """
    Repository for circuit CRUD operations
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
    
    async def create(self, circuit: CircuitNode) -> CircuitNode:
        """Create a new circuit"""
        
        query = """
        CREATE (c:Circuit $props)
        RETURN c
        """
        
        result = await self.neo4j.run(query, props=circuit.to_neo4j_dict())
        
        if not result.records:
            raise Exception("Failed to create circuit")
        
        created_data = dict(result.records[0]["c"])
        logger.info(f"Created circuit: {circuit.id}")
        return CircuitNode.from_neo4j_dict(created_data)
    
    async def get_by_id(self, circuit_id: str, vehicle_signature: str) -> Optional[CircuitNode]:
        """Get circuit by ID and vehicle signature"""
        
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
    
    async def update(self, circuit_id: str, vehicle_signature: str, updates: Dict[str, Any]) -> Optional[CircuitNode]:
        """Update circuit with provided fields"""
        
        if not updates:
            existing = await self.get_by_id(circuit_id, vehicle_signature)
            return existing
        
        # Add update timestamp
        updates["updated_at"] = datetime.utcnow().isoformat()
        
        # Build SET clause dynamically
        set_clauses = [f"c.{key} = ${key}" for key in updates.keys()]
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        MATCH (c:Circuit {{id: $circuit_id, vehicle_signature: $vehicle_signature}})
        SET {set_clause}
        RETURN c
        """
        
        params = {
            "circuit_id": circuit_id,
            "vehicle_signature": vehicle_signature,
            **updates
        }
        
        result = await self.neo4j.run(query, **params)
        
        if not result.records:
            return None
        
        updated_data = dict(result.records[0]["c"])
        logger.info(f"Updated circuit: {circuit_id}")
        return CircuitNode.from_neo4j_dict(updated_data)
    
    async def delete(self, circuit_id: str, vehicle_signature: str) -> bool:
        """Delete circuit and its relationships"""
        
        query = """
        MATCH (c:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        DETACH DELETE c
        RETURN count(c) as deleted_count
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        deleted_count = result.records[0]["deleted_count"] if result.records else 0
        
        if deleted_count > 0:
            logger.info(f"Deleted circuit: {circuit_id}")
            return True
        
        return False
    
    async def find_by_criteria(self, query_params: CircuitQuerySchema) -> List[CircuitNode]:
        """Find circuits matching query criteria"""
        
        where_clauses = ["c.vehicle_signature = $vehicle_signature"]
        params = {"vehicle_signature": query_params.vehicle_signature}
        
        # Build dynamic WHERE clause
        if query_params.circuit_types:
            where_clauses.append("c.circuit_type IN $circuit_types")
            params["circuit_types"] = [t.value for t in query_params.circuit_types]
        
        # Voltage range filters
        if query_params.voltage_min is not None:
            where_clauses.append("c.voltage >= $voltage_min")
            params["voltage_min"] = query_params.voltage_min
        
        if query_params.voltage_max is not None:
            where_clauses.append("c.voltage <= $voltage_max")
            params["voltage_max"] = query_params.voltage_max
        
        # Current range filters
        if query_params.current_min is not None:
            where_clauses.append("c.max_current >= $current_min")
            params["current_min"] = query_params.current_min
        
        if query_params.current_max is not None:
            where_clauses.append("c.max_current <= $current_max")
            params["current_max"] = query_params.current_max
        
        # Safety and redundancy filters
        if query_params.safety_critical is not None:
            where_clauses.append("c.safety_critical = $safety_critical")
            params["safety_critical"] = query_params.safety_critical
        
        if query_params.has_redundancy is not None:
            where_clauses.append("c.redundancy_available = $has_redundancy")
            params["has_redundancy"] = query_params.has_redundancy
        
        if query_params.protection_type:
            where_clauses.append("c.protection = $protection_type")
            params["protection_type"] = query_params.protection_type.value
        
        where_clause = " AND ".join(where_clauses)
        
        # Build complete query
        query = f"""
        MATCH (c:Circuit)
        WHERE {where_clause}
        RETURN c
        ORDER BY c.circuit_type, c.name
        """
        
        # Add pagination
        if query_params.limit:
            query += f" LIMIT {query_params.limit}"
            if query_params.offset:
                query += f" SKIP {query_params.offset}"
        
        result = await self.neo4j.run(query, **params)
        
        circuits = []
        for record in result.records:
            circuit_data = dict(record["c"])
            circuits.append(CircuitNode.from_neo4j_dict(circuit_data))
        
        return circuits
    
    async def find_by_type(self, vehicle_signature: str, circuit_type: str) -> List[CircuitNode]:
        """Find all circuits of a specific type"""
        
        query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature, circuit_type: $circuit_type})
        RETURN c
        ORDER BY c.name
        """
        
        result = await self.neo4j.run(
            query,
            vehicle_signature=vehicle_signature,
            circuit_type=circuit_type
        )
        
        circuits = []
        for record in result.records:
            circuit_data = dict(record["c"])
            circuits.append(CircuitNode.from_neo4j_dict(circuit_data))
        
        return circuits
    
    async def find_overloaded(self, vehicle_signature: str) -> List[Dict[str, Any]]:
        """Find circuits that are overloaded (load > capacity)"""
        
        query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(c)
        
        WITH c, sum(coalesce(comp.current_rating, 0)) as total_load
        WHERE total_load > c.max_current
        
        RETURN c, total_load, 
               (total_load / c.max_current) as load_factor,
               (total_load - c.max_current) as overload_amount
        ORDER BY load_factor DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        overloaded_circuits = []
        for record in result.records:
            circuit_data = dict(record["c"])
            circuit = CircuitNode.from_neo4j_dict(circuit_data)
            
            overloaded_circuits.append({
                "circuit": circuit,
                "total_load": record["total_load"],
                "load_factor": record["load_factor"],
                "overload_amount": record["overload_amount"]
            })
        
        return overloaded_circuits
    
    async def find_without_protection(self, vehicle_signature: str) -> List[CircuitNode]:
        """Find circuits without protection devices"""
        
        query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        WHERE NOT exists(c.protection) OR c.protection = 'none'
        RETURN c
        ORDER BY c.circuit_type, c.name
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        circuits = []
        for record in result.records:
            circuit_data = dict(record["c"])
            circuits.append(CircuitNode.from_neo4j_dict(circuit_data))
        
        return circuits
    
    async def find_safety_critical(self, vehicle_signature: str) -> List[CircuitNode]:
        """Find safety-critical circuits"""
        
        query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        WHERE c.safety_critical = true
        RETURN c
        ORDER BY c.name
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        circuits = []
        for record in result.records:
            circuit_data = dict(record["c"])
            circuits.append(CircuitNode.from_neo4j_dict(circuit_data))
        
        return circuits
    
    async def get_circuit_components(self, circuit_id: str, vehicle_signature: str) -> List[Dict[str, Any]]:
        """Get all components in a circuit with their roles"""
        
        query = """
        MATCH (c:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        MATCH (comp:Component)-[r:PART_OF]->(c)
        
        RETURN comp, r.role as role, r.is_critical as is_critical
        ORDER BY r.role, comp.type, comp.name
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        components = []
        for record in result.records:
            component_data = dict(record["comp"])
            components.append({
                "component": component_data,
                "role": record["role"],
                "is_critical": record["is_critical"]
            })
        
        return components
    
    async def get_circuit_load_analysis(self, circuit_id: str, vehicle_signature: str) -> Dict[str, Any]:
        """Get detailed load analysis for a circuit"""
        
        query = """
        MATCH (c:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        
        // Get components and their loads
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(c)
        
        WITH c, collect(comp) as components,
             sum(coalesce(comp.current_rating, 0)) as total_current_load,
             sum(coalesce(comp.voltage_rating * comp.current_rating, 0)) as total_power_load
        
        // Calculate metrics
        WITH c, components, total_current_load, total_power_load,
             CASE 
                 WHEN c.max_current > 0 THEN (total_current_load / c.max_current) * 100
                 ELSE 0
             END as load_percentage,
             CASE
                 WHEN c.max_current > 0 THEN c.max_current - total_current_load
                 ELSE 0
             END as available_capacity
        
        RETURN 
            c,
            size(components) as component_count,
            total_current_load,
            total_power_load,
            load_percentage,
            available_capacity,
            [comp in components | {
                id: comp.id,
                name: comp.name,
                type: comp.type,
                current_rating: comp.current_rating,
                voltage_rating: comp.voltage_rating
            }] as component_details
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        if not result.records:
            return {}
        
        record = result.records[0]
        circuit_data = dict(record["c"])
        
        return {
            "circuit": CircuitNode.from_neo4j_dict(circuit_data),
            "component_count": record["component_count"],
            "total_current_load": record["total_current_load"],
            "total_power_load": record["total_power_load"],
            "load_percentage": record["load_percentage"],
            "available_capacity": record["available_capacity"],
            "component_details": record["component_details"],
            "is_overloaded": record["load_percentage"] > 100,
            "safety_margin": record["available_capacity"]
        }
    
    async def get_type_distribution(self, vehicle_signature: str) -> Dict[str, int]:
        """Get distribution of circuit types"""
        
        query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        RETURN c.circuit_type as circuit_type, count(c) as count
        ORDER BY count DESC
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        distribution = {}
        for record in result.records:
            circuit_type = record["circuit_type"]
            count = record["count"]
            if circuit_type:
                distribution[circuit_type] = count
        
        return distribution
    
    async def get_voltage_distribution(self, vehicle_signature: str) -> Dict[str, int]:
        """Get distribution of circuits by voltage level"""
        
        query = """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        WHERE c.voltage IS NOT NULL
        RETURN c.voltage as voltage, count(c) as count
        ORDER BY voltage
        """
        
        result = await self.neo4j.run(query, vehicle_signature=vehicle_signature)
        
        distribution = {}
        for record in result.records:
            voltage = f"{record['voltage']}V"
            count = record["count"]
            distribution[voltage] = count
        
        return distribution
    
    async def count_by_criteria(self, query_params: CircuitQuerySchema) -> int:
        """Count circuits matching criteria"""
        
        where_clauses = ["c.vehicle_signature = $vehicle_signature"]
        params = {"vehicle_signature": query_params.vehicle_signature}
        
        # Build same WHERE clause as find_by_criteria but return count
        if query_params.circuit_types:
            where_clauses.append("c.circuit_type IN $circuit_types")
            params["circuit_types"] = [t.value for t in query_params.circuit_types]
        
        if query_params.safety_critical is not None:
            where_clauses.append("c.safety_critical = $safety_critical")
            params["safety_critical"] = query_params.safety_critical
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
        MATCH (c:Circuit)
        WHERE {where_clause}
        RETURN count(c) as circuit_count
        """
        
        result = await self.neo4j.run(query, **params)
        
        return result.records[0]["circuit_count"] if result.records else 0
    
    async def batch_create(self, circuits: List[CircuitNode]) -> List[CircuitNode]:
        """Create multiple circuits in a batch operation"""
        
        if not circuits:
            return []
        
        # Prepare batch data
        circuit_props = [circuit.to_neo4j_dict() for circuit in circuits]
        
        query = """
        UNWIND $circuit_props as props
        CREATE (c:Circuit)
        SET c = props
        RETURN c
        """
        
        result = await self.neo4j.run(query, circuit_props=circuit_props)
        
        created_circuits = []
        for record in result.records:
            circuit_data = dict(record["c"])
            created_circuits.append(CircuitNode.from_neo4j_dict(circuit_data))
        
        logger.info(f"Batch created {len(created_circuits)} circuits")
        return created_circuits
    
    async def exists(self, circuit_id: str, vehicle_signature: str) -> bool:
        """Check if circuit exists"""
        
        query = """
        MATCH (c:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        RETURN count(c) > 0 as exists
        """
        
        result = await self.neo4j.run(
            query,
            circuit_id=circuit_id,
            vehicle_signature=vehicle_signature
        )
        
        return result.records[0]["exists"] if result.records else False
    
    async def get_circuit_topology(self, circuit_id: str, vehicle_signature: str) -> Dict[str, Any]:
        """Get complete circuit topology with connections"""
        
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
                properties: properties(rel)
            }) as connections,
            collect(DISTINCT {
                from: power_source.id,
                to: component.id,
                type: 'POWERED_BY',
                properties: properties(power_rel)
            }) as power_relationships,
            collect(DISTINCT {
                from: component.id,
                to: controlled.id,
                type: 'CONTROLS',
                properties: properties(control_rel)
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
            "connections": [conn for conn in record["connections"] if conn["to"] is not None],
            "power_relationships": [rel for rel in record["power_relationships"] if rel["from"] is not None],
            "control_relationships": [rel for rel in record["control_relationships"] if rel["to"] is not None],
            "metadata": {
                "component_count": len(record["components"]),
                "connection_count": len([c for c in record["connections"] if c["to"] is not None]),
                "generated_at": datetime.utcnow().isoformat()
            }
        }