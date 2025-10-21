"""Optimized Cypher queries for circuit operations"""

class CircuitQueries:
    @staticmethod
    def get_circuit_by_id() -> str:
        return """
        MATCH (c:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        RETURN c
        """
    
    @staticmethod
    def get_circuit_load_analysis() -> str:
        return """
        MATCH (c:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(c)
        WITH c, collect(comp) as components,
             sum(coalesce(comp.current_rating, 0)) as total_load
        RETURN c, components, total_load,
               (total_load / c.max_current) * 100 as load_percentage
        """
    
    @staticmethod
    def get_overloaded_circuits() -> str:
        return """
        MATCH (c:Circuit {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(c)
        WITH c, sum(coalesce(comp.current_rating, 0)) as total_load
        WHERE total_load > c.max_current
        RETURN c, total_load, (total_load / c.max_current) * 100 as overload_percentage
        ORDER BY overload_percentage DESC
        """
    
    @staticmethod
    def get_circuit_topology() -> str:
        return """
        MATCH (circuit:Circuit {id: $circuit_id, vehicle_signature: $vehicle_signature})
        MATCH (circuit)<-[:PART_OF]-(component:Component)
        OPTIONAL MATCH (component)-[rel:CONNECTS_TO]->(connected:Component)-[:PART_OF]->(circuit)
        RETURN circuit, collect(DISTINCT component) as components,
               collect(DISTINCT {from: component.id, to: connected.id, type: 'CONNECTS_TO'}) as connections
        """