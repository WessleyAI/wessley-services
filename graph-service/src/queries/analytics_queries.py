"""Optimized Cypher queries for analytics and reporting"""

class AnalyticsQueries:
    @staticmethod
    def get_system_overview() -> str:
        return """
        MATCH (v:Vehicle {signature: $vehicle_signature})
        OPTIONAL MATCH (c:Component {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (circuit:Circuit {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (z:Zone {vehicle_signature: $vehicle_signature})
        RETURN v, count(DISTINCT c) as component_count,
               count(DISTINCT circuit) as circuit_count,
               count(DISTINCT z) as zone_count,
               count(CASE WHEN c.position IS NOT NULL THEN 1 END) as positioned_components
        """
    
    @staticmethod
    def get_component_type_distribution() -> str:
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        RETURN c.type as component_type, count(c) as count,
               avg(c.voltage_rating) as avg_voltage,
               avg(c.current_rating) as avg_current
        ORDER BY count DESC
        """
    
    @staticmethod
    def get_power_flow_analysis() -> str:
        return """
        MATCH (ps:Component {vehicle_signature: $vehicle_signature})
        WHERE ps.type IN ['battery', 'alternator', 'power_supply']
        OPTIONAL MATCH (load:Component)-[:POWERED_BY*1..5]->(ps)
        WHERE load.vehicle_signature = $vehicle_signature
        RETURN ps.id as source_id, ps.name as source_name,
               count(DISTINCT load) as loads_powered,
               sum(coalesce(load.current_rating, 0)) as total_load_current
        ORDER BY total_load_current DESC
        """
    
    @staticmethod
    def get_data_quality_metrics() -> str:
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        RETURN count(c) as total_components,
               count(CASE WHEN c.position IS NOT NULL THEN 1 END) as with_position,
               count(CASE WHEN c.voltage_rating IS NOT NULL THEN 1 END) as with_voltage,
               count(CASE WHEN c.current_rating IS NOT NULL THEN 1 END) as with_current,
               count(CASE WHEN NOT (c)-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]-() 
                         AND NOT ()-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]->(c) THEN 1 END) as orphaned
        """
    
    @staticmethod
    def get_circuit_health_summary() -> str:
        return """
        MATCH (circuit:Circuit {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (comp:Component)-[:PART_OF]->(circuit)
        WITH circuit, sum(coalesce(comp.current_rating, 0)) as total_load
        RETURN count(circuit) as total_circuits,
               count(CASE WHEN total_load > circuit.max_current THEN 1 END) as overloaded_circuits,
               count(CASE WHEN circuit.safety_critical = true THEN 1 END) as safety_critical_circuits,
               avg(total_load / circuit.max_current * 100) as avg_load_percentage
        """