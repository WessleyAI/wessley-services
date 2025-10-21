"""
Optimized Cypher queries for component operations
"""

from typing import Dict, List, Optional, Any


class ComponentQueries:
    """
    Collection of optimized Cypher queries for component operations
    """
    
    @staticmethod
    def get_component_by_id() -> str:
        """Get component by ID with vehicle signature"""
        return """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        RETURN c
        """
    
    @staticmethod
    def get_components_by_type() -> str:
        """Get all components of a specific type"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature, type: $component_type})
        RETURN c
        ORDER BY c.name
        """
    
    @staticmethod
    def get_components_by_zone() -> str:
        """Get all components in a specific zone"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature, zone: $zone})
        RETURN c
        ORDER BY c.type, c.name
        """
    
    @staticmethod
    def get_components_with_connections() -> str:
        """Get components with their direct connections"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        OPTIONAL MATCH (c)-[r_out:CONNECTS_TO]->(connected_out:Component)
        WHERE connected_out.vehicle_signature = $vehicle_signature
        
        OPTIONAL MATCH (connected_in:Component)-[r_in:CONNECTS_TO]->(c)
        WHERE connected_in.vehicle_signature = $vehicle_signature
        
        RETURN c,
               collect(DISTINCT {
                   direction: 'outgoing',
                   relationship: r_out,
                   component: connected_out
               }) as outgoing_connections,
               collect(DISTINCT {
                   direction: 'incoming',
                   relationship: r_in,
                   component: connected_in
               }) as incoming_connections
        ORDER BY c.type, c.name
        """
    
    @staticmethod
    def get_components_with_power_relationships() -> str:
        """Get components with their power supply relationships"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Power sources (what powers this component)
        OPTIONAL MATCH (c)-[r_powered:POWERED_BY]->(power_source:Component)
        WHERE power_source.vehicle_signature = $vehicle_signature
        
        // Power loads (what this component powers)
        OPTIONAL MATCH (power_load:Component)-[r_powers:POWERED_BY]->(c)
        WHERE power_load.vehicle_signature = $vehicle_signature
        
        RETURN c,
               collect(DISTINCT {
                   relationship: r_powered,
                   source: power_source
               }) as power_sources,
               collect(DISTINCT {
                   relationship: r_powers,
                   load: power_load
               }) as power_loads
        ORDER BY c.type, c.name
        """
    
    @staticmethod
    def get_components_without_position() -> str:
        """Get components that don't have spatial position data"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.position IS NULL
        RETURN c
        ORDER BY c.zone, c.type, c.name
        """
    
    @staticmethod
    def get_orphaned_components() -> str:
        """Get components with no relationships (orphaned)"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE NOT (c)-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]-() 
          AND NOT ()-[:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]->(c)
        RETURN c
        ORDER BY c.type, c.name
        """
    
    @staticmethod
    def get_components_by_electrical_rating() -> str:
        """Get components filtered by electrical ratings"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE ($min_voltage IS NULL OR c.voltage_rating >= $min_voltage)
          AND ($max_voltage IS NULL OR c.voltage_rating <= $max_voltage)
          AND ($min_current IS NULL OR c.current_rating >= $min_current)
          AND ($max_current IS NULL OR c.current_rating <= $max_current)
        RETURN c
        ORDER BY c.voltage_rating DESC, c.current_rating DESC
        """
    
    @staticmethod
    def get_high_power_components() -> str:
        """Get components with high power consumption (> threshold)"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.voltage_rating IS NOT NULL 
          AND c.current_rating IS NOT NULL
          AND (c.voltage_rating * c.current_rating) > $power_threshold
        
        WITH c, (c.voltage_rating * c.current_rating) as power_consumption
        
        RETURN c, power_consumption
        ORDER BY power_consumption DESC
        """
    
    @staticmethod
    def get_components_with_spatial_data() -> str:
        """Get components with complete spatial information"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.position IS NOT NULL
        
        OPTIONAL MATCH (c)-[:LOCATED_IN]->(z:Zone)
        WHERE z.vehicle_signature = $vehicle_signature
        
        RETURN c, z,
               c.position[0] as x,
               c.position[1] as y, 
               c.position[2] as z
        ORDER BY z.name, c.type, c.name
        """
    
    @staticmethod
    def get_components_by_manufacturer() -> str:
        """Get components grouped by manufacturer"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.manufacturer IS NOT NULL
        
        RETURN c.manufacturer as manufacturer,
               collect(c) as components,
               count(c) as component_count,
               collect(DISTINCT c.type) as component_types
        ORDER BY component_count DESC, manufacturer
        """
    
    @staticmethod
    def search_components_by_name() -> str:
        """Search components by name pattern"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.name =~ $name_pattern
        RETURN c
        ORDER BY c.name
        """
    
    @staticmethod
    def get_component_network_neighbors() -> str:
        """Get all components within N hops of a given component"""
        return """
        MATCH (start:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        
        MATCH path = (start)-[:CONNECTS_TO|POWERED_BY|CONTROLS*1..$max_hops]-(neighbor:Component)
        WHERE neighbor.vehicle_signature = $vehicle_signature
          AND neighbor.id <> $component_id
        
        RETURN DISTINCT neighbor,
               min(length(path)) as min_distance,
               collect(DISTINCT type(last(relationships(path)))) as relationship_types
        ORDER BY min_distance, neighbor.type, neighbor.name
        """
    
    @staticmethod
    def get_critical_components() -> str:
        """Get components marked as critical or with high dependency count"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Count dependencies (components that depend on this one)
        OPTIONAL MATCH (dependent:Component)-[:POWERED_BY|CONTROLS*1..3]->(c)
        WHERE dependent.vehicle_signature = $vehicle_signature
        
        WITH c, count(DISTINCT dependent) as dependency_count
        
        // Filter for critical components
        WHERE (exists(c.is_critical) AND c.is_critical = true) 
           OR dependency_count > $dependency_threshold
           OR c.type IN ['battery', 'alternator', 'ecu']
        
        RETURN c, dependency_count,
               CASE 
                   WHEN exists(c.is_critical) AND c.is_critical THEN 'explicitly_critical'
                   WHEN dependency_count > $dependency_threshold THEN 'high_dependency'
                   WHEN c.type IN ['battery', 'alternator', 'ecu'] THEN 'critical_type'
                   ELSE 'other'
               END as criticality_reason
        ORDER BY dependency_count DESC, c.type
        """
    
    @staticmethod
    def get_component_circuits() -> str:
        """Get all circuits a component belongs to"""
        return """
        MATCH (c:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        MATCH (c)-[r:PART_OF]->(circuit:Circuit)
        WHERE circuit.vehicle_signature = $vehicle_signature
        
        RETURN c, circuit, r.role as role, r.is_critical as is_critical_to_circuit
        ORDER BY circuit.name
        """
    
    @staticmethod
    def get_components_summary_stats() -> str:
        """Get summary statistics for components"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        RETURN 
            count(c) as total_components,
            count(DISTINCT c.type) as unique_types,
            count(DISTINCT c.zone) as zones_used,
            count(DISTINCT c.manufacturer) as manufacturers,
            count(CASE WHEN c.position IS NOT NULL THEN 1 END) as components_with_position,
            count(CASE WHEN c.voltage_rating IS NOT NULL THEN 1 END) as components_with_voltage,
            count(CASE WHEN c.current_rating IS NOT NULL THEN 1 END) as components_with_current,
            avg(c.voltage_rating) as avg_voltage,
            avg(c.current_rating) as avg_current,
            max(c.voltage_rating) as max_voltage,
            max(c.current_rating) as max_current
        """
    
    @staticmethod
    def bulk_update_components() -> str:
        """Bulk update multiple components"""
        return """
        UNWIND $updates as update
        MATCH (c:Component {id: update.component_id, vehicle_signature: update.vehicle_signature})
        SET c += update.properties
        SET c.updated_at = timestamp()
        RETURN c.id as updated_id
        """
    
    @staticmethod
    def get_component_compatibility() -> str:
        """Find components compatible with given electrical specifications"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.voltage_rating >= $min_voltage 
          AND c.voltage_rating <= $max_voltage
          AND ($current_requirement IS NULL OR c.current_rating >= $current_requirement)
          AND ($component_types IS NULL OR c.type IN $component_types)
        
        // Calculate compatibility score
        WITH c,
             CASE 
                 WHEN c.voltage_rating = $target_voltage THEN 1.0
                 ELSE 1.0 - abs(c.voltage_rating - $target_voltage) / $target_voltage
             END as voltage_compatibility,
             CASE
                 WHEN $current_requirement IS NULL THEN 1.0
                 WHEN c.current_rating >= $current_requirement * 2 THEN 1.0
                 WHEN c.current_rating >= $current_requirement THEN 0.8
                 ELSE 0.3
             END as current_compatibility
        
        WITH c, (voltage_compatibility + current_compatibility) / 2 as compatibility_score
        
        RETURN c, compatibility_score
        ORDER BY compatibility_score DESC, c.name
        LIMIT $limit
        """
    
    @staticmethod
    def get_component_replacement_candidates() -> str:
        """Find replacement candidates for a component"""
        return """
        MATCH (original:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        
        // Find components of the same type
        MATCH (candidate:Component {vehicle_signature: $vehicle_signature, type: original.type})
        WHERE candidate.id <> original.id
        
        // Check electrical compatibility
        WHERE (candidate.voltage_rating IS NULL OR original.voltage_rating IS NULL 
               OR abs(candidate.voltage_rating - original.voltage_rating) <= $voltage_tolerance)
          AND (candidate.current_rating IS NULL OR original.current_rating IS NULL
               OR candidate.current_rating >= original.current_rating * 0.8)
        
        // Calculate similarity score
        WITH original, candidate,
             CASE 
                 WHEN candidate.manufacturer = original.manufacturer THEN 0.3 
                 ELSE 0.0 
             END +
             CASE 
                 WHEN candidate.part_number = original.part_number THEN 0.4
                 ELSE 0.0
             END +
             CASE
                 WHEN candidate.voltage_rating = original.voltage_rating THEN 0.2
                 ELSE 0.0
             END +
             CASE
                 WHEN candidate.current_rating >= original.current_rating THEN 0.1
                 ELSE 0.0
             END as similarity_score
        
        RETURN candidate, similarity_score
        ORDER BY similarity_score DESC, candidate.name
        LIMIT $limit
        """
    
    @staticmethod
    def get_components_needing_inspection() -> str:
        """Get components that may need inspection based on criteria"""
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        
        // Count relationships to assess connectivity
        OPTIONAL MATCH (c)-[r]-()
        WITH c, count(r) as relationship_count
        
        // Identify components needing inspection
        WHERE relationship_count = 0  // Orphaned
           OR c.position IS NULL      // Missing spatial data
           OR c.voltage_rating IS NULL // Missing electrical data
           OR c.current_rating IS NULL
           OR (exists(c.updated_at) AND c.updated_at < $cutoff_date)
        
        RETURN c, relationship_count,
               CASE 
                   WHEN relationship_count = 0 THEN 'orphaned'
                   WHEN c.position IS NULL THEN 'missing_position'
                   WHEN c.voltage_rating IS NULL OR c.current_rating IS NULL THEN 'missing_electrical'
                   WHEN exists(c.updated_at) AND c.updated_at < $cutoff_date THEN 'outdated'
                   ELSE 'other'
               END as inspection_reason
        ORDER BY inspection_reason, c.type, c.name
        """


class ComponentQueryBuilder:
    """
    Dynamic query builder for component queries
    """
    
    def __init__(self):
        self.queries = ComponentQueries()
    
    def build_filtered_query(
        self,
        vehicle_signature: str,
        filters: Optional[Dict[str, Any]] = None,
        include_relationships: bool = False,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> tuple[str, Dict[str, Any]]:
        """Build a dynamic filtered query for components"""
        
        filters = filters or {}
        params = {"vehicle_signature": vehicle_signature}
        
        # Base query
        base_query = "MATCH (c:Component {vehicle_signature: $vehicle_signature})"
        
        # Build WHERE clauses
        where_clauses = []
        
        if "component_types" in filters and filters["component_types"]:
            where_clauses.append("c.type IN $component_types")
            params["component_types"] = filters["component_types"]
        
        if "zone" in filters and filters["zone"]:
            where_clauses.append("c.zone = $zone")
            params["zone"] = filters["zone"]
        
        if "manufacturer" in filters and filters["manufacturer"]:
            where_clauses.append("c.manufacturer = $manufacturer")
            params["manufacturer"] = filters["manufacturer"]
        
        if "has_position" in filters:
            if filters["has_position"]:
                where_clauses.append("c.position IS NOT NULL")
            else:
                where_clauses.append("c.position IS NULL")
        
        if "voltage_min" in filters and filters["voltage_min"] is not None:
            where_clauses.append("c.voltage_rating >= $voltage_min")
            params["voltage_min"] = filters["voltage_min"]
        
        if "voltage_max" in filters and filters["voltage_max"] is not None:
            where_clauses.append("c.voltage_rating <= $voltage_max")
            params["voltage_max"] = filters["voltage_max"]
        
        if "current_min" in filters and filters["current_min"] is not None:
            where_clauses.append("c.current_rating >= $current_min")
            params["current_min"] = filters["current_min"]
        
        if "current_max" in filters and filters["current_max"] is not None:
            where_clauses.append("c.current_rating <= $current_max")
            params["current_max"] = filters["current_max"]
        
        if "name_pattern" in filters and filters["name_pattern"]:
            where_clauses.append("c.name =~ $name_pattern")
            params["name_pattern"] = f"(?i).*{filters['name_pattern']}.*"
        
        # Add WHERE clause if any filters
        where_clause = ""
        if where_clauses:
            where_clause = " WHERE " + " AND ".join(where_clauses)
        
        # Add relationships if requested
        relationship_clause = ""
        if include_relationships:
            relationship_clause = """
            OPTIONAL MATCH (c)-[r:CONNECTS_TO|POWERED_BY|CONTROLS|PART_OF]->(related)
            """
        
        # Build return clause
        return_clause = "RETURN c"
        if include_relationships:
            return_clause += ", collect({type: type(r), relationship: r, related: related}) as relationships"
        
        # Build order clause
        order_clause = " ORDER BY c.type, c.name"
        
        # Build limit/offset clause
        limit_clause = ""
        if limit:
            limit_clause = f" LIMIT {limit}"
            if offset:
                limit_clause = f" SKIP {offset}" + limit_clause
        
        # Combine all parts
        query = base_query + where_clause + relationship_clause + return_clause + order_clause + limit_clause
        
        return query, params