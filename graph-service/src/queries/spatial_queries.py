"""Optimized Cypher queries for spatial operations"""

class SpatialQueries:
    @staticmethod
    def get_components_with_positions() -> str:
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.position IS NOT NULL
        OPTIONAL MATCH (c)-[:LOCATED_IN]->(z:Zone)
        RETURN c, z, c.position as position
        ORDER BY z.name, c.type, c.name
        """
    
    @staticmethod
    def get_spatial_clusters() -> str:
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        WHERE c.position IS NOT NULL
        WITH c, c.position[0] as x, c.position[1] as y, c.position[2] as z
        RETURN c, x, y, z
        ORDER BY x, y, z
        """
    
    @staticmethod
    def get_wire_routing_data() -> str:
        return """
        MATCH (c1:Component {vehicle_signature: $vehicle_signature})-[conn:CONNECTS_TO]->(c2:Component {vehicle_signature: $vehicle_signature})
        WHERE c1.position IS NOT NULL AND c2.position IS NOT NULL
        RETURN c1, c2, conn,
               sqrt((c1.position[0] - c2.position[0])^2 + 
                    (c1.position[1] - c2.position[1])^2 + 
                    (c1.position[2] - c2.position[2])^2) as wire_length
        ORDER BY wire_length DESC
        """
    
    @staticmethod
    def find_nearby_components() -> str:
        return """
        MATCH (center:Component {id: $component_id, vehicle_signature: $vehicle_signature})
        WHERE center.position IS NOT NULL
        MATCH (nearby:Component {vehicle_signature: $vehicle_signature})
        WHERE nearby.position IS NOT NULL AND nearby.id <> $component_id
        WITH center, nearby,
             sqrt((center.position[0] - nearby.position[0])^2 + 
                  (center.position[1] - nearby.position[1])^2 + 
                  (center.position[2] - nearby.position[2])^2) as distance
        WHERE distance <= $radius
        RETURN nearby, distance
        ORDER BY distance
        """