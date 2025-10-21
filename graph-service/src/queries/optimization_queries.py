"""Performance-optimized Cypher queries"""

class OptimizationQueries:
    @staticmethod
    def bulk_component_lookup() -> str:
        return """
        UNWIND $component_ids as comp_id
        MATCH (c:Component {id: comp_id, vehicle_signature: $vehicle_signature})
        RETURN c
        """
    
    @staticmethod
    def component_with_relationships_optimized() -> str:
        return """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (c)-[r:CONNECTS_TO|POWERED_BY|CONTROLS]->(related:Component {vehicle_signature: $vehicle_signature})
        RETURN c, collect({type: type(r), related: related}) as relationships
        """
    
    @staticmethod
    def power_tree_traversal() -> str:
        return """
        MATCH (source:Component {id: $source_id, vehicle_signature: $vehicle_signature})
        CALL apoc.path.expandConfig(source, {
            relationshipFilter: "POWERED_BY>",
            labelFilter: "+Component",
            maxLevel: $max_depth,
            uniqueness: "NODE_GLOBAL"
        }) YIELD path
        RETURN path
        """
    
    @staticmethod
    def shortest_path_optimized() -> str:
        return """
        MATCH (start:Component {id: $from_id, vehicle_signature: $vehicle_signature})
        MATCH (end:Component {id: $to_id, vehicle_signature: $vehicle_signature})
        CALL apoc.algo.dijkstra(start, end, 'CONNECTS_TO|POWERED_BY>', 'distance', 1.0) 
        YIELD path, weight
        RETURN path, weight
        """
    
    @staticmethod
    def batch_update_positions() -> str:
        return """
        UNWIND $updates as update
        MATCH (c:Component {id: update.component_id, vehicle_signature: update.vehicle_signature})
        SET c.position = [update.x, update.y, update.z]
        SET c.updated_at = timestamp()
        RETURN count(c) as updated_count
        """
    
    @staticmethod
    def indexed_component_search() -> str:
        return """
        CALL db.index.fulltext.queryNodes("component_search", $search_term) 
        YIELD node, score
        WHERE node.vehicle_signature = $vehicle_signature
        RETURN node, score
        ORDER BY score DESC
        LIMIT $limit
        """