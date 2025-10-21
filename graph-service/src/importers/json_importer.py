"""JSON data importer for electrical system data"""

import json
from typing import Dict, List, Any, Optional
from ..models.schemas.import_schema import JSONImportSchema, ImportMetadata
from ..utils.neo4j_utils import Neo4jClient

class JSONImporter:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client
    
    async def import_data(self, schema: JSONImportSchema) -> Dict[str, Any]:
        """Import JSON data into Neo4j"""
        try:
            # Validate JSON structure
            if not self._validate_json_structure(schema.json_data):
                return {
                    "status": "error",
                    "error": "Invalid JSON structure",
                    "vehicle_signature": schema.metadata.vehicle_signature
                }
            
            # Extract components and connections
            components = schema.json_data.get('components', [])
            connections = schema.json_data.get('connections', [])
            
            # Enrich with vehicle signature
            components = self._enrich_components(components, schema.metadata.vehicle_signature)
            connections = self._enrich_connections(connections, schema.metadata.vehicle_signature)
            
            # Import components
            component_results = await self._import_components(components)
            
            # Import connections
            connection_results = await self._import_connections(connections)
            
            return {
                "status": "success",
                "components_imported": len(components),
                "connections_imported": len(connections),
                "vehicle_signature": schema.metadata.vehicle_signature,
                "source_file": schema.metadata.source_file
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "vehicle_signature": schema.metadata.vehicle_signature
            }
    
    def _validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """Validate required JSON structure"""
        required_fields = ['components']
        return all(field in data for field in required_fields)
    
    def _enrich_components(self, components: List[Dict[str, Any]], vehicle_signature: str) -> List[Dict[str, Any]]:
        """Add vehicle signature to components"""
        enriched = []
        for component in components:
            enriched_component = component.copy()
            enriched_component['vehicle_signature'] = vehicle_signature
            if 'type' not in enriched_component:
                enriched_component['type'] = 'unknown'
            enriched.append(enriched_component)
        return enriched
    
    def _enrich_connections(self, connections: List[Dict[str, Any]], vehicle_signature: str) -> List[Dict[str, Any]]:
        """Add vehicle signature to connections"""
        enriched = []
        for connection in connections:
            enriched_connection = connection.copy()
            enriched_connection['vehicle_signature'] = vehicle_signature
            if 'type' not in enriched_connection:
                enriched_connection['type'] = 'CONNECTED_TO'
            enriched.append(enriched_connection)
        return enriched
    
    async def _import_components(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import components into Neo4j"""
        query = """
        UNWIND $components as comp
        MERGE (c:Component {id: comp.id, vehicle_signature: comp.vehicle_signature})
        SET c += comp,
            c.created_at = datetime(),
            c.updated_at = datetime()
        RETURN count(c) as components_created
        """
        
        result = await self.neo4j_client.run(query, components=components)
        return result
    
    async def _import_connections(self, connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import connections into Neo4j"""
        query = """
        UNWIND $connections as conn
        MATCH (source:Component {id: conn.from, vehicle_signature: conn.vehicle_signature})
        MATCH (target:Component {id: conn.to, vehicle_signature: conn.vehicle_signature})
        CALL apoc.merge.relationship(source, conn.type, {}, {
            vehicle_signature: conn.vehicle_signature,
            created_at: datetime()
        }, target) YIELD rel
        RETURN count(rel) as connections_created
        """
        
        result = await self.neo4j_client.run(query, connections=connections)
        return result
    
    async def validate_import(self, vehicle_signature: str) -> Dict[str, Any]:
        """Validate imported data integrity"""
        query = """
        MATCH (c:Component {vehicle_signature: $vehicle_signature})
        OPTIONAL MATCH (c)-[r]-(other:Component {vehicle_signature: $vehicle_signature})
        RETURN 
            count(DISTINCT c) as total_components,
            count(DISTINCT r) as total_relationships,
            count(DISTINCT c.type) as component_types,
            collect(DISTINCT c.type) as types_list
        """
        
        result = await self.neo4j_client.run(query, vehicle_signature=vehicle_signature)
        return result