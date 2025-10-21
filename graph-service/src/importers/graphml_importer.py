"""GraphML data importer for electrical system data"""

import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from ..models.schemas.import_schema import GraphMLImportSchema, ImportMetadata
from ..utils.neo4j_utils import Neo4jClient

class GraphMLImporter:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client
    
    async def import_data(self, schema: GraphMLImportSchema) -> Dict[str, Any]:
        """Import GraphML data into Neo4j"""
        try:
            root = ET.fromstring(schema.graphml_content)
            
            # Extract nodes and edges from GraphML
            nodes = self._extract_nodes(root, schema.metadata.vehicle_signature)
            edges = self._extract_edges(root, schema.metadata.vehicle_signature)
            
            # Import nodes
            node_results = await self._import_nodes(nodes)
            
            # Import edges
            edge_results = await self._import_edges(edges)
            
            return {
                "status": "success",
                "nodes_imported": len(nodes),
                "edges_imported": len(edges),
                "vehicle_signature": schema.metadata.vehicle_signature,
                "source_file": schema.metadata.source_file
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "vehicle_signature": schema.metadata.vehicle_signature
            }
    
    def _extract_nodes(self, root: ET.Element, vehicle_signature: str) -> List[Dict[str, Any]]:
        """Extract nodes from GraphML"""
        nodes = []
        ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        for node in root.findall('.//graphml:node', ns):
            node_id = node.get('id')
            node_data = {
                'id': node_id,
                'vehicle_signature': vehicle_signature,
                'label': 'Component'  # Default label
            }
            
            # Extract data attributes
            for data in node.findall('graphml:data', ns):
                key = data.get('key')
                value = data.text
                if key and value:
                    node_data[key] = value
            
            nodes.append(node_data)
        
        return nodes
    
    def _extract_edges(self, root: ET.Element, vehicle_signature: str) -> List[Dict[str, Any]]:
        """Extract edges from GraphML"""
        edges = []
        ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        for edge in root.findall('.//graphml:edge', ns):
            source = edge.get('source')
            target = edge.get('target')
            
            edge_data = {
                'source': source,
                'target': target,
                'vehicle_signature': vehicle_signature,
                'type': 'CONNECTED_TO'  # Default relationship
            }
            
            # Extract data attributes
            for data in edge.findall('graphml:data', ns):
                key = data.get('key')
                value = data.text
                if key and value:
                    edge_data[key] = value
            
            edges.append(edge_data)
        
        return edges
    
    async def _import_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import nodes into Neo4j"""
        query = """
        UNWIND $nodes as node
        MERGE (n:Component {id: node.id, vehicle_signature: node.vehicle_signature})
        SET n += node
        RETURN count(n) as nodes_created
        """
        
        result = await self.neo4j_client.run(query, nodes=nodes)
        return result
    
    async def _import_edges(self, edges: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import edges into Neo4j"""
        query = """
        UNWIND $edges as edge
        MATCH (source:Component {id: edge.source, vehicle_signature: edge.vehicle_signature})
        MATCH (target:Component {id: edge.target, vehicle_signature: edge.vehicle_signature})
        MERGE (source)-[r:CONNECTED_TO]->(target)
        SET r += edge
        RETURN count(r) as edges_created
        """
        
        result = await self.neo4j_client.run(query, edges=edges)
        return result