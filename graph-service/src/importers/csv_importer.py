"""CSV data importer for electrical system data"""

import csv
import io
from typing import Dict, List, Any, Optional
from ..models.schemas.import_schema import CSVImportSchema, ImportMetadata
from ..utils.neo4j_utils import Neo4jClient

class CSVImporter:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client
    
    async def import_data(self, schema: CSVImportSchema) -> Dict[str, Any]:
        """Import CSV data into Neo4j"""
        try:
            # Parse CSV content
            rows = self._parse_csv(schema.csv_content)
            
            if not rows:
                return {
                    "status": "error",
                    "error": "No data found in CSV",
                    "vehicle_signature": schema.metadata.vehicle_signature
                }
            
            # Map columns based on provided mapping
            mapped_data = self._map_columns(rows, schema.column_mapping)
            
            # Enrich with metadata
            enriched_data = self._enrich_data(mapped_data, schema)
            
            # Import based on entity type
            if schema.entity_type == "components":
                results = await self._import_components(enriched_data)
            elif schema.entity_type == "connections":
                results = await self._import_connections(enriched_data)
            elif schema.entity_type == "circuits":
                results = await self._import_circuits(enriched_data)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported entity type: {schema.entity_type}",
                    "vehicle_signature": schema.metadata.vehicle_signature
                }
            
            return {
                "status": "success",
                "records_imported": len(enriched_data),
                "entity_type": schema.entity_type,
                "vehicle_signature": schema.metadata.vehicle_signature,
                "source_file": schema.metadata.source_file
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "vehicle_signature": schema.metadata.vehicle_signature
            }
    
    def _parse_csv(self, csv_content: str) -> List[Dict[str, str]]:
        """Parse CSV content into list of dictionaries"""
        rows = []
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)
        
        for row in reader:
            # Clean empty values
            cleaned_row = {k: v for k, v in row.items() if v and v.strip()}
            if cleaned_row:
                rows.append(cleaned_row)
        
        return rows
    
    def _map_columns(self, rows: List[Dict[str, str]], column_mapping: Dict[str, str]) -> List[Dict[str, str]]:
        """Map CSV columns to expected field names"""
        mapped_rows = []
        
        for row in rows:
            mapped_row = {}
            for csv_col, target_field in column_mapping.items():
                if csv_col in row:
                    mapped_row[target_field] = row[csv_col]
            
            # Add unmapped columns as additional properties
            for col, value in row.items():
                if col not in column_mapping:
                    # Clean column name for Neo4j property
                    clean_col = col.lower().replace(' ', '_').replace('-', '_')
                    mapped_row[clean_col] = value
            
            mapped_rows.append(mapped_row)
        
        return mapped_rows
    
    def _enrich_data(self, data: List[Dict[str, str]], schema: CSVImportSchema) -> List[Dict[str, Any]]:
        """Enrich data with metadata and vehicle signature"""
        enriched = []
        
        for i, row in enumerate(data):
            enriched_row = row.copy()
            enriched_row['vehicle_signature'] = schema.metadata.vehicle_signature
            enriched_row['import_source'] = schema.metadata.source_file
            enriched_row['import_format'] = 'csv'
            
            # Generate ID if not present
            if 'id' not in enriched_row:
                enriched_row['id'] = f"{schema.entity_type}_{i+1}"
            
            enriched.append(enriched_row)
        
        return enriched
    
    async def _import_components(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import component data"""
        query = """
        UNWIND $data as comp
        MERGE (c:Component {id: comp.id, vehicle_signature: comp.vehicle_signature})
        SET c += comp,
            c.created_at = datetime(),
            c.updated_at = datetime()
        RETURN count(c) as components_created
        """
        
        result = await self.neo4j_client.run(query, data=data)
        return result
    
    async def _import_connections(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import connection data"""
        query = """
        UNWIND $data as conn
        MATCH (source:Component {id: conn.from_component, vehicle_signature: conn.vehicle_signature})
        MATCH (target:Component {id: conn.to_component, vehicle_signature: conn.vehicle_signature})
        MERGE (source)-[r:CONNECTED_TO]->(target)
        SET r += conn,
            r.created_at = datetime()
        RETURN count(r) as connections_created
        """
        
        result = await self.neo4j_client.run(query, data=data)
        return result
    
    async def _import_circuits(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import circuit data"""
        query = """
        UNWIND $data as circuit
        MERGE (c:Circuit {id: circuit.id, vehicle_signature: circuit.vehicle_signature})
        SET c += circuit,
            c.created_at = datetime(),
            c.updated_at = datetime()
        RETURN count(c) as circuits_created
        """
        
        result = await self.neo4j_client.run(query, data=data)
        return result
    
    def get_sample_mapping(self, entity_type: str) -> Dict[str, str]:
        """Get sample column mapping for entity type"""
        mappings = {
            "components": {
                "ID": "id",
                "Name": "name",
                "Type": "type",
                "Description": "description",
                "Part Number": "part_number",
                "Location": "location"
            },
            "connections": {
                "From": "from_component",
                "To": "to_component",
                "Wire Color": "wire_color",
                "Wire Gauge": "wire_gauge",
                "Signal Type": "signal_type"
            },
            "circuits": {
                "Circuit ID": "id",
                "Name": "name",
                "Type": "circuit_type",
                "Max Current": "max_current",
                "Voltage": "voltage"
            }
        }
        
        return mappings.get(entity_type, {})