"""
Data import service for processing various data formats (GraphML, JSON, CSV)
"""

import logging
import xml.etree.ElementTree as ET
import json
import csv
import io
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from ..models.schemas.import_schema import (
    ImportMetadata,
    GraphMLImportSchema,
    JSONImportSchema,
    CSVImportSchema,
    NDJSONImportSchema,
    ImportValidationResult,
    ImportExecutionResult,
    ImportMode,
    ValidationLevel
)
from ..models.schemas.component_schema import ComponentValidationSchema
from ..models.schemas.circuit_schema import CircuitValidationSchema
from ..utils.neo4j_utils import Neo4jClient
from ..utils.data_validation import DataValidator
from .graph_service import GraphService

logger = logging.getLogger(__name__)


class ImportService:
    """
    Service for importing electrical system data from various formats
    """
    
    def __init__(
        self, 
        neo4j_client: Neo4jClient, 
        graph_service: GraphService,
        data_validator: DataValidator
    ):
        self.neo4j = neo4j_client
        self.graph_service = graph_service
        self.validator = data_validator
    
    # ==================== GraphML Import ====================
    
    async def import_graphml(
        self, 
        import_schema: GraphMLImportSchema
    ) -> ImportExecutionResult:
        """Import electrical system data from GraphML format"""
        
        start_time = datetime.utcnow()
        
        try:
            # Validate GraphML content
            validation_result = await self._validate_graphml(import_schema)
            
            if not validation_result.is_valid:
                return ImportExecutionResult(
                    success=False,
                    records_processed=0,
                    records_created=0,
                    records_updated=0,
                    records_skipped=0,
                    records_failed=validation_result.invalid_records,
                    execution_time=0.0,
                    errors=[{"type": "validation_failed", "details": validation_result.errors}]
                )
            
            # Parse GraphML
            components, circuits, relationships = await self._parse_graphml(import_schema)
            
            # Execute import
            result = await self._execute_batch_import(
                import_schema.metadata,
                components,
                circuits,
                relationships
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"GraphML import completed: {result.records_created} created, {result.records_failed} failed")
            return result
            
        except Exception as e:
            logger.error(f"GraphML import failed: {e}")
            return ImportExecutionResult(
                success=False,
                records_processed=0,
                records_created=0,
                records_updated=0,
                records_skipped=0,
                records_failed=0,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                errors=[{"type": "import_exception", "message": str(e)}]
            )
    
    async def _parse_graphml(
        self, 
        import_schema: GraphMLImportSchema
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Parse GraphML content into components, circuits, and relationships"""
        
        root = ET.fromstring(import_schema.graphml_content)
        
        # Handle namespaces
        namespaces = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        if import_schema.graphml_namespace:
            namespaces['graphml'] = import_schema.graphml_namespace
        
        components = []
        circuits = []
        relationships = []
        
        # Parse nodes (components and circuits)
        for node in root.findall('.//graphml:node', namespaces):
            node_data = self._extract_node_data(node, import_schema)
            
            if node_data.get('type') in ['circuit']:
                circuits.append(node_data)
            else:
                # Assume component if not circuit
                node_data['vehicle_signature'] = import_schema.metadata.vehicle_signature
                components.append(node_data)
        
        # Parse edges (relationships)
        for edge in root.findall('.//graphml:edge', namespaces):
            rel_data = self._extract_edge_data(edge, import_schema)
            rel_data['vehicle_signature'] = import_schema.metadata.vehicle_signature
            relationships.append(rel_data)
        
        return components, circuits, relationships
    
    def _extract_node_data(self, node_element: ET.Element, import_schema: GraphMLImportSchema) -> Dict[str, Any]:
        """Extract node data from GraphML node element"""
        
        node_data = {
            'id': node_element.get('id'),
            'vehicle_signature': import_schema.metadata.vehicle_signature
        }
        
        # Extract data elements
        for data_elem in node_element.findall('.//data'):
            key = data_elem.get('key')
            value = data_elem.text
            
            # Map common GraphML attributes to our schema
            if key == 'type':
                node_data['type'] = value
            elif key == 'name' or key == 'label':
                node_data['name'] = value
            elif key in ['x', 'y', 'z']:
                if 'position' not in node_data:
                    node_data['position'] = {}
                try:
                    node_data['position'][key] = float(value)
                except (ValueError, TypeError):
                    pass
            elif key in ['voltage', 'voltage_rating']:
                try:
                    node_data['voltage_rating'] = float(value)
                except (ValueError, TypeError):
                    pass
            elif key in ['current', 'current_rating']:
                try:
                    node_data['current_rating'] = float(value)
                except (ValueError, TypeError):
                    pass
            else:
                node_data[key] = value
        
        # Convert position dict to separate coordinates
        if 'position' in node_data and isinstance(node_data['position'], dict):
            pos = node_data['position']
            if 'x' in pos and 'y' in pos and 'z' in pos:
                node_data['position_x'] = pos['x']
                node_data['position_y'] = pos['y']
                node_data['position_z'] = pos['z']
            del node_data['position']
        
        return node_data
    
    def _extract_edge_data(self, edge_element: ET.Element, import_schema: GraphMLImportSchema) -> Dict[str, Any]:
        """Extract edge data from GraphML edge element"""
        
        edge_data = {
            'from_id': edge_element.get('source'),
            'to_id': edge_element.get('target'),
            'type': 'CONNECTS_TO'  # Default relationship type
        }
        
        # Extract data elements
        for data_elem in edge_element.findall('.//data'):
            key = data_elem.get('key')
            value = data_elem.text
            
            if key == 'type' or key == 'relationship':
                edge_data['type'] = value
            elif key == 'wire_gauge':
                edge_data['wire_gauge'] = value
            elif key == 'wire_color':
                edge_data['wire_color'] = value
            elif key == 'voltage':
                try:
                    edge_data['voltage'] = float(value)
                except (ValueError, TypeError):
                    pass
            else:
                edge_data[key] = value
        
        return edge_data
    
    async def _validate_graphml(self, import_schema: GraphMLImportSchema) -> ImportValidationResult:
        """Validate GraphML import data"""
        
        start_time = datetime.utcnow()
        errors = []
        warnings = []
        
        try:
            # Basic XML validation
            ET.fromstring(import_schema.graphml_content)
            
            # Parse and validate components/circuits
            components, circuits, relationships = await self._parse_graphml(import_schema)
            
            valid_components = 0
            valid_circuits = 0
            valid_relationships = 0
            
            # Validate components
            for comp_data in components:
                try:
                    ComponentValidationSchema(**comp_data)
                    valid_components += 1
                except Exception as e:
                    errors.append({
                        "type": "component_validation",
                        "component_id": comp_data.get('id'),
                        "error": str(e)
                    })
            
            # Validate circuits
            for circuit_data in circuits:
                try:
                    CircuitValidationSchema(**circuit_data)
                    valid_circuits += 1
                except Exception as e:
                    errors.append({
                        "type": "circuit_validation",
                        "circuit_id": circuit_data.get('id'),
                        "error": str(e)
                    })
            
            # Validate relationships
            for rel_data in relationships:
                if not rel_data.get('from_id') or not rel_data.get('to_id'):
                    errors.append({
                        "type": "relationship_validation",
                        "error": "Missing from_id or to_id"
                    })
                else:
                    valid_relationships += 1
            
            total_records = len(components) + len(circuits) + len(relationships)
            valid_records = valid_components + valid_circuits + valid_relationships
            
            return ImportValidationResult(
                is_valid=len(errors) == 0,
                total_records=total_records,
                valid_records=valid_records,
                invalid_records=total_records - valid_records,
                errors=errors,
                warnings=warnings,
                validation_time=(datetime.utcnow() - start_time).total_seconds(),
                component_count=len(components),
                circuit_count=len(circuits),
                relationship_count=len(relationships)
            )
            
        except ET.ParseError as e:
            return ImportValidationResult(
                is_valid=False,
                total_records=0,
                valid_records=0,
                invalid_records=1,
                errors=[{"type": "xml_parse_error", "error": str(e)}],
                warnings=[],
                validation_time=(datetime.utcnow() - start_time).total_seconds(),
                component_count=0,
                circuit_count=0,
                relationship_count=0
            )
    
    # ==================== JSON Import ====================
    
    async def import_json(self, import_schema: JSONImportSchema) -> ImportExecutionResult:
        """Import electrical system data from JSON format"""
        
        start_time = datetime.utcnow()
        
        try:
            # Execute import
            result = await self._execute_batch_import(
                import_schema.metadata,
                import_schema.components or [],
                import_schema.circuits or [],
                import_schema.relationships or []
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"JSON import completed: {result.records_created} created, {result.records_failed} failed")
            return result
            
        except Exception as e:
            logger.error(f"JSON import failed: {e}")
            return ImportExecutionResult(
                success=False,
                records_processed=0,
                records_created=0,
                records_updated=0,
                records_skipped=0,
                records_failed=0,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                errors=[{"type": "import_exception", "message": str(e)}]
            )
    
    # ==================== CSV Import ====================
    
    async def import_csv(self, import_schema: CSVImportSchema) -> ImportExecutionResult:
        """Import electrical system data from CSV format"""
        
        start_time = datetime.utcnow()
        
        try:
            # Parse CSV content
            data_records = await self._parse_csv(import_schema)
            
            # Convert to appropriate format based on entity type
            components = []
            circuits = []
            relationships = []
            
            if import_schema.entity_type == 'components':
                components = data_records
            elif import_schema.entity_type == 'circuits':
                circuits = data_records
            elif import_schema.entity_type == 'relationships':
                relationships = data_records
            
            # Execute import
            result = await self._execute_batch_import(
                import_schema.metadata,
                components,
                circuits,
                relationships
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"CSV import completed: {result.records_created} created, {result.records_failed} failed")
            return result
            
        except Exception as e:
            logger.error(f"CSV import failed: {e}")
            return ImportExecutionResult(
                success=False,
                records_processed=0,
                records_created=0,
                records_updated=0,
                records_skipped=0,
                records_failed=0,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                errors=[{"type": "import_exception", "message": str(e)}]
            )
    
    async def _parse_csv(self, import_schema: CSVImportSchema) -> List[Dict[str, Any]]:
        """Parse CSV content into records"""
        
        csv_file = io.StringIO(import_schema.csv_content)
        reader = csv.DictReader(
            csv_file,
            delimiter=import_schema.delimiter,
            quotechar=import_schema.quote_char
        )
        
        records = []
        column_mapping = {v: k for k, v in import_schema.column_mapping.items()}  # Reverse mapping
        
        for row in reader:
            record = {'vehicle_signature': import_schema.metadata.vehicle_signature}
            
            for csv_column, value in row.items():
                if csv_column in column_mapping:
                    field_name = column_mapping[csv_column]
                    
                    # Handle null values
                    if value in import_schema.null_values:
                        value = None
                    
                    # Type conversion for numeric fields
                    if field_name in ['voltage_rating', 'current_rating', 'position_x', 'position_y', 'position_z']:
                        try:
                            value = float(value) if value is not None else None
                        except (ValueError, TypeError):
                            value = None
                    
                    record[field_name] = value
            
            records.append(record)
        
        return records
    
    # ==================== NDJSON Import ====================
    
    async def import_ndjson(self, import_schema: NDJSONImportSchema) -> ImportExecutionResult:
        """Import electrical system data from NDJSON format"""
        
        start_time = datetime.utcnow()
        
        try:
            # Parse NDJSON content
            data_records = await self._parse_ndjson(import_schema)
            
            # Separate records by type
            components = []
            circuits = []
            relationships = []
            
            for record in data_records:
                record['vehicle_signature'] = import_schema.metadata.vehicle_signature
                
                if import_schema.entity_type == 'mixed':
                    # Determine type from record
                    if record.get('circuit_type') or record.get('max_current'):
                        circuits.append(record)
                    elif record.get('from_id') and record.get('to_id'):
                        relationships.append(record)
                    else:
                        components.append(record)
                elif import_schema.entity_type == 'components':
                    components.append(record)
                elif import_schema.entity_type == 'circuits':
                    circuits.append(record)
                elif import_schema.entity_type == 'relationships':
                    relationships.append(record)
            
            # Execute import
            result = await self._execute_batch_import(
                import_schema.metadata,
                components,
                circuits,
                relationships
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time = execution_time
            
            logger.info(f"NDJSON import completed: {result.records_created} created, {result.records_failed} failed")
            return result
            
        except Exception as e:
            logger.error(f"NDJSON import failed: {e}")
            return ImportExecutionResult(
                success=False,
                records_processed=0,
                records_created=0,
                records_updated=0,
                records_skipped=0,
                records_failed=0,
                execution_time=(datetime.utcnow() - start_time).total_seconds(),
                errors=[{"type": "import_exception", "message": str(e)}]
            )
    
    async def _parse_ndjson(self, import_schema: NDJSONImportSchema) -> List[Dict[str, Any]]:
        """Parse NDJSON content into records"""
        
        records = []
        lines = import_schema.ndjson_content.strip().split('\n')
        
        for line_num, line in enumerate(lines):
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num + 1}: {e}")
        
        return records
    
    # ==================== Batch Import Execution ====================
    
    async def _execute_batch_import(
        self,
        metadata: ImportMetadata,
        components: List[Dict],
        circuits: List[Dict],
        relationships: List[Dict]
    ) -> ImportExecutionResult:
        """Execute batch import with transaction support"""
        
        result = ImportExecutionResult(
            success=True,
            records_processed=0,
            records_created=0,
            records_updated=0,
            records_skipped=0,
            records_failed=0,
            execution_time=0.0,
            errors=[],
            warnings=[],
            created_components=[],
            created_circuits=[],
            created_relationships=[]
        )
        
        # Start transaction for batch operation
        async with self.neo4j.transaction() as tx:
            try:
                # Import components
                for comp_data in components:
                    try:
                        if metadata.import_mode == ImportMode.CREATE_ONLY:
                            # Check if exists
                            existing = await self.graph_service.get_component(
                                comp_data['id'], comp_data['vehicle_signature']
                            )
                            if existing:
                                result.records_skipped += 1
                                continue
                        
                        # Validate component
                        comp_schema = ComponentValidationSchema(**comp_data)
                        
                        # Create or update component
                        component = await self.graph_service.create_component(comp_schema, validate=False)
                        
                        result.records_created += 1
                        result.created_components.append(component.id)
                        
                    except Exception as e:
                        result.records_failed += 1
                        result.errors.append({
                            "type": "component_import_error",
                            "component_id": comp_data.get('id'),
                            "error": str(e)
                        })
                    
                    result.records_processed += 1
                
                # Import circuits
                for circuit_data in circuits:
                    try:
                        # Validate circuit
                        circuit_schema = CircuitValidationSchema(**circuit_data)
                        
                        # Create circuit
                        circuit = await self.graph_service.create_circuit(circuit_schema, validate=False)
                        
                        result.records_created += 1
                        result.created_circuits.append(circuit.id)
                        
                    except Exception as e:
                        result.records_failed += 1
                        result.errors.append({
                            "type": "circuit_import_error",
                            "circuit_id": circuit_data.get('id'),
                            "error": str(e)
                        })
                    
                    result.records_processed += 1
                
                # Import relationships
                for rel_data in relationships:
                    try:
                        # Create appropriate relationship based on type
                        rel_type = rel_data.get('type', 'CONNECTS_TO')
                        
                        if rel_type == 'CONNECTS_TO':
                            # Create connection
                            from ..models.relationships import ConnectsToRelationship
                            connection_props = ConnectsToRelationship(
                                wire_gauge=rel_data.get('wire_gauge'),
                                wire_color=rel_data.get('wire_color'),
                                connection_type=rel_data.get('connection_type', 'direct')
                            )
                            
                            success = await self.graph_service.create_connection(
                                rel_data['from_id'],
                                rel_data['to_id'],
                                rel_data['vehicle_signature'],
                                connection_props
                            )
                            
                            if success:
                                result.records_created += 1
                                result.created_relationships.append(f"{rel_data['from_id']}->{rel_data['to_id']}")
                            else:
                                result.records_failed += 1
                        
                        # Add other relationship types as needed
                        
                    except Exception as e:
                        result.records_failed += 1
                        result.errors.append({
                            "type": "relationship_import_error",
                            "relationship": f"{rel_data.get('from_id')}->{rel_data.get('to_id')}",
                            "error": str(e)
                        })
                    
                    result.records_processed += 1
                
                # Commit transaction if successful
                await tx.commit()
                
            except Exception as e:
                # Rollback on error
                await tx.rollback()
                result.success = False
                result.errors.append({
                    "type": "transaction_error",
                    "error": str(e)
                })
        
        return result