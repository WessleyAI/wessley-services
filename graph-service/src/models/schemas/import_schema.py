"""
Import validation schemas for data import operations (GraphML, JSON, CSV)
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
import re
from enum import Enum

from .component_schema import ComponentValidationSchema, ComponentType
from .circuit_schema import CircuitValidationSchema, CircuitType


class ImportFormat(str, Enum):
    """Supported import formats"""
    GRAPHML = "graphml"
    JSON = "json"
    CSV = "csv"
    NDJSON = "ndjson"
    EXCEL = "excel"


class ImportMode(str, Enum):
    """Import operation modes"""
    CREATE_ONLY = "create_only"
    UPDATE_ONLY = "update_only"
    UPSERT = "upsert"
    REPLACE = "replace"
    MERGE = "merge"


class ValidationLevel(str, Enum):
    """Validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"
    SKIP = "skip"


class ImportMetadata(BaseModel):
    """Metadata for import operations"""
    
    source_file: str = Field(..., description="Source file name")
    format: ImportFormat = Field(..., description="Import format")
    vehicle_signature: str = Field(..., min_length=1, max_length=100)
    import_mode: ImportMode = Field(default=ImportMode.UPSERT)
    validation_level: ValidationLevel = Field(default=ValidationLevel.STRICT)
    
    # Optional metadata
    description: Optional[str] = Field(None, max_length=500)
    source_system: Optional[str] = Field(None, max_length=100)
    version: Optional[str] = Field(None, max_length=50)
    author: Optional[str] = Field(None, max_length=100)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Import settings
    batch_size: int = Field(default=1000, ge=1, le=10000)
    max_errors: int = Field(default=100, ge=0, le=1000)
    skip_duplicates: bool = Field(default=True)
    preserve_ids: bool = Field(default=True)
    auto_generate_missing: bool = Field(default=False)
    
    @validator('source_file')
    def validate_source_file(cls, v):
        """Validate source file name"""
        if not re.match(r'^[a-zA-Z0-9._-]+$', v):
            raise ValueError('Source file name contains invalid characters')
        return v


class GraphMLImportSchema(BaseModel):
    """Schema for GraphML import validation"""
    
    metadata: ImportMetadata = Field(...)
    graphml_content: str = Field(..., min_length=1)
    
    # GraphML specific settings
    node_id_attribute: str = Field(default="id")
    node_type_attribute: str = Field(default="type")
    edge_type_attribute: str = Field(default="relationship")
    coordinate_attributes: Optional[Dict[str, str]] = Field(
        default={"x": "x", "y": "y", "z": "z"}
    )
    
    # Namespace and schema settings
    graphml_namespace: Optional[str] = Field(None)
    validate_graphml_schema: bool = Field(default=True)
    
    @validator('graphml_content')
    def validate_graphml_format(cls, v):
        """Basic GraphML format validation"""
        if not v.strip().startswith('<?xml') and not v.strip().startswith('<graphml'):
            raise ValueError('Content does not appear to be valid GraphML')
        
        # Check for required GraphML elements
        required_elements = ['<graphml', '<graph', '</graphml>']
        for element in required_elements:
            if element not in v:
                raise ValueError(f'Missing required GraphML element: {element}')
        
        return v
    
    @root_validator
    def validate_format_consistency(cls, values):
        """Ensure metadata format matches schema"""
        metadata = values.get('metadata')
        if metadata and metadata.format != ImportFormat.GRAPHML:
            raise ValueError('Metadata format must be GRAPHML for GraphML import')
        return values


class JSONImportSchema(BaseModel):
    """Schema for JSON import validation"""
    
    metadata: ImportMetadata = Field(...)
    components: Optional[List[ComponentValidationSchema]] = Field(None)
    circuits: Optional[List[CircuitValidationSchema]] = Field(None)
    relationships: Optional[List[Dict[str, Any]]] = Field(None)
    zones: Optional[List[Dict[str, Any]]] = Field(None)
    vehicles: Optional[List[Dict[str, Any]]] = Field(None)
    
    @root_validator
    def validate_format_consistency(cls, values):
        """Ensure metadata format matches schema"""
        metadata = values.get('metadata')
        if metadata and metadata.format != ImportFormat.JSON:
            raise ValueError('Metadata format must be JSON for JSON import')
        return values
    
    @root_validator
    def validate_content_exists(cls, values):
        """Ensure at least one content type is provided"""
        content_fields = ['components', 'circuits', 'relationships', 'zones', 'vehicles']
        provided_content = [field for field in content_fields if values.get(field)]
        
        if not provided_content:
            raise ValueError('At least one content type must be provided')
        
        return values
    
    @validator('relationships')
    def validate_relationships(cls, v):
        """Validate relationship data structure"""
        if v is None:
            return v
        
        for relationship in v:
            # Required fields for relationships
            required_fields = ['from_id', 'to_id', 'type']
            missing_fields = [field for field in required_fields if field not in relationship]
            if missing_fields:
                raise ValueError(f'Relationship missing required fields: {", ".join(missing_fields)}')
            
            # Validate relationship type
            valid_types = ['CONNECTS_TO', 'POWERED_BY', 'CONTROLS', 'LOCATED_IN', 'PART_OF']
            if relationship['type'] not in valid_types:
                raise ValueError(f'Invalid relationship type: {relationship["type"]}')
        
        return v


class CSVImportSchema(BaseModel):
    """Schema for CSV import validation"""
    
    metadata: ImportMetadata = Field(...)
    csv_content: str = Field(..., min_length=1)
    
    # CSV specific settings
    delimiter: str = Field(default=",", max_length=1)
    quote_char: str = Field(default='"', max_length=1)
    escape_char: Optional[str] = Field(None, max_length=1)
    has_header: bool = Field(default=True)
    encoding: str = Field(default="utf-8")
    
    # Data mapping
    entity_type: Literal["components", "circuits", "relationships"] = Field(...)
    column_mapping: Dict[str, str] = Field(..., min_items=1)
    
    # Data transformation
    null_values: List[str] = Field(default=["", "NULL", "null", "None", "N/A"])
    date_format: Optional[str] = Field(None)
    coordinate_unit: Optional[str] = Field(None, description="Unit for coordinate values (mm, cm, m)")
    
    @validator('csv_content')
    def validate_csv_format(cls, v):
        """Basic CSV format validation"""
        lines = v.strip().split('\n')
        if len(lines) < 2:  # At least header and one data row
            raise ValueError('CSV must contain at least a header and one data row')
        
        return v
    
    @validator('column_mapping')
    def validate_column_mapping(cls, v, values):
        """Validate column mapping for entity type"""
        entity_type = values.get('entity_type')
        
        if entity_type == 'components':
            required_columns = ['id', 'vehicle_signature', 'type', 'name']
        elif entity_type == 'circuits':
            required_columns = ['id', 'vehicle_signature', 'name', 'circuit_type', 'voltage', 'max_current']
        elif entity_type == 'relationships':
            required_columns = ['from_id', 'to_id', 'type']
        else:
            raise ValueError(f'Unsupported entity type: {entity_type}')
        
        missing_mappings = [col for col in required_columns if col not in v.values()]
        if missing_mappings:
            raise ValueError(f'Missing column mappings for: {", ".join(missing_mappings)}')
        
        return v
    
    @root_validator
    def validate_format_consistency(cls, values):
        """Ensure metadata format matches schema"""
        metadata = values.get('metadata')
        if metadata and metadata.format != ImportFormat.CSV:
            raise ValueError('Metadata format must be CSV for CSV import')
        return values


class NDJSONImportSchema(BaseModel):
    """Schema for NDJSON (newline-delimited JSON) import validation"""
    
    metadata: ImportMetadata = Field(...)
    ndjson_content: str = Field(..., min_length=1)
    
    # NDJSON specific settings
    entity_type: Literal["components", "circuits", "relationships", "mixed"] = Field(...)
    strict_json: bool = Field(default=True)
    
    @validator('ndjson_content')
    def validate_ndjson_format(cls, v):
        """Validate NDJSON format"""
        import json
        
        lines = v.strip().split('\n')
        if not lines:
            raise ValueError('NDJSON content cannot be empty')
        
        # Validate each line is valid JSON
        for i, line in enumerate(lines):
            if line.strip():  # Skip empty lines
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid JSON on line {i+1}: {e}')
        
        return v
    
    @root_validator
    def validate_format_consistency(cls, values):
        """Ensure metadata format matches schema"""
        metadata = values.get('metadata')
        if metadata and metadata.format != ImportFormat.NDJSON:
            raise ValueError('Metadata format must be NDJSON for NDJSON import')
        return values


class ImportValidationResult(BaseModel):
    """Result of import validation"""
    
    is_valid: bool = Field(...)
    total_records: int = Field(..., ge=0)
    valid_records: int = Field(..., ge=0)
    invalid_records: int = Field(..., ge=0)
    
    # Validation details
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Processing statistics
    validation_time: float = Field(..., ge=0, description="Validation time in seconds")
    memory_usage: Optional[float] = Field(None, description="Peak memory usage in MB")
    
    # Entity counts
    component_count: int = Field(default=0, ge=0)
    circuit_count: int = Field(default=0, ge=0)
    relationship_count: int = Field(default=0, ge=0)
    zone_count: int = Field(default=0, ge=0)
    
    # Data quality metrics
    duplicate_ids: List[str] = Field(default_factory=list)
    missing_required_fields: List[Dict[str, Any]] = Field(default_factory=list)
    invalid_references: List[Dict[str, Any]] = Field(default_factory=list)
    data_inconsistencies: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate validation success rate"""
        if self.total_records == 0:
            return 0.0
        return (self.valid_records / self.total_records) * 100
    
    @property
    def can_proceed(self) -> bool:
        """Check if import can proceed based on validation results"""
        return self.is_valid and self.valid_records > 0


class ImportExecutionResult(BaseModel):
    """Result of import execution"""
    
    success: bool = Field(...)
    records_processed: int = Field(..., ge=0)
    records_created: int = Field(..., ge=0)
    records_updated: int = Field(..., ge=0)
    records_skipped: int = Field(..., ge=0)
    records_failed: int = Field(..., ge=0)
    
    # Execution details
    execution_time: float = Field(..., ge=0, description="Execution time in seconds")
    memory_peak: Optional[float] = Field(None, description="Peak memory usage in MB")
    
    # Error details
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Transaction details
    transaction_id: Optional[str] = Field(None, description="Database transaction ID")
    rollback_available: bool = Field(default=False)
    
    # Created entities
    created_components: List[str] = Field(default_factory=list)
    created_circuits: List[str] = Field(default_factory=list)
    created_relationships: List[str] = Field(default_factory=list)
    
    @property
    def processing_rate(self) -> float:
        """Calculate records processed per second"""
        if self.execution_time == 0:
            return 0.0
        return self.records_processed / self.execution_time


class BatchImportSchema(BaseModel):
    """Schema for batch import operations"""
    
    vehicle_signature: str = Field(..., min_length=1, max_length=100)
    imports: List[Union[GraphMLImportSchema, JSONImportSchema, CSVImportSchema, NDJSONImportSchema]] = Field(
        ..., min_items=1, max_items=50
    )
    
    # Batch settings
    parallel_processing: bool = Field(default=False)
    max_concurrent: int = Field(default=5, ge=1, le=20)
    stop_on_error: bool = Field(default=False)
    
    @validator('imports')
    def validate_vehicle_signature_consistency(cls, v, values):
        """Ensure all imports have the same vehicle signature"""
        batch_vehicle_sig = values.get('vehicle_signature')
        if not batch_vehicle_sig:
            return v
        
        for import_item in v:
            if import_item.metadata.vehicle_signature != batch_vehicle_sig:
                raise ValueError('All imports in batch must have the same vehicle_signature')
        
        return v