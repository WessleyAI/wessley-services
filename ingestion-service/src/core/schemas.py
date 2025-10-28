"""
Core Pydantic schemas for the ingestion service.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Source types for ingestion."""
    UPLOAD = "upload"
    URL = "url"


class IngestionStatus(str, Enum):
    """Status values for ingestion jobs."""
    QUEUED = "queued"
    PROCESSING = "processing"
    OCR_DONE = "ocr_done"
    SCHEMATIC_DONE = "schematic_done"
    PERSISTING = "persisting"
    COMPLETED = "completed"
    FAILED = "failed"


class OcrEngine(str, Enum):
    """Available OCR engines."""
    TESSERACT = "tesseract"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"


class ComponentType(str, Enum):
    """Component types for schematic parsing."""
    RESISTOR = "resistor"
    CAPACITOR = "capacitor"
    POLARIZED_CAP = "polarized_cap"
    INDUCTOR = "inductor"
    DIODE = "diode"
    ZENER = "zener"
    BJT_NPN = "bjt_npn"
    BJT_PNP = "bjt_pnp"
    MOSFET_N = "mosfet_n"
    MOSFET_P = "mosfet_p"
    OPAMP = "opamp"
    GROUND = "ground"
    POWER_FLAG = "power_flag"
    CONNECTOR = "connector"
    IC = "ic"
    FUSE = "fuse"
    RELAY = "relay"
    LAMP = "lamp"
    SWITCH = "switch"
    NET_LABEL = "net_label"
    JUNCTION = "junction"
    ARROW = "arrow"


# Core data models

class TextSpan(BaseModel):
    """Text span extracted from document."""
    page: int = Field(ge=1, description="Page number (1-indexed)")
    bbox: List[float] = Field(description="Bounding box [x1,y1,x2,y2] in pixels")
    text: str = Field(min_length=1, description="Extracted text content")
    rotation: int = Field(default=0, description="Text rotation in degrees")
    confidence: float = Field(ge=0.0, le=1.0, description="OCR confidence score")
    engine: OcrEngine = Field(description="OCR engine used")
    
    @classmethod
    def model_validate(cls, value):
        """Custom validation for bbox length."""
        if isinstance(value, dict) and 'bbox' in value:
            bbox = value['bbox']
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("bbox must be a list of exactly 4 floats")
        return super().model_validate(value)


class Pin(BaseModel):
    """Component pin definition."""
    name: str = Field(description="Pin name/number")
    bbox: List[float] = Field(description="Pin bounding box [x1,y1,x2,y2] in pixels")
    page: int = Field(ge=1, description="Page number")


class Component(BaseModel):
    """Electronic component detected in schematic."""
    id: str = Field(description="Component identifier (e.g., R1, C2)")
    type: ComponentType = Field(description="Component type")
    value: Optional[str] = Field(default=None, description="Component value (e.g., 10k, 100nF)")
    footprint: Optional[str] = Field(default=None, description="Component footprint")
    page: int = Field(ge=1, description="Page number")
    bbox: List[float] = Field(description="Component bounding box [x1,y1,x2,y2] in pixels")
    pins: List[Pin] = Field(default_factory=list, description="Component pins")
    attributes: Dict[str, str] = Field(default_factory=dict, description="Additional attributes")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    provenance: Dict[str, List[str]] = Field(default_factory=dict, description="Source text spans")


class NetConnection(BaseModel):
    """Connection between component pin and net."""
    component_id: str = Field(description="Component identifier")
    pin: str = Field(description="Pin name/number")


class Net(BaseModel):
    """Electrical net in schematic."""
    name: str = Field(description="Net name")
    connections: List[NetConnection] = Field(description="Connected pins")
    page_spans: List[int] = Field(description="Pages where net appears")
    confidence: float = Field(ge=0.0, le=1.0, description="Net detection confidence")


class UnresolvedConnection(BaseModel):
    """Unresolved connection in netlist."""
    reason: str = Field(description="Reason for unresolved connection")
    component_id: str = Field(description="Component identifier")
    pin: str = Field(description="Pin name/number")


class Netlist(BaseModel):
    """Complete netlist for schematic."""
    nets: List[Net] = Field(description="Electrical nets")
    unresolved: List[UnresolvedConnection] = Field(
        default_factory=list, 
        description="Unresolved connections"
    )


# API request/response models

class VehicleMeta(BaseModel):
    """Vehicle metadata."""
    make: str = Field(description="Vehicle manufacturer")
    model: str = Field(description="Vehicle model")
    year: int = Field(ge=1900, le=2100, description="Vehicle year")


class DocumentMeta(BaseModel):
    """Document metadata."""
    project_id: UUID = Field(description="Project identifier")
    vehicle: VehicleMeta = Field(description="Vehicle information")


class IngestionSource(BaseModel):
    """Source for ingestion job."""
    type: SourceType = Field(description="Source type")
    file_id: str = Field(description="File identifier (URI)")


class IngestionModes(BaseModel):
    """Processing modes for ingestion."""
    ocr: List[OcrEngine] = Field(default=[OcrEngine.TESSERACT], description="OCR engines to use")
    schematic_parse: bool = Field(default=False, description="Enable schematic parsing")


class CreateIngestionRequest(BaseModel):
    """Request to create ingestion job."""
    source: IngestionSource = Field(description="Input source")
    doc_meta: DocumentMeta = Field(description="Document metadata")
    modes: IngestionModes = Field(description="Processing modes")
    notify_channel: Optional[str] = Field(default=None, description="Realtime notification channel")


class CreateIngestionResponse(BaseModel):
    """Response from creating ingestion job."""
    job_id: UUID = Field(description="Job identifier")
    status: IngestionStatus = Field(description="Initial job status")


class ProcessingMetrics(BaseModel):
    """Metrics from processing job."""
    cer: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Character error rate")
    wer: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Word error rate")
    struct_accuracy: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Structural accuracy")


class IngestionArtifacts(BaseModel):
    """Artifacts produced by ingestion job."""
    text_spans: Optional[str] = Field(default=None, description="Text spans NDJSON file URL")
    components: Optional[str] = Field(default=None, description="Components JSON file URL")
    netlist: Optional[str] = Field(default=None, description="Netlist JSON file URL")
    graphml: Optional[str] = Field(default=None, description="GraphML file URL")
    debug_overlay: Optional[str] = Field(default=None, description="Debug overlay image URL")


class GetIngestionResponse(BaseModel):
    """Response from getting ingestion job status."""
    job_id: UUID = Field(description="Job identifier")
    status: IngestionStatus = Field(description="Job status")
    progress: Optional[int] = Field(default=None, ge=0, le=100, description="Progress percentage")
    current_stage: Optional[str] = Field(default=None, description="Current processing stage")
    artifacts: Optional[IngestionArtifacts] = Field(default=None, description="Generated artifacts")
    metrics: Optional[ProcessingMetrics] = Field(default=None, description="Processing metrics")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: Optional[datetime] = Field(default=None, description="Job creation time")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")


class PageImage(BaseModel):
    """Processed page image."""
    page: int = Field(ge=1, description="Page number")
    dpi: int = Field(ge=72, description="Image DPI")
    width: int = Field(ge=1, description="Image width in pixels")
    height: int = Field(ge=1, description="Image height in pixels")
    file_path: str = Field(description="Path to processed image file")


class BenchmarkRequest(BaseModel):
    """Request to run benchmark suite."""
    engine: Optional[str] = Field(default="all", description="OCR engine to test")
    report_format: str = Field(default="json", pattern="^(json|md)$", description="Report format")


class BenchmarkResult(BaseModel):
    """Benchmark test result."""
    timestamp: datetime = Field(description="Test execution time")
    engine: str = Field(description="OCR engine tested")
    metrics: ProcessingMetrics = Field(description="Performance metrics")
    dataset: str = Field(description="Test dataset name")


class RealtimeUpdate(BaseModel):
    """Realtime job status update."""
    job_id: UUID = Field(description="Job identifier")
    status: IngestionStatus = Field(description="Current status")
    progress: int = Field(ge=0, le=100, description="Progress percentage")
    stage: str = Field(description="Current processing stage")
    metrics: Optional[ProcessingMetrics] = Field(default=None, description="Current metrics")
    timestamp: datetime = Field(description="Update timestamp")


class HealthStatus(BaseModel):
    """Health check response."""
    status: str = Field(description="Service status")
    version: str = Field(description="Service version")
    timestamp: datetime = Field(description="Health check timestamp")
    dependencies: Dict[str, bool] = Field(description="Dependency health status")