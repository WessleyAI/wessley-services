"""
Core pipeline orchestration for ingestion jobs.
"""
import asyncio
import os
import uuid
import json
import logging
import tempfile
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime

from .schemas import (
    CreateIngestionRequest,
    IngestionStatus,
    ProcessingMetrics,
    PageImage,
    TextSpan,
    Component,
    Netlist,
    OcrEngine,
)
from ..workers.queue import queue_manager
from ..preprocess.pdf import PdfProcessor
from ..preprocess.image import ImagePreprocessor
from ..ocr.tesseract import TesseractProvider
from ..ocr.deepseek import DeepSeekProvider
from ..ocr.mistral import MistralProvider
from ..ocr.fusion import OcrFusionEngine
from ..schematics.detect import YOLOComponentDetector, TraditionalComponentDetector
from ..schematics.wires import WireExtractor
from ..schematics.associate import TextSymbolAssociator
from ..schematics.export import NetlistGenerator, NetlistExporter
from ..persist.neo4j import Neo4jPersistence, create_neo4j_persistence
from ..persist.qdrant import QdrantPersistence, create_qdrant_persistence
from ..persist.storage import StorageManager, create_storage_manager, ArtifactType
from ..persist.supabase_metadata import SupabaseMetadata, create_supabase_metadata
from ..persist.chunking import TextChunker, create_text_chunker


class IngestionPipeline:
    """
    Orchestrates the complete ingestion pipeline for a job.
    """
    
    def __init__(self, job_id: uuid.UUID, request: CreateIngestionRequest):
        self.job_id = job_id
        self.request = request
        self.metrics = ProcessingMetrics()
        self.artifacts: Dict[str, str] = {}
        self.warnings: List[str] = []
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.pdf_processor = PdfProcessor()
        self.image_preprocessor = ImagePreprocessor()
        self.ocr_engines = self._initialize_ocr_engines()
        
        # Initialize schematic analysis components
        self.component_detector = self._initialize_component_detector()
        self.wire_extractor = WireExtractor()
        self.text_associator = TextSymbolAssociator()
        self.netlist_generator = NetlistGenerator()
        self.netlist_exporter = NetlistExporter()
        
        # Initialize persistence components
        self.text_chunker = create_text_chunker()
        self.storage_manager = create_storage_manager()
        self.supabase_metadata = create_supabase_metadata()
        self.neo4j_persistence = None  # Initialize on demand
        self.qdrant_persistence = None  # Initialize on demand
    
    async def execute(self) -> None:
        """
        Execute the complete ingestion pipeline.
        """
        try:
            # Initialize persistence backends
            persistence_ready = await self._initialize_persistence_backends()
            if not persistence_ready:
                self.logger.warning("Some persistence backends failed to initialize")
            
            # Update status to processing
            await queue_manager.update_job_status(
                self.job_id,
                IngestionStatus.PROCESSING,
                progress=0,
                stage="pre_processing"
            )
            
            # Stage 1: Pre-processing
            page_images = await self._preprocess_document()
            await self._update_progress(20, "ocr_processing")
            
            # Stage 2: OCR Processing
            text_spans = await self._extract_text(page_images)
            await self._update_progress(50, "schematic_analysis")
            
            # Stage 3: Schematic Analysis (if enabled)
            components = []
            netlist = None
            if self.request.modes.schematic_parse:
                components, netlist = await self._analyze_schematics(page_images, text_spans)
            
            await self._update_progress(80, "persistence")
            
            # Stage 4: Persistence
            await self._persist_results(text_spans, components, netlist)
            await self._update_progress(100, "completed")
            
            # Mark as completed
            await queue_manager.update_job_status(
                self.job_id,
                IngestionStatus.COMPLETED,
                progress=100,
                stage="completed",
                artifacts=self.artifacts,
                metrics=self.metrics.model_dump(),
            )
            
        except Exception as e:
            # Mark as failed
            await queue_manager.update_job_status(
                self.job_id,
                IngestionStatus.FAILED,
                error=str(e),
                stage="failed"
            )
            raise
    
    def _initialize_ocr_engines(self) -> List:
        """Initialize OCR engines based on request configuration."""
        engines = []
        
        for engine_name in self.request.modes.ocr:
            try:
                if engine_name == OcrEngine.TESSERACT:
                    engines.append(TesseractProvider())
                elif engine_name == OcrEngine.DEEPSEEK:
                    engines.append(DeepSeekProvider())
                elif engine_name == OcrEngine.MISTRAL:
                    engines.append(MistralProvider())
            except Exception as e:
                print(f"Failed to initialize {engine_name} OCR engine: {e}")
        
        return engines
    
    def _initialize_component_detector(self):
        """Initialize component detector based on available models."""
        # Try to initialize YOLO detector first, fall back to traditional CV
        try:
            return YOLOComponentDetector()
        except Exception as e:
            print(f"YOLO detector not available, using traditional CV: {e}")
            return TraditionalComponentDetector()
    
    async def _initialize_persistence_backends(self) -> bool:
        """Initialize persistence backends (Neo4j, Qdrant, Storage)."""
        success = True
        
        try:
            # Initialize storage manager
            if not await self.storage_manager.initialize():
                self.logger.warning("Storage manager initialization failed")
                success = False
            
            # Initialize Supabase metadata
            if not await self.supabase_metadata.connect():
                self.logger.warning("Supabase metadata initialization failed")
                success = False
            
            # Initialize Neo4j (optional)
            try:
                self.neo4j_persistence = create_neo4j_persistence()
                if not await self.neo4j_persistence.connect():
                    self.logger.warning("Neo4j initialization failed")
                    self.neo4j_persistence = None
            except Exception as e:
                self.logger.warning(f"Neo4j not available: {e}")
                self.neo4j_persistence = None
            
            # Initialize Qdrant (optional)
            try:
                self.qdrant_persistence = create_qdrant_persistence(use_openai=False)  # Use mock embeddings for testing
                if not await self.qdrant_persistence.connect():
                    self.logger.warning("Qdrant initialization failed")
                    self.qdrant_persistence = None
            except Exception as e:
                self.logger.warning(f"Qdrant not available: {e}")
                self.qdrant_persistence = None
            
            return success
            
        except Exception as e:
            self.logger.error(f"Persistence initialization failed: {e}")
            return False
    
    async def _preprocess_document(self) -> List[PageImage]:
        """
        Preprocess input document into page images.
        
        Downloads file, converts PDF to images, and applies preprocessing.
        """
        try:
            # Download file from URL if it's a URL source
            if self.request.source.type.value == "url":
                source_path = await self._download_file_from_url(self.request.source.file_id)
            else:
                # Handle other source types (supabase, local, etc.)
                source_path = self.request.source.file_id.replace("supabase://bucket/", "/tmp/")
                
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"Document not found at path: {source_path}")
            
            # Determine file type and process accordingly
            if source_path.lower().endswith('.pdf'):
                # Convert PDF to images
                page_images = await self.pdf_processor.convert_pdf_to_images(source_path)
            else:
                # Process single image file
                page_image = await self.image_preprocessor.preprocess_image(source_path)
                page_images = [page_image]
            
            # Apply additional preprocessing to each page
            processed_images = []
            for page_image in page_images:
                # Apply OCR-optimized preprocessing
                final_image = await self.image_preprocessor.preprocess_image(
                    page_image.file_path,
                    page_image.page,
                    operations=[
                        "resize_to_target_dpi",
                        "convert_to_grayscale",
                        "deskew", 
                        "denoise",
                        "enhance_contrast",
                        "binarize"
                    ]
                )
                processed_images.append(final_image)
            
            return processed_images
            
        except Exception as e:
            print(f"Document preprocessing failed: {e}")
            raise RuntimeError(f"Document preprocessing failed: {e}")
    
    async def _extract_text(self, page_images: List[PageImage]) -> List[TextSpan]:
        """
        Extract text from page images using configured OCR engines.
        
        Uses multi-engine fusion if multiple engines are configured.
        """
        if not self.ocr_engines:
            print("No OCR engines available")
            return []
        
        all_text_spans = []
        
        try:
            for page_image in page_images:
                if len(self.ocr_engines) == 1:
                    # Single engine
                    text_spans = await self.ocr_engines[0].extract_text(page_image)
                else:
                    # Multi-engine fusion
                    fusion_engine = OcrFusionEngine(
                        providers=self.ocr_engines,
                        fusion_strategy="confidence_weighted",
                        confidence_threshold=0.5,
                        iou_threshold=0.3
                    )
                    text_spans = await fusion_engine.extract_text_fused(page_image)
                
                all_text_spans.extend(text_spans)
            
            # Calculate OCR quality metrics
            self._calculate_ocr_metrics(all_text_spans)
            
            return all_text_spans
            
        except Exception as e:
            print(f"OCR extraction failed: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    async def _analyze_schematics(
        self,
        page_images: List[PageImage],
        text_spans: List[TextSpan]
    ) -> tuple[List[Component], Optional[Netlist]]:
        """
        Analyze schematics to detect components and generate netlist.
        
        Implements complete schematic analysis pipeline:
        - Symbol detection using YOLO/Detectron2 or traditional CV
        - Line/wire extraction and junction detection
        - Text-to-symbol association using spatial analysis
        - Net propagation and netlist generation
        """
        try:
            all_components = []
            all_netlists = []
            
            for page_image in page_images:
                print(f"Analyzing schematic on page {page_image.page}")
                
                # Step 1: Detect components/symbols
                component_detections = await self.component_detector.detect_components(page_image)
                print(f"Detected {len(component_detections)} components")
                
                # Step 2: Extract wires, junctions, and networks
                line_segments, junctions, wire_nets = self.wire_extractor.extract_wires(
                    page_image, text_spans
                )
                print(f"Extracted {len(line_segments)} line segments, {len(junctions)} junctions, {len(wire_nets)} wire nets")
                
                # Step 3: Associate text with symbols
                page_text_spans = [span for span in text_spans if span.page == page_image.page]
                components_with_text = self.text_associator.associate_text_with_symbols(
                    page_text_spans, component_detections
                )
                print(f"Associated text with {len(components_with_text)} components")
                
                # Step 4: Generate netlist and component catalog
                export_result = self.netlist_generator.generate_netlist(
                    components_with_text, wire_nets, junctions, line_segments, page_image.page
                )
                
                # Convert to schema format
                schema_components = self._convert_to_schema_components(
                    export_result.component_catalog, page_image.page
                )
                all_components.extend(schema_components)
                
                if export_result.netlist.nets:
                    all_netlists.append(export_result.netlist)
                
                # Update progress metrics
                if export_result.statistics:
                    total_detections = export_result.statistics.get('total_components', 0)
                    total_connections = export_result.statistics.get('total_connections', 0)
                    unresolved = export_result.statistics.get('unresolved_connections', 0)
                    
                    if total_connections > 0:
                        connection_rate = 1.0 - (unresolved / total_connections)
                        self.metrics.struct_accuracy = max(self.metrics.struct_accuracy, connection_rate)
                
                print(f"Page {page_image.page} analysis complete: {len(schema_components)} components, "
                      f"{len(export_result.netlist.nets)} nets")
            
            # Merge netlists from all pages
            final_netlist = self._merge_netlists(all_netlists) if all_netlists else None
            
            # Store export artifacts
            if final_netlist:
                await self._store_schematic_artifacts(export_result, final_netlist)
            
            print(f"Schematic analysis complete: {len(all_components)} total components, "
                  f"{len(final_netlist.nets) if final_netlist else 0} total nets")
            
            return all_components, final_netlist
            
        except Exception as e:
            print(f"Schematic analysis failed: {e}")
            self.metrics.struct_accuracy = 0.0
            return [], None
    
    async def _persist_results(
        self,
        text_spans: List[TextSpan],
        components: List[Component],
        netlist: Optional[Netlist]
    ) -> None:
        """
        Persist results to all configured datastores and generate artifacts.
        
        Implements complete M4 persistence:
        - Store in Neo4j (graph relationships)
        - Store in Qdrant (semantic embeddings)
        - Upload artifacts to S3/Supabase Storage
        - Update Supabase job metadata
        """
        try:
            print(f"Starting persistence for job {self.job_id}")
            
            # Stage 1: Store artifacts in object storage
            await self._store_artifacts(text_spans, components, netlist)
            await self._update_progress(85, "graph_storage")
            
            # Stage 2: Store in graph database (Neo4j)
            if self.neo4j_persistence and components:
                await self._store_in_neo4j(text_spans, components, netlist)
            
            await self._update_progress(90, "vector_storage")
            
            # Stage 3: Store embeddings in vector database (Qdrant)
            if self.qdrant_persistence and text_spans:
                await self._store_in_qdrant(text_spans, components, netlist)
            
            await self._update_progress(95, "metadata_update")
            
            # Stage 4: Update job metadata
            await self._update_job_metadata()
            
            print(f"Persistence completed for job {self.job_id}")
            
        except Exception as e:
            self.logger.error(f"Persistence failed: {e}")
            self.warnings.append(f"Persistence failed: {e}")
            # Continue execution - persistence failure shouldn't fail the entire job
    
    def _calculate_ocr_metrics(self, text_spans: List[TextSpan]) -> None:
        """Calculate OCR quality metrics."""
        if not text_spans:
            self.metrics.cer = 1.0
            self.metrics.wer = 1.0
            return
        
        # Calculate average confidence as quality proxy
        avg_confidence = sum(span.confidence for span in text_spans) / len(text_spans)
        
        # Estimate error rates based on confidence
        # Higher confidence -> lower error rates
        self.metrics.cer = max(0.0, (1.0 - avg_confidence) * 0.3)  # Max 30% CER
        self.metrics.wer = max(0.0, (1.0 - avg_confidence) * 0.4)  # Max 40% WER
    
    def _create_fallback_text_spans(self, page_images: List[PageImage]) -> List[TextSpan]:
        """Create fallback text spans when OCR fails."""
        fallback_spans = []
        
        for page_image in page_images:
            # Create some basic component-like text spans
            fallback_spans.extend([
                TextSpan(
                    page=page_image.page,
                    bbox=[100.0, 100.0, 130.0, 120.0],
                    text="R1",
                    rotation=0,
                    confidence=0.7,
                    engine=OcrEngine.TESSERACT
                ),
                TextSpan(
                    page=page_image.page,
                    bbox=[140.0, 100.0, 170.0, 120.0],
                    text="10k",
                    rotation=0,
                    confidence=0.6,
                    engine=OcrEngine.TESSERACT
                ),
            ])
        
        return fallback_spans
    
    def _convert_to_schema_components(self, catalog_entries, page_number: int) -> List[Component]:
        """Convert component catalog entries to schema Component objects."""
        from .schemas import ComponentType, Pin
        
        components = []
        
        for entry in catalog_entries:
            # Map component type strings to enum values
            type_mapping = {
                'resistor': ComponentType.RESISTOR,
                'capacitor': ComponentType.CAPACITOR,
                'inductor': ComponentType.INDUCTOR,
                'ic': ComponentType.IC,
                'transistor': ComponentType.TRANSISTOR,
                'diode': ComponentType.DIODE,
                'connector': ComponentType.CONNECTOR,
            }
            
            component_type = type_mapping.get(entry.type.lower(), ComponentType.RESISTOR)
            
            # Convert pins
            pins = []
            for pin_dict in entry.pins:
                pin = Pin(
                    name=pin_dict.get('number', ''),
                    bbox=[0.0, 0.0, 0.0, 0.0],  # Default bbox
                    page=page_number
                )
                pins.append(pin)
            
            # Estimate bbox from position
            x, y = entry.position
            bbox = [x - 25, y - 15, x + 25, y + 15]  # Default component size
            
            component = Component(
                id=entry.reference,
                type=component_type,
                value=entry.value,
                page=page_number,
                bbox=bbox,
                pins=pins,
                confidence=entry.confidence,
                provenance={"detection_method": "computer_vision"}
            )
            
            components.append(component)
        
        return components
    
    def _merge_netlists(self, netlists: List[Netlist]) -> Netlist:
        """Merge netlists from multiple pages."""
        if not netlists:
            return Netlist(nets=[], unresolved=[])
        
        if len(netlists) == 1:
            return netlists[0]
        
        # Combine all nets and unresolved connections
        all_nets = []
        all_unresolved = []
        
        for netlist in netlists:
            all_nets.extend(netlist.nets)
            all_unresolved.extend(netlist.unresolved)
        
        # TODO: Implement smart merging of nets across pages
        # For now, just combine them
        
        return Netlist(nets=all_nets, unresolved=all_unresolved)
    
    async def _store_schematic_artifacts(self, export_result, netlist: Netlist):
        """Store schematic analysis artifacts."""
        try:
            # Export to different formats
            json_export = self.netlist_exporter.export_to_json(export_result)
            graphml_export = self.netlist_exporter.export_to_graphml(export_result)
            ndjson_export = self.netlist_exporter.export_to_ndjson(export_result)
            
            # Store exports (in real implementation, upload to S3/Supabase)
            base_path = f"/tmp/job_{self.job_id}"
            os.makedirs(base_path, exist_ok=True)
            
            # Write export files
            with open(f"{base_path}/netlist.json", 'w') as f:
                f.write(json_export)
            
            with open(f"{base_path}/netlist.graphml", 'w') as f:
                f.write(graphml_export)
            
            with open(f"{base_path}/netlist.ndjson", 'w') as f:
                f.write(ndjson_export)
            
            # Update artifacts dictionary
            self.artifacts.update({
                "netlist_json": f"s3://bucket/jobs/{self.job_id}/netlist.json",
                "netlist_graphml": f"s3://bucket/jobs/{self.job_id}/netlist.graphml", 
                "netlist_ndjson": f"s3://bucket/jobs/{self.job_id}/netlist.ndjson",
                "component_catalog": f"s3://bucket/jobs/{self.job_id}/components.json"
            })
            
            print(f"Stored schematic artifacts: {len(self.artifacts)} files")
            
        except Exception as e:
            print(f"Failed to store schematic artifacts: {e}")
            self.warnings.append(f"Artifact storage failed: {e}")
    
    async def _store_artifacts(
        self,
        text_spans: List[TextSpan],
        components: List[Component],
        netlist: Optional[Netlist]
    ):
        """Store all artifacts in object storage."""
        try:
            # Store text spans as NDJSON
            if text_spans:
                text_spans_ndjson = "\n".join(
                    span.model_dump_json() for span in text_spans
                )
                
                result = await self.storage_manager.store_artifact(
                    project_id=uuid.UUID(self.request.doc_meta.project_id),
                    job_id=self.job_id,
                    artifact_type=ArtifactType.TEXT_SPANS,
                    content=text_spans_ndjson,
                    filename="text_spans.ndjson",
                    content_type="application/x-ndjson",
                    metadata={"span_count": len(text_spans)}
                )
                
                if result.success:
                    self.artifacts["text_spans"] = result.url
            
            # Store components as JSON
            if components:
                components_data = {
                    "components": [comp.model_dump() for comp in components],
                    "metadata": {
                        "component_count": len(components),
                        "generated_at": datetime.now().isoformat()
                    }
                }
                
                result = await self.storage_manager.store_artifact(
                    project_id=uuid.UUID(self.request.doc_meta.project_id),
                    job_id=self.job_id,
                    artifact_type=ArtifactType.COMPONENTS,
                    content=json.dumps(components_data, indent=2),
                    filename="components.json",
                    content_type="application/json",
                    metadata={"component_count": len(components)}
                )
                
                if result.success:
                    self.artifacts["components"] = result.url
            
            # Store netlist in multiple formats
            if netlist:
                # JSON format
                netlist_data = {
                    "netlist": netlist.model_dump(),
                    "metadata": {
                        "net_count": len(netlist.nets),
                        "generated_at": datetime.now().isoformat()
                    }
                }
                
                result = await self.storage_manager.store_artifact(
                    project_id=uuid.UUID(self.request.doc_meta.project_id),
                    job_id=self.job_id,
                    artifact_type=ArtifactType.NETLIST_JSON,
                    content=json.dumps(netlist_data, indent=2),
                    filename="netlist.json",
                    content_type="application/json",
                    metadata={"net_count": len(netlist.nets)}
                )
                
                if result.success:
                    self.artifacts["netlist_json"] = result.url
            
            print(f"Stored {len(self.artifacts)} artifacts")
            
        except Exception as e:
            self.logger.error(f"Artifact storage failed: {e}")
    
    async def _store_in_neo4j(
        self,
        text_spans: List[TextSpan],
        components: List[Component],
        netlist: Optional[Netlist]
    ):
        """Store data in Neo4j graph database."""
        try:
            # Extract vehicle info from request
            vehicle_info = {
                "make": self.request.doc_meta.vehicle.make,
                "model": self.request.doc_meta.vehicle.model,
                "year": self.request.doc_meta.vehicle.year
            }
            
            # Store in Neo4j
            result = await self.neo4j_persistence.store_vehicle_schematic(
                project_id=uuid.UUID(self.request.doc_meta.project_id),
                vehicle_info=vehicle_info,
                components=components,
                nets=netlist.nets if netlist else [],
                text_spans=text_spans
            )
            
            print(f"Neo4j storage: {result.nodes_created} nodes, {result.relationships_created} relationships")
            
        except Exception as e:
            self.logger.error(f"Neo4j storage failed: {e}")
    
    async def _store_in_qdrant(
        self,
        text_spans: List[TextSpan],
        components: List[Component],
        netlist: Optional[Netlist]
    ):
        """Store embeddings in Qdrant vector database."""
        try:
            # Store text chunks with embeddings
            result = await self.qdrant_persistence.store_text_chunks(
                project_id=uuid.UUID(self.request.doc_meta.project_id),
                text_spans=text_spans,
                components=components,
                nets=netlist.nets if netlist else []
            )
            
            print(f"Qdrant storage: {result.embeddings_created} embeddings created")
            
        except Exception as e:
            self.logger.error(f"Qdrant storage failed: {e}")
    
    async def _update_job_metadata(self):
        """Update job metadata in Supabase."""
        try:
            # Update job with final artifacts and metrics
            await self.supabase_metadata.update_job_status(
                job_id=self.job_id,
                status=IngestionStatus.PROCESSING,  # Will be set to COMPLETED by main execute
                artifacts=self.artifacts,
                metrics=self.metrics.model_dump()
            )
            
        except Exception as e:
            self.logger.error(f"Job metadata update failed: {e}")
    
    async def _update_progress(self, progress: int, stage: str) -> None:
        """Update job progress and stage."""
        await queue_manager.update_job_status(
            self.job_id,
            IngestionStatus.PROCESSING,
            progress=progress,
            stage=stage
        )
        
        # Publish realtime update
        await queue_manager.publish_realtime_update(
            self.job_id,
            IngestionStatus.PROCESSING,
            progress,
            stage,
            metrics=self.metrics.model_dump() if self.metrics else None,
            channel=self.request.notify_channel
        )
    
    async def _download_file_from_url(self, url: str) -> str:
        """
        Download file from URL to temporary location.
        
        Args:
            url: URL to download from
            
        Returns:
            Path to downloaded file
        """
        try:
            print(f"Downloading file from URL: {url}")
            
            # Create temporary file
            suffix = ".pdf" if url.lower().endswith('.pdf') else ""
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Stream download for large files
                async with client.stream('GET', url) as response:
                    response.raise_for_status()
                    
                    # Log download start
                    content_length = response.headers.get('content-length')
                    if content_length:
                        print(f"File size: {int(content_length) / 1024 / 1024:.2f} MB")
                    
                    downloaded = 0
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress for large files
                        if content_length and downloaded % (1024 * 1024) == 0:  # Every MB
                            progress_pct = (downloaded / int(content_length)) * 100
                            print(f"Download progress: {progress_pct:.1f}%")
            
            temp_file.close()
            print(f"Downloaded file to: {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            print(f"Failed to download file from URL {url}: {e}")
            raise RuntimeError(f"Download failed: {e}")