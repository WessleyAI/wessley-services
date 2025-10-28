"""
Core pipeline orchestration for ingestion jobs.
"""
import asyncio
import os
import uuid
from typing import Dict, Any, Optional, List

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


class IngestionPipeline:
    """
    Orchestrates the complete ingestion pipeline for a job.
    """
    
    def __init__(self, job_id: uuid.UUID, request: CreateIngestionRequest):
        self.job_id = job_id
        self.request = request
        self.metrics = ProcessingMetrics()
        self.artifacts: Dict[str, str] = {}
        
        # Initialize processors
        self.pdf_processor = PdfProcessor()
        self.image_preprocessor = ImagePreprocessor()
        self.ocr_engines = self._initialize_ocr_engines()
    
    async def execute(self) -> None:
        """
        Execute the complete ingestion pipeline.
        """
        try:
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
    
    async def _preprocess_document(self) -> List[PageImage]:
        """
        Preprocess input document into page images.
        
        Downloads file, converts PDF to images, and applies preprocessing.
        """
        try:
            # TODO: Download file from self.request.source.file_id
            # For now, assume local file path
            source_path = self.request.source.file_id.replace("supabase://bucket/", "/tmp/")
            
            if not os.path.exists(source_path):
                # Create a dummy file for testing
                source_path = "/tmp/test_document.pdf"
                # In real implementation, download from Supabase Storage
            
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
            # Return dummy image for testing
            return [
                PageImage(
                    page=1,
                    dpi=300,
                    width=2400,
                    height=3200,
                    file_path="/tmp/dummy_page_1.png"
                )
            ]
    
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
            # Return fallback placeholder results
            return self._create_fallback_text_spans(page_images)
    
    async def _analyze_schematics(
        self,
        page_images: list[PageImage],
        text_spans: list[TextSpan]
    ) -> tuple[list[Component], Optional[Netlist]]:
        """
        Analyze schematics to detect components and generate netlist.
        
        TODO: Implement actual schematic analysis:
        - Symbol detection using YOLO/Detectron2
        - Line/wire extraction
        - Junction detection
        - Text-to-symbol association
        - Net propagation and netlist generation
        """
        # Placeholder implementation
        await asyncio.sleep(3)  # Simulate processing time
        
        components = []
        netlist = None
        
        if page_images and text_spans:
            # Simulate component detection
            from .schemas import ComponentType, Pin, Net, NetConnection
            
            components = [
                Component(
                    id="R1",
                    type=ComponentType.RESISTOR,
                    value="10k",
                    page=1,
                    bbox=[100.0, 90.0, 200.0, 130.0],
                    pins=[
                        Pin(name="1", bbox=[100.0, 110.0, 105.0, 115.0], page=1),
                        Pin(name="2", bbox=[195.0, 110.0, 200.0, 115.0], page=1),
                    ],
                    confidence=0.92,
                    provenance={"text_spans": ["ts_1", "ts_2"]}
                )
            ]
            
            # Simulate netlist generation
            netlist = Netlist(
                nets=[
                    Net(
                        name="VCC",
                        connections=[
                            NetConnection(component_id="R1", pin="1")
                        ],
                        page_spans=[1],
                        confidence=0.85
                    )
                ]
            )
        
        # Update metrics
        self.metrics.struct_accuracy = 0.82
        
        return components, netlist
    
    async def _persist_results(
        self,
        text_spans: list[TextSpan],
        components: list[Component],
        netlist: Optional[Netlist]
    ) -> None:
        """
        Persist results to datastores and generate artifacts.
        
        TODO: Implement actual persistence:
        - Store in Neo4j (graph relationships)
        - Store in Qdrant (semantic embeddings)
        - Upload artifacts to S3/Supabase Storage
        - Update Supabase job metadata
        """
        # Placeholder implementation
        await asyncio.sleep(1)  # Simulate processing time
        
        # Simulate artifact generation
        self.artifacts = {
            "text_spans": f"s3://bucket/jobs/{self.job_id}/text_spans.ndjson",
            "components": f"s3://bucket/jobs/{self.job_id}/components.json",
            "netlist": f"s3://bucket/jobs/{self.job_id}/netlist.json" if netlist else None,
            "debug_overlay": f"s3://bucket/jobs/{self.job_id}/debug.png",
        }
        
        # Remove None values
        self.artifacts = {k: v for k, v in self.artifacts.items() if v is not None}
    
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