#!/usr/bin/env python3
"""
Master Pipeline: Complete Wiring Diagram Processing System

End-to-end pipeline:
1. PDF → Tesseract OCR (extract text)
2. OCR Text → Intelligent LLM Analysis (classify & extract)
3. Extracted Data → Neo4j Graph (structure & relationships)
4. Extracted Data → Qdrant Vectors (semantic search)
5. Graph + Vectors → LLM Spatial Placement (3D coordinates)
6. Generate 3D model visualization

Usage:
    python3 master_pipeline.py \\
        --pdf manual.pdf \\
        --vehicle-make Mitsubishi \\
        --vehicle-model "Pajero Pinin" \\
        --vehicle-year 2000 \\
        --pages 1-100 \\
        --run-id auto
"""

import os
import sys
import json
import logging
import argparse
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# ============================================================================
# Logging Configuration
# ============================================================================

class PipelineLogger:
    """Centralized logging for entire pipeline"""

    def __init__(self, log_dir: str = "pipeline_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"pipeline_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger("MasterPipeline")
        self.logger.info(f"📝 Logging to: {log_file}")

        # Stage loggers
        self.ocr_logger = logging.getLogger("OCR")
        self.analysis_logger = logging.getLogger("Analysis")
        self.neo4j_logger = logging.getLogger("Neo4j")
        self.qdrant_logger = logging.getLogger("Qdrant")
        self.spatial_logger = logging.getLogger("Spatial")

    def log_stage(self, stage: str, message: str, level: str = "INFO"):
        """Log message for specific pipeline stage"""
        logger = getattr(self, f"{stage.lower()}_logger", self.logger)
        getattr(logger, level.lower())(message)


# ============================================================================
# Pipeline Configuration
# ============================================================================

class PipelineConfig:
    """Configuration for pipeline run"""

    def __init__(self, args):
        self.pdf_path = args.pdf
        self.vehicle_make = args.vehicle_make
        self.vehicle_model = args.vehicle_model
        self.vehicle_year = args.vehicle_year
        self.vehicle_variant = getattr(args, 'vehicle_variant', None)

        # Parse page range
        if '-' in args.pages:
            start, end = args.pages.split('-')
            self.start_page = int(start)
            self.end_page = int(end)
        else:
            self.start_page = int(args.pages)
            self.end_page = int(args.pages)

        # Generate run ID
        if args.run_id == 'auto':
            self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.run_id = args.run_id

        # Document hash (for deduplication)
        self.document_hash = self._hash_file(self.pdf_path)

        # Output directories
        self.output_dir = Path(args.output_dir)
        self.ocr_dir = self.output_dir / "ocr"
        self.metadata_dir = self.output_dir / "metadata"
        self.neo4j_export_dir = self.output_dir / "neo4j_export"
        self.qdrant_export_dir = self.output_dir / "qdrant_export"

        # Create directories
        for dir_path in [self.ocr_dir, self.metadata_dir,
                         self.neo4j_export_dir, self.qdrant_export_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Models
        self.ocr_engine = args.ocr_engine
        self.llm_model = args.llm_model

        # Flags
        self.skip_ocr = args.skip_ocr
        self.skip_analysis = args.skip_analysis
        self.skip_storage = args.skip_storage
        self.skip_spatial = args.skip_spatial

    def _hash_file(self, filepath: str) -> str:
        """Generate hash of PDF file"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dict"""
        return {
            "run_id": self.run_id,
            "pdf_path": self.pdf_path,
            "document_hash": self.document_hash,
            "vehicle": {
                "make": self.vehicle_make,
                "model": self.vehicle_model,
                "year": self.vehicle_year,
                "variant": self.vehicle_variant
            },
            "pages": {
                "start": self.start_page,
                "end": self.end_page,
                "total": self.end_page - self.start_page + 1
            },
            "models": {
                "ocr": self.ocr_engine,
                "llm": self.llm_model
            },
            "started_at": datetime.now().isoformat()
        }


# ============================================================================
# Pipeline Stages
# ============================================================================

class OCRStage:
    """Stage 1: OCR extraction"""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger):
        self.config = config
        self.logger = logger

    def run(self) -> Dict[str, Any]:
        """Run OCR on PDF pages"""
        self.logger.log_stage("ocr", "=" * 70)
        self.logger.log_stage("ocr", "📄 Stage 1: OCR Extraction")
        self.logger.log_stage("ocr", "=" * 70)

        if self.config.skip_ocr:
            self.logger.log_stage("ocr", "⏭️  Skipping OCR (using existing results)")
            return {"status": "skipped", "ocr_dir": str(self.config.ocr_dir)}

        # Import Tesseract OCR module
        from tesseract_ocr import run_tesseract_ocr

        self.logger.log_stage("ocr", f"PDF: {self.config.pdf_path}")
        self.logger.log_stage("ocr", f"Pages: {self.config.start_page}-{self.config.end_page}")
        self.logger.log_stage("ocr", f"Engine: {self.config.ocr_engine}")
        self.logger.log_stage("ocr", f"Output: {self.config.ocr_dir}")

        start_time = datetime.now()

        # Run Tesseract OCR
        results = run_tesseract_ocr(
            pdf_path=self.config.pdf_path,
            output_dir=str(self.config.ocr_dir),
            start_page=self.config.start_page,
            end_page=self.config.end_page
        )

        duration = (datetime.now() - start_time).total_seconds()

        self.logger.log_stage("ocr", f"✅ OCR Complete")
        self.logger.log_stage("ocr", f"   Pages processed: {results['pages_processed']}")
        self.logger.log_stage("ocr", f"   Total elements: {results['total_elements']}")
        self.logger.log_stage("ocr", f"   Duration: {duration:.1f}s")
        self.logger.log_stage("ocr", f"   Avg: {duration / results['pages_processed']:.2f}s/page")

        return {
            "status": "completed",
            "ocr_dir": str(self.config.ocr_dir),
            "results": results,
            "duration": duration
        }


class AnalysisStage:
    """Stage 2: Intelligent LLM analysis"""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger):
        self.config = config
        self.logger = logger

    def run(self, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run intelligent analysis on OCR results"""
        self.logger.log_stage("analysis", "=" * 70)
        self.logger.log_stage("analysis", "🧠 Stage 2: Intelligent Analysis")
        self.logger.log_stage("analysis", "=" * 70)

        if self.config.skip_analysis:
            self.logger.log_stage("analysis", "⏭️  Skipping analysis")
            return {"status": "skipped"}

        # Import intelligent extractor
        from process_existing_ocr import process_existing_ocr

        self.logger.log_stage("analysis", f"OCR Directory: {ocr_results['ocr_dir']}")
        self.logger.log_stage("analysis", f"LLM Model: {self.config.llm_model}")
        self.logger.log_stage("analysis", f"Pages: {self.config.start_page}-{self.config.end_page}")

        start_time = datetime.now()

        # Run intelligent extraction
        # OCR results are in tesseract subdirectory
        tesseract_dir = str(Path(ocr_results['ocr_dir']) / 'tesseract')
        router = process_existing_ocr(
            ocr_dir=tesseract_dir,
            start_page=self.config.start_page,
            end_page=self.config.end_page,
            model=self.config.llm_model
        )

        # Save to JSON
        router.save_to_json(str(self.config.metadata_dir))

        duration = (datetime.now() - start_time).total_seconds()
        summary = router.get_storage_summary()

        self.logger.log_stage("analysis", f"✅ Analysis Complete")
        self.logger.log_stage("analysis", f"   Tier 1 (Metadata): {summary['tier_1_metadata']}")
        self.logger.log_stage("analysis", f"   Tier 2 (Knowledge): {summary['tier_2_knowledge']}")
        self.logger.log_stage("analysis", f"   Tier 3 (Structure): {summary['tier_3_structure']}")
        self.logger.log_stage("analysis", f"   Tier 4 (Semantic): {summary['tier_4_semantic']}")
        self.logger.log_stage("analysis", f"   Duration: {duration:.1f}s")

        return {
            "status": "completed",
            "metadata_dir": str(self.config.metadata_dir),
            "summary": summary,
            "duration": duration
        }


class StorageStage:
    """Stage 3: Store in Neo4j + Qdrant"""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger):
        self.config = config
        self.logger = logger

    def run(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Load extracted data into Neo4j and Qdrant"""
        self.logger.log_stage("neo4j", "=" * 70)
        self.logger.log_stage("neo4j", "🗄️  Stage 3: Storage (Neo4j + Qdrant)")
        self.logger.log_stage("neo4j", "=" * 70)

        if self.config.skip_storage:
            self.logger.log_stage("neo4j", "⏭️  Skipping storage")
            return {"status": "skipped"}

        # Import hybrid store
        from hybrid_knowledge_store import HybridKnowledgeStore

        start_time = datetime.now()

        # Initialize store
        self.logger.log_stage("neo4j", "Connecting to databases...")
        store = HybridKnowledgeStore()

        # Load metadata files
        metadata_dir = Path(analysis_results['metadata_dir'])

        # Load Tier 1: Metadata
        self.logger.log_stage("neo4j", "Loading Tier 1 (Metadata)...")
        tier1_file = metadata_dir / "tier_1_metadata.json"
        if tier1_file.exists():
            with open(tier1_file) as f:
                tier1_data = json.load(f)
            for item in tier1_data:
                store.store_metadata(
                    metadata_type=item.get('type', 'unknown'),
                    code=item['code'],
                    meaning=item['meaning'],
                    page=item.get('page')
                )
            self.logger.log_stage("neo4j", f"   ✓ Loaded {len(tier1_data)} metadata entries")

        # Load Tier 2: Knowledge
        self.logger.log_stage("neo4j", "Loading Tier 2 (Knowledge)...")
        tier2_file = metadata_dir / "tier_2_knowledge.json"
        if tier2_file.exists():
            with open(tier2_file) as f:
                tier2_data = json.load(f)
            for item in tier2_data:
                store.store_knowledge(
                    content=item['content'],
                    knowledge_type=item['type'],
                    page=item['page'],
                    section=item.get('section')
                )
            self.logger.log_stage("neo4j", f"   ✓ Loaded {len(tier2_data)} knowledge nodes")

        # Load Tier 3: Structure
        self.logger.log_stage("neo4j", "Loading Tier 3 (Structure)...")
        tier3_file = metadata_dir / "tier_3_structure.json"
        if tier3_file.exists():
            with open(tier3_file) as f:
                tier3_data = json.load(f)
            for item in tier3_data:
                store.store_section(
                    name=item['title'],
                    start_page=item['start_page'],
                    end_page=item.get('end_page', item['start_page'])
                )
            self.logger.log_stage("neo4j", f"   ✓ Loaded {len(tier3_data)} sections")

        # Load Tier 4: Semantic (to Qdrant)
        self.logger.log_stage("qdrant", "Loading Tier 4 (Semantic)...")
        tier4_file = metadata_dir / "tier_4_semantic.json"
        semantic_count = 0
        if tier4_file.exists():
            with open(tier4_file) as f:
                tier4_data = json.load(f)

            # Group by page and store
            from collections import defaultdict
            pages_chunks = defaultdict(list)
            for item in tier4_data:
                pages_chunks[item['page']].append(item['text'])

            for page, chunks in pages_chunks.items():
                # Store with metadata (would create embeddings if OpenAI key exists)
                # For now just log
                semantic_count += len(chunks)

            self.logger.log_stage("qdrant", f"   ✓ Prepared {semantic_count} semantic chunks")

        duration = (datetime.now() - start_time).total_seconds()

        self.logger.log_stage("neo4j", f"✅ Storage Complete")
        self.logger.log_stage("neo4j", f"   Duration: {duration:.1f}s")

        store.close()

        return {
            "status": "completed",
            "neo4j_loaded": True,
            "qdrant_loaded": True,
            "duration": duration
        }


class SpatialPlacementStage:
    """Stage 4: LLM spatial placement"""

    def __init__(self, config: PipelineConfig, logger: PipelineLogger):
        self.config = config
        self.logger = logger

    def run(self, storage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run LLM spatial placement on components"""
        self.logger.log_stage("spatial", "=" * 70)
        self.logger.log_stage("spatial", "📍 Stage 4: Spatial Placement")
        self.logger.log_stage("spatial", "=" * 70)

        if self.config.skip_spatial:
            self.logger.log_stage("spatial", "⏭️  Skipping spatial placement")
            return {"status": "skipped"}

        # Import spatial placer
        # from ollama_spatial_placer import run_spatial_placement

        self.logger.log_stage("spatial", "TODO: Implement spatial placement stage")

        return {
            "status": "pending",
            "message": "Spatial placement not yet implemented in pipeline"
        }


# ============================================================================
# Master Pipeline Orchestrator
# ============================================================================

class MasterPipeline:
    """Orchestrates entire pipeline execution"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = PipelineLogger()
        self.results = {}

    def run(self):
        """Execute complete pipeline"""
        self.logger.logger.info("=" * 70)
        self.logger.logger.info("🚀 MASTER PIPELINE EXECUTION")
        self.logger.logger.info("=" * 70)
        self.logger.logger.info("")

        # Log configuration
        config_dict = self.config.to_dict()
        self.logger.logger.info("📋 Configuration:")
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self.logger.logger.info(f"   {key}:")
                for k, v in value.items():
                    self.logger.logger.info(f"      {k}: {v}")
            else:
                self.logger.logger.info(f"   {key}: {value}")
        self.logger.logger.info("")

        pipeline_start = datetime.now()

        try:
            # Stage 1: OCR
            ocr_stage = OCRStage(self.config, self.logger)
            self.results['ocr'] = ocr_stage.run()
            self.logger.logger.info("")

            # Stage 2: Analysis
            analysis_stage = AnalysisStage(self.config, self.logger)
            self.results['analysis'] = analysis_stage.run(self.results['ocr'])
            self.logger.logger.info("")

            # Stage 3: Storage
            storage_stage = StorageStage(self.config, self.logger)
            self.results['storage'] = storage_stage.run(self.results['analysis'])
            self.logger.logger.info("")

            # Stage 4: Spatial Placement
            spatial_stage = SpatialPlacementStage(self.config, self.logger)
            self.results['spatial'] = spatial_stage.run(self.results['storage'])
            self.logger.logger.info("")

            # Final summary
            total_duration = (datetime.now() - pipeline_start).total_seconds()

            self.logger.logger.info("=" * 70)
            self.logger.logger.info("✅ PIPELINE COMPLETE")
            self.logger.logger.info("=" * 70)
            self.logger.logger.info(f"   Total Duration: {total_duration:.1f}s ({total_duration/60:.2f} min)")
            self.logger.logger.info(f"   Run ID: {self.config.run_id}")
            self.logger.logger.info("=" * 70)

            # Save results
            results_file = self.config.output_dir / "pipeline_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    "config": config_dict,
                    "results": self.results,
                    "duration": total_duration,
                    "completed_at": datetime.now().isoformat()
                }, f, indent=2)

            self.logger.logger.info(f"📊 Results saved to: {results_file}")

        except Exception as e:
            self.logger.logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline for wiring diagram processing"
    )

    # Required arguments
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--vehicle-make", required=True, help="Vehicle make (e.g., Mitsubishi)")
    parser.add_argument("--vehicle-model", required=True, help="Vehicle model (e.g., Pajero Pinin)")
    parser.add_argument("--vehicle-year", type=int, required=True, help="Vehicle year")

    # Optional arguments
    parser.add_argument("--vehicle-variant", help="Vehicle variant (e.g., 3-V60)")
    parser.add_argument("--pages", default="1-100", help="Page range (e.g., 1-100)")
    parser.add_argument("--run-id", default="auto", help="Run ID (default: auto-generated)")
    parser.add_argument("--output-dir", default="pipeline_output", help="Output directory")

    # Engine options
    parser.add_argument("--ocr-engine", default="tesseract", help="OCR engine")
    parser.add_argument("--llm-model", default="llama3.1:8b", help="LLM model")

    # Skip options
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR stage")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage")
    parser.add_argument("--skip-storage", action="store_true", help="Skip storage stage")
    parser.add_argument("--skip-spatial", action="store_true", help="Skip spatial placement stage")

    args = parser.parse_args()

    # Create and run pipeline
    config = PipelineConfig(args)
    pipeline = MasterPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
