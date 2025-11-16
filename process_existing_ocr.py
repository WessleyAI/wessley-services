#!/usr/bin/env python3
"""
Process Existing OCR Results with Intelligent LLM Analysis

Uses pre-extracted OCR text (from Tesseract) and runs intelligent
LLM-based classification and extraction WITHOUT re-running OCR.

Usage:
    python3 process_existing_ocr.py \\
        --ocr-dir ocr_batch_results/tesseract \\
        --start-page 1 \\
        --end-page 15 \\
        --output metadata_intelligent
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Import from our intelligent extractor
from intelligent_metadata_extractor import (
    ContentAnalyzer,
    KnowledgeRouter,
    FailureLogger
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_ocr_text(ocr_file: str) -> str:
    """Load text from Tesseract OCR JSON file"""
    with open(ocr_file, 'r') as f:
        data = json.load(f)

    # Extract all text elements and join
    texts = [elem['text'] for elem in data.get('elements', [])]
    return ' '.join(texts)


def process_existing_ocr(ocr_dir: str,
                         start_page: int = 1,
                         end_page: int = 15,
                         model: str = "llama3.1") -> KnowledgeRouter:
    """
    Process existing OCR results with intelligent LLM analysis

    Args:
        ocr_dir: Directory containing OCR JSON files
        start_page: First page to process
        end_page: Last page to process
        model: LLM model for analysis

    Returns:
        KnowledgeRouter with extracted data
    """
    logger.info("=" * 70)
    logger.info("🧠 Intelligent Metadata Extraction (Using Existing OCR)")
    logger.info("=" * 70)
    logger.info(f"OCR Directory: {ocr_dir}")
    logger.info(f"Pages: {start_page}-{end_page}")
    logger.info(f"Model: {model}")
    logger.info("")

    # Initialize components
    failure_logger = FailureLogger()
    analyzer = ContentAnalyzer(model=model, failure_logger=failure_logger)
    router = KnowledgeRouter()

    # Process each page
    total_pages = end_page - start_page + 1
    errors = 0
    skipped = 0

    from datetime import datetime
    import time

    start_time = time.time()

    for idx, page_num in enumerate(range(start_page, end_page + 1), 1):
        page_start = time.time()
        progress_pct = (idx / total_pages) * 100

        logger.info("=" * 70)
        logger.info(f"📄 Page {page_num}/{end_page} ({progress_pct:.1f}% complete)")
        logger.info("=" * 70)

        # Load OCR text
        ocr_file = os.path.join(ocr_dir, f"page_{page_num:03d}.json")

        if not os.path.exists(ocr_file):
            logger.warning(f"   ⚠️  OCR file not found: {ocr_file}")
            skipped += 1
            continue

        try:
            logger.info("   1️⃣  Loading existing OCR text...")
            text = load_ocr_text(ocr_file)
            logger.info(f"   ✓ Loaded {len(text)} characters")

            if len(text) < 10:
                logger.warning(f"   ⚠️  Page has very little text, skipping")
                skipped += 1
                continue

            # Step 2: Analyze page type
            logger.info("   2️⃣  Analyzing page type with LLM...")
            page_analysis = analyzer.analyze_page_type(text, page_num)
            logger.info(f"   ✓ Type: {page_analysis['page_type']}")
            logger.info(f"   ✓ Tier: {page_analysis['storage_tier']} ({page_analysis['content_category']})")
            logger.info(f"   ✓ Confidence: {page_analysis.get('confidence', 0.0):.2f}")

            # Step 3: Extract structured data
            logger.info("   3️⃣  Extracting structured data...")
            structured_data = analyzer.extract_structured_data(
                text,
                page_analysis['page_type'],
                page_num
            )
            data_items = len(structured_data.get('entries', [])) or len(structured_data.get('sections', [])) or len(structured_data.get('specs', []))
            logger.info(f"   ✓ Extracted {data_items} structured items")

            # Step 4: Extract semantic chunks
            logger.info("   4️⃣  Creating semantic chunks...")
            semantic_chunks = analyzer.extract_semantic_knowledge(text, page_num)
            logger.info(f"   ✓ Created {len(semantic_chunks)} semantic chunks")

            # Step 5: Add to accumulated context for future pages
            logger.info("   5️⃣  Adding to context for future pages...")
            analyzer.add_to_context(structured_data, page_analysis['page_type'])

            # Step 6: Route to storage
            logger.info("   6️⃣  Routing to storage tiers...")
            router.route(page_analysis, structured_data, semantic_chunks, page_num)

            # Page timing
            page_duration = time.time() - page_start
            avg_time = (time.time() - start_time) / idx
            eta_seconds = avg_time * (total_pages - idx)
            eta_minutes = eta_seconds / 60

            logger.info(f"   ⏱️  Page time: {page_duration:.1f}s | Avg: {avg_time:.1f}s/page | ETA: {eta_minutes:.1f} min")
            logger.info(f"   ✅ Page {page_num} complete")

        except Exception as e:
            logger.error(f"   ❌ Error processing page {page_num}: {e}")
            errors += 1

        logger.info("")

    # Summary
    total_duration = time.time() - start_time
    logger.info("=" * 70)
    logger.info("✅ Extraction Complete")
    logger.info("=" * 70)
    logger.info(f"   Total pages processed: {total_pages}")
    logger.info(f"   Successful: {total_pages - errors - skipped}")
    logger.info(f"   Errors: {errors}")
    logger.info(f"   Skipped: {skipped}")
    logger.info(f"   Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    logger.info(f"   Average: {total_duration/total_pages:.1f}s/page")
    logger.info("")
    summary = router.get_storage_summary()
    for tier, count in summary.items():
        logger.info(f"   {tier}: {count} items")
    logger.info("=" * 70)

    # Generate failure analysis report
    failure_logger.generate_summary()
    if failure_logger.failures:
        logger.info(f"\n⚠️  {len(failure_logger.failures)} failures logged")
        logger.info(f"   Failure report: {failure_logger.summary_file}")

    return router


def main():
    parser = argparse.ArgumentParser(
        description="Process existing OCR results with intelligent LLM analysis"
    )
    parser.add_argument(
        "--ocr-dir",
        required=True,
        help="Directory containing OCR JSON files"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        help="First page to process (default: 1)"
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=15,
        help="Last page to process (default: 15)"
    )
    parser.add_argument(
        "--model",
        default="llama3.1",
        help="LLM model (default: llama3.1)"
    )
    parser.add_argument(
        "--output",
        default="metadata_intelligent",
        help="Output directory (default: metadata_intelligent)"
    )

    args = parser.parse_args()

    # Run extraction
    router = process_existing_ocr(
        ocr_dir=args.ocr_dir,
        start_page=args.start_page,
        end_page=args.end_page,
        model=args.model
    )

    # Save results
    router.save_to_json(args.output)

    logger.info(f"\n✅ Metadata saved to: {args.output}/")
    logger.info("\nFine Count: $0")


if __name__ == "__main__":
    main()
