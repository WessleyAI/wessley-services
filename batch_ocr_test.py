#!/usr/bin/env python3
"""
Batch OCR Comparison: Tesseract vs Ollama Vision
Process first 100 pages of Pajero PDF with detailed logging
"""

import os
import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from pdf2image import convert_from_path
import ollama
from tqdm import tqdm

# Configuration
PDF_PATH = "/Users/moon/workspace/wessley.ai/services/public/Mitsubishi-Pajero-Pinin-3-V60-2000-2003-–-Wiring-Diagrams.pdf"
OUTPUT_DIR = "/Users/moon/workspace/wessley.ai/services/ocr_batch_results"
FIRST_PAGE = 1
LAST_PAGE = 100
DPI = 300

# Create output directories FIRST (before logging setup)
Path(OUTPUT_DIR).mkdir(exist_ok=True)
Path(f"{OUTPUT_DIR}/pages").mkdir(exist_ok=True)
Path(f"{OUTPUT_DIR}/tesseract").mkdir(exist_ok=True)
Path(f"{OUTPUT_DIR}/ollama").mkdir(exist_ok=True)

# Setup detailed logging (after directory creation)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'{OUTPUT_DIR}/ocr_batch.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def log_progress(current, total, operation, start_time):
    """Log detailed progress with ETA"""
    elapsed = (datetime.now() - start_time).total_seconds()
    if current > 0:
        avg_time = elapsed / current
        remaining = total - current
        eta_seconds = remaining * avg_time
        eta = timedelta(seconds=int(eta_seconds))
        logger.info(f"{operation}: {current}/{total} ({current/total*100:.1f}%) - ETA: {eta}")

if __name__ == "__main__":
    if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🔬 Batch OCR Comparison Test")
    logger.info("=" * 70)
    logger.info(f"PDF: {PDF_PATH}")
    logger.info(f"Pages: {FIRST_PAGE} to {LAST_PAGE}")
    logger.info(f"DPI: {DPI}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info(f"Log file: {OUTPUT_DIR}/ocr_batch.log")
    logger.info("")

    # ============================================================================
    # Step 1: Extract all 100 pages as images
    # ============================================================================

    logger.info(f"📄 Step 1: Extracting pages {FIRST_PAGE}-{LAST_PAGE} from PDF (DPI={DPI})...")
    logger.info("   This may take a few minutes...")
    try:
    start = datetime.now()
    logger.info(f"   Starting PDF extraction at {start.strftime('%H:%M:%S')}")

    images = convert_from_path(
        PDF_PATH,
        first_page=FIRST_PAGE,
        last_page=LAST_PAGE,
        dpi=DPI
    )
    duration = (datetime.now() - start).total_seconds()

    logger.info(f"✓ Extracted {len(images)} pages in {duration:.1f}s ({duration/60:.2f} min)")
    logger.info(f"  Average: {duration/len(images):.2f}s per page")
    logger.info(f"  Saving images to {OUTPUT_DIR}/pages/...")

    # Save all images
    save_start = datetime.now()
    for i, img in enumerate(tqdm(images, desc="Saving pages", file=sys.stdout), start=FIRST_PAGE):
        img.save(f"{OUTPUT_DIR}/pages/page_{i:03d}.png", "PNG")
        if i % 20 == 0:
            logger.info(f"   Saved {i - FIRST_PAGE + 1}/{len(images)} pages...")

    save_duration = (datetime.now() - save_start).total_seconds()
    logger.info(f"✓ All pages saved in {save_duration:.1f}s")
    logger.info(f"  Total extraction time: {(duration + save_duration):.1f}s")
    logger.info("")

    except Exception as e:
    logger.error(f"❌ Error extracting PDF: {e}", exc_info=True)
    sys.exit(1)

    # ============================================================================
    # Step 2: Check Tesseract availability
    # ============================================================================

    TESSERACT_AVAILABLE = False
    logger.info("📝 Step 2: Checking Tesseract availability...")
    try:
    import pytesseract
    from PIL import Image

    # Check if tesseract binary exists
    version = pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    logger.info(f"✓ Tesseract installed: {version}")
    logger.info(f"  Binary path: {pytesseract.pytesseract.tesseract_cmd}")
    logger.info("")
    except Exception as e:
    logger.warning(f"⚠️  Tesseract not available: {e}")
    logger.warning(f"   Install with: brew install tesseract")
    logger.info("")

    # ============================================================================
    # Step 3: Run Tesseract OCR on all pages
    # ============================================================================

    tesseract_results = {}
    tesseract_stats = {"total_elements": 0, "total_time": 0, "errors": 0}

    if TESSERACT_AVAILABLE:
    logger.info(f"📝 Step 3: Running Tesseract OCR on {len(images)} pages...")
    logger.info(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
    tesseract_start = datetime.now()

    for idx, page_num in enumerate(range(FIRST_PAGE, LAST_PAGE + 1), 1):
        try:
            page_start = datetime.now()

            # Load image
            img_path = f"{OUTPUT_DIR}/pages/page_{page_num:03d}.png"
            image = Image.open(img_path)

            # Run OCR with detailed data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            duration = (datetime.now() - page_start).total_seconds()

            # Extract text elements with bounding boxes
            elements = []
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                if text and ocr_data['conf'][i] > 0:  # Skip empty and low confidence
                    elements.append({
                        "text": text,
                        "bbox": [
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['left'][i] + ocr_data['width'][i],
                            ocr_data['top'][i] + ocr_data['height'][i]
                        ],
                        "confidence": ocr_data['conf'][i] / 100.0,
                        "page": page_num
                    })

            # Save page result
            result = {
                "page": page_num,
                "elements": elements,
                "duration_sec": duration,
                "element_count": len(elements)
            }

            with open(f"{OUTPUT_DIR}/tesseract/page_{page_num:03d}.json", "w") as f:
                json.dump(result, f, indent=2)

            tesseract_results[page_num] = result
            tesseract_stats["total_elements"] += len(elements)
            tesseract_stats["total_time"] += duration

            # Log progress every 10 pages
            if idx % 10 == 0:
                elapsed = (datetime.now() - tesseract_start).total_seconds()
                avg_time = elapsed / idx
                remaining = len(images) - idx
                eta = timedelta(seconds=int(remaining * avg_time))
                logger.info(f"   Tesseract: {idx}/{len(images)} pages ({idx/len(images)*100:.1f}%) | "
                           f"Avg: {avg_time:.2f}s/page | ETA: {eta} | "
                           f"Last page: {len(elements)} elements in {duration:.2f}s")

        except Exception as e:
            tesseract_stats["errors"] += 1
            logger.error(f"  ❌ Tesseract error on page {page_num}: {e}")

    logger.info(f"✓ Tesseract OCR completed!")
    logger.info(f"  Total elements: {tesseract_stats['total_elements']}")
    logger.info(f"  Total time: {tesseract_stats['total_time']:.1f}s ({tesseract_stats['total_time']/60:.2f} min)")
    logger.info(f"  Avg time/page: {tesseract_stats['total_time']/len(images):.2f}s")
    logger.info(f"  Avg elements/page: {tesseract_stats['total_elements']/len(images):.1f}")
    logger.info(f"  Errors: {tesseract_stats['errors']}")
    logger.info("")

    # ============================================================================
    # Step 4: Run Ollama Vision OCR on all pages
    # ============================================================================

    ollama_results = {}
    ollama_stats = {"total_elements": 0, "total_time": 0, "errors": 0, "parse_errors": 0}

    logger.info(f"🤖 Step 4: Running Ollama Vision OCR (llama3.2) on {len(images)} pages...")
    logger.info(f"   This will take longer (estimated 5-10 sec/page)...")
    logger.info(f"   Started at: {datetime.now().strftime('%H:%M:%S')}")
    ollama_start = datetime.now()

    for idx, page_num in enumerate(range(FIRST_PAGE, LAST_PAGE + 1), 1):
    try:
        page_start = datetime.now()

        img_path = f"{OUTPUT_DIR}/pages/page_{page_num:03d}.png"

        response = ollama.chat(
            model="llama3.2:latest",
            messages=[{
                "role": "user",
                "content": """Extract ALL visible text from this automotive wiring diagram page.

    Return as JSON array of text elements:
    [{"text": "exact text", "type": "component|value|label|other"}]

    Extract: component IDs, values, wire labels, fuse numbers, relay codes, ground points, connectors, everything visible.""",
                "images": [img_path]
            }],
            format="json",
            options={
                "temperature": 0.1,
                "num_predict": 1500
            }
        )

        duration = (datetime.now() - page_start).total_seconds()

        ollama_raw = response['message']['content']

        # Try to parse JSON
        try:
            parsed = json.loads(ollama_raw)
            elements = parsed if isinstance(parsed, list) else []
        except:
            elements = []
            ollama_stats["parse_errors"] += 1
            logger.warning(f"  ⚠️  Page {page_num}: Failed to parse JSON response")

        # Save page result
        result = {
            "page": page_num,
            "elements": elements,
            "raw_response": ollama_raw,
            "duration_sec": duration,
            "element_count": len(elements)
        }

        with open(f"{OUTPUT_DIR}/ollama/page_{page_num:03d}.json", "w") as f:
            json.dump(result, f, indent=2)

        ollama_results[page_num] = result
        ollama_stats["total_elements"] += len(elements)
        ollama_stats["total_time"] += duration

        # Log progress every 10 pages
        if idx % 10 == 0:
            elapsed = (datetime.now() - ollama_start).total_seconds()
            avg_time = elapsed / idx
            remaining = len(images) - idx
            eta = timedelta(seconds=int(remaining * avg_time))
            logger.info(f"   Ollama: {idx}/{len(images)} pages ({idx/len(images)*100:.1f}%) | "
                       f"Avg: {avg_time:.2f}s/page | ETA: {eta} | "
                       f"Last page: {len(elements)} elements in {duration:.2f}s")

    except Exception as e:
        ollama_stats["errors"] += 1
        logger.error(f"  ❌ Ollama error on page {page_num}: {e}")

    logger.info(f"✓ Ollama Vision OCR completed!")
    logger.info(f"  Total elements: {ollama_stats['total_elements']}")
    logger.info(f"  Total time: {ollama_stats['total_time']:.1f}s ({ollama_stats['total_time']/60:.2f} min)")
    logger.info(f"  Avg time/page: {ollama_stats['total_time']/len(images):.2f}s")
    logger.info(f"  Avg elements/page: {ollama_stats['total_elements']/len(images):.1f}")
    logger.info(f"  Errors: {ollama_stats['errors']}")
    logger.info(f"  Parse errors: {ollama_stats['parse_errors']}")
    logger.info("")

    # ============================================================================
    # Step 5: Generate comparison report
    # ============================================================================

    logger.info("=" * 70)
    logger.info("📊 Step 5: Generating Comparison Report")
    logger.info("=" * 70)
    logger.info("")

    comparison = {
    "test_date": datetime.now().isoformat(),
    "pdf": PDF_PATH,
    "pages": f"{FIRST_PAGE}-{LAST_PAGE}",
    "total_pages": len(images),
    "dpi": DPI,
    "tesseract": {
        "available": TESSERACT_AVAILABLE,
        "total_elements": tesseract_stats["total_elements"],
        "total_time_sec": tesseract_stats["total_time"],
        "avg_time_per_page": tesseract_stats["total_time"]/len(images) if TESSERACT_AVAILABLE else 0,
        "avg_elements_per_page": tesseract_stats["total_elements"]/len(images) if TESSERACT_AVAILABLE else 0,
        "errors": tesseract_stats["errors"]
    },
    "ollama": {
        "model": "llama3.2:latest",
        "total_elements": ollama_stats["total_elements"],
        "total_time_sec": ollama_stats["total_time"],
        "avg_time_per_page": ollama_stats["total_time"]/len(images),
        "avg_elements_per_page": ollama_stats["total_elements"]/len(images),
        "errors": ollama_stats["errors"],
        "parse_errors": ollama_stats["parse_errors"]
    }
    }

    # Save comparison report
    with open(f"{OUTPUT_DIR}/comparison_report.json", "w") as f:
    json.dump(comparison, f, indent=2)

    logger.info("=" * 70)
    logger.info("ENGINE COMPARISON")
    logger.info("=" * 70)
    logger.info("")

    if TESSERACT_AVAILABLE:
    logger.info("Tesseract OCR:")
    logger.info(f"  Elements extracted: {tesseract_stats['total_elements']:,}")
    logger.info(f"  Total time: {tesseract_stats['total_time']:.1f}s ({tesseract_stats['total_time']/60:.2f} min)")
    logger.info(f"  Speed: {tesseract_stats['total_time']/len(images):.2f}s/page")
    logger.info(f"  Avg elements/page: {tesseract_stats['total_elements']/len(images):.1f}")
    logger.info(f"  Errors: {tesseract_stats['errors']}")
    logger.info("")

    logger.info("Ollama Vision (llama3.2):")
    logger.info(f"  Elements extracted: {ollama_stats['total_elements']:,}")
    logger.info(f"  Total time: {ollama_stats['total_time']:.1f}s ({ollama_stats['total_time']/60:.2f} min)")
    logger.info(f"  Speed: {ollama_stats['total_time']/len(images):.2f}s/page")
    logger.info(f"  Avg elements/page: {ollama_stats['total_elements']/len(images):.1f}")
    logger.info(f"  Errors: {ollama_stats['errors']}")
    logger.info(f"  Parse errors: {ollama_stats['parse_errors']}")
    logger.info("")

    if TESSERACT_AVAILABLE:
    logger.info("=" * 70)
    logger.info("WINNER ANALYSIS")
    logger.info("=" * 70)
    if tesseract_stats['total_time'] < ollama_stats['total_time']:
        speed_diff = ollama_stats['total_time']/tesseract_stats['total_time']
        logger.info(f"  ⚡ Speed: Tesseract is {speed_diff:.1f}x faster than Ollama")
    else:
        speed_diff = tesseract_stats['total_time']/ollama_stats['total_time']
        logger.info(f"  ⚡ Speed: Ollama is {speed_diff:.1f}x faster than Tesseract")

    if tesseract_stats['total_elements'] > ollama_stats['total_elements']:
        elem_diff = tesseract_stats['total_elements'] - ollama_stats['total_elements']
        logger.info(f"  📊 Coverage: Tesseract extracted {elem_diff:,} more elements ({elem_diff/ollama_stats['total_elements']*100:.1f}% more)")
    elif ollama_stats['total_elements'] > tesseract_stats['total_elements']:
        elem_diff = ollama_stats['total_elements'] - tesseract_stats['total_elements']
        logger.info(f"  📊 Coverage: Ollama extracted {elem_diff:,} more elements ({elem_diff/tesseract_stats['total_elements']*100:.1f}% more)")
    else:
        logger.info(f"  📊 Coverage: Both extracted the same number of elements")
    logger.info("")

    logger.info("=" * 70)
    logger.info("✅ BATCH OCR TEST COMPLETE!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Output files:")
    logger.info(f"  📁 Images: {OUTPUT_DIR}/pages/")
    logger.info(f"  📁 Tesseract results: {OUTPUT_DIR}/tesseract/")
    logger.info(f"  📁 Ollama results: {OUTPUT_DIR}/ollama/")
    logger.info(f"  📄 Comparison report: {OUTPUT_DIR}/comparison_report.json")
    logger.info(f"  📄 Log file: {OUTPUT_DIR}/ocr_batch.log")
    logger.info("")
    logger.info("🚀 Next steps:")
    logger.info("  1. Review results in ocr_batch_results/")
    logger.info("  2. Choose best OCR engine based on quality + speed")
    logger.info("  3. Create Neo4j components from OCR data")
    logger.info("  4. Test LLM spatial placer with contextual map")
    logger.info("")
    logger.info("💰 Fine Count: $0")
    logger.info("")
logger.info(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# FUNCTION API FOR MASTER PIPELINE
# ============================================================================

def run_tesseract_ocr(pdf_path: str, output_dir: str, start_page: int = 1, end_page: int = 100):
    """
    Run Tesseract OCR on specified page range (for master pipeline integration)

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files
        start_page: First page to process
        end_page: Last page to process

    Returns:
        dict with keys: pages_processed, total_elements, duration, errors
    """
    import pytesseract
    from PIL import Image

    logger.info(f"📄 Extracting pages {start_page}-{end_page} from PDF...")

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/tesseract").mkdir(exist_ok=True)

    # Extract images
    images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page, dpi=300)
    logger.info(f"✓ Extracted {len(images)} pages")

    # Run Tesseract OCR
    stats = {"total_elements": 0, "total_time": 0, "errors": 0}
    start_time = datetime.now()

    for idx, page_num in enumerate(range(start_page, end_page + 1), 1):
        try:
            page_start = datetime.now()

            # OCR this page
            img = images[idx - 1]
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            # Extract elements
            elements = []
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                text = ocr_data['text'][i].strip()
                if text and ocr_data['conf'][i] > 0:
                    elements.append({
                        "text": text,
                        "bbox": [
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['left'][i] + ocr_data['width'][i],
                            ocr_data['top'][i] + ocr_data['height'][i]
                        ],
                        "confidence": ocr_data['conf'][i] / 100.0,
                        "page": page_num
                    })

            # Save result
            result = {
                "page": page_num,
                "elements": elements,
                "duration_sec": (datetime.now() - page_start).total_seconds(),
                "element_count": len(elements)
            }

            with open(f"{output_dir}/tesseract/page_{page_num:03d}.json", "w") as f:
                json.dump(result, f, indent=2)

            stats["total_elements"] += len(elements)
            stats["total_time"] += result["duration_sec"]

            # Log progress every 50 pages
            if idx % 50 == 0:
                pct = (idx / len(images)) * 100
                avg = stats["total_time"] / idx
                eta = (len(images) - idx) * avg / 60
                logger.info(f"   {idx}/{len(images)} ({pct:.1f}%) | Avg: {avg:.1f}s/page | ETA: {eta:.1f} min")

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"❌ Error on page {page_num}: {e}")

    total_duration = (datetime.now() - start_time).total_seconds()

    return {
        "pages_processed": len(images),
        "total_elements": stats["total_elements"],
        "duration": total_duration,
        "errors": stats["errors"]
    }
