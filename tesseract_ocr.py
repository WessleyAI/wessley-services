#!/usr/bin/env python3
"""
Simple Tesseract OCR Module for Master Pipeline
Extracts text from PDF pages with bounding boxes and confidence scores
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


def run_tesseract_ocr(pdf_path: str, output_dir: str, start_page: int = 1, end_page: int = 680):
    """
    Run Tesseract OCR on specified page range

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory for output files
        start_page: First page to process
        end_page: Last page to process

    Returns:
        dict with keys: pages_processed, total_elements, duration, errors
    """
    logger.info(f"📄 Extracting pages {start_page}-{end_page} from PDF...")

    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tesseract_dir = Path(output_dir) / "tesseract"
    tesseract_dir.mkdir(exist_ok=True)

    # Extract images from PDF
    start_time = datetime.now()
    images = convert_from_path(pdf_path, first_page=start_page, last_page=end_page, dpi=300)
    logger.info(f"✓ Extracted {len(images)} pages from PDF")

    # Run Tesseract OCR on each page
    stats = {"total_elements": 0, "total_time": 0, "errors": 0}
    ocr_start = datetime.now()

    for idx, page_num in enumerate(range(start_page, end_page + 1), 1):
        try:
            page_start = datetime.now()

            # Get the image for this page
            img = images[idx - 1]

            # Run OCR with detailed data
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

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
            page_duration = (datetime.now() - page_start).total_seconds()
            result = {
                "page": page_num,
                "elements": elements,
                "duration_sec": page_duration,
                "element_count": len(elements)
            }

            output_file = tesseract_dir / f"page_{page_num:03d}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            stats["total_elements"] += len(elements)
            stats["total_time"] += page_duration

            # Log progress every 50 pages
            if idx % 50 == 0:
                elapsed = (datetime.now() - ocr_start).total_seconds()
                avg_time = elapsed / idx
                remaining_pages = len(images) - idx
                eta_seconds = remaining_pages * avg_time
                eta_minutes = eta_seconds / 60

                pct = (idx / len(images)) * 100
                logger.info(f"   Progress: {idx}/{len(images)} ({pct:.1f}%) | "
                           f"Avg: {avg_time:.1f}s/page | "
                           f"ETA: {eta_minutes:.1f} min | "
                           f"Last: {len(elements)} elements")

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"❌ Error processing page {page_num}: {e}")

    total_duration = (datetime.now() - start_time).total_seconds()

    logger.info(f"✅ OCR Complete")
    logger.info(f"   Pages: {len(images)}")
    logger.info(f"   Elements: {stats['total_elements']:,}")
    logger.info(f"   Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    logger.info(f"   Average: {stats['total_time']/len(images):.2f}s/page")
    logger.info(f"   Errors: {stats['errors']}")

    return {
        "pages_processed": len(images),
        "total_elements": stats["total_elements"],
        "duration": total_duration,
        "errors": stats["errors"]
    }
