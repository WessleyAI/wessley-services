"""
PDF processing and conversion to images.
"""
import os
import tempfile
from typing import List, Optional, Tuple
from pathlib import Path

try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

import cv2
import numpy as np
from PIL import Image

from ..core.schemas import PageImage


class PdfProcessor:
    """
    PDF processing and conversion to images for OCR.
    """
    
    def __init__(self, dpi: int = 300, format: str = "PNG"):
        """
        Initialize PDF processor.
        
        Args:
            dpi: DPI for rasterization (300-600 recommended for OCR)
            format: Output image format (PNG, JPEG)
        """
        self.dpi = dpi
        self.format = format.upper()
        
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError(
                "pdf2image not available. Install with: pip install pdf2image\n"
                "Also requires poppler-utils: apt-get install poppler-utils (Linux) "
                "or brew install poppler (macOS)"
            )
    
    async def convert_pdf_to_images(
        self, 
        pdf_path: str, 
        page_range: Optional[Tuple[int, int]] = None,
        output_dir: Optional[str] = None
    ) -> List[PageImage]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to input PDF file
            page_range: Optional tuple (first_page, last_page) 1-indexed
            output_dir: Output directory (uses temp dir if None)
            
        Returns:
            List of PageImage objects with converted pages
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Setup output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="pdf_conversion_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Convert PDF to images
            images = self._convert_pdf_pages(pdf_path, page_range)
            
            # Save images and create PageImage objects
            page_images = []
            for i, image in enumerate(images):
                page_num = i + 1
                if page_range:
                    page_num = page_range[0] + i
                
                # Save image
                output_path = os.path.join(output_dir, f"page_{page_num:03d}.{self.format.lower()}")
                image.save(output_path, self.format)
                
                # Create PageImage object
                page_image = PageImage(
                    page=page_num,
                    dpi=self.dpi,
                    width=image.width,
                    height=image.height,
                    file_path=output_path
                )
                
                page_images.append(page_image)
            
            return page_images
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")
    
    def _convert_pdf_pages(
        self, 
        pdf_path: str, 
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Image.Image]:
        """
        Convert PDF pages to PIL Images.
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional page range (first_page, last_page) 1-indexed
            
        Returns:
            List of PIL Images
        """
        try:
            # Setup conversion parameters
            kwargs = {
                'dpi': self.dpi,
                'fmt': self.format.lower(),
                'thread_count': 2,  # Limit threads for container environments
                'use_cropbox': True,  # Use PDF cropbox if available
                'strict': False,  # Don't fail on minor PDF issues
            }
            
            # Add page range if specified
            if page_range:
                kwargs['first_page'] = page_range[0]
                kwargs['last_page'] = page_range[1]
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, **kwargs)
            
            return images
            
        except Exception as e:
            # Try alternative method with bytes
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                
                images = convert_from_bytes(pdf_bytes, **kwargs)
                return images
                
            except Exception as e2:
                raise RuntimeError(f"PDF conversion failed with both methods: {e}, {e2}")
    
    def get_pdf_info(self, pdf_path: str) -> dict:
        """
        Get PDF metadata and page information.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with PDF information
        """
        try:
            from PyPDF2 import PdfReader
            
            reader = PdfReader(pdf_path)
            
            info = {
                "page_count": len(reader.pages),
                "metadata": dict(reader.metadata) if reader.metadata else {},
                "encrypted": reader.is_encrypted,
                "pages": []
            }
            
            # Get page dimensions
            for i, page in enumerate(reader.pages):
                page_info = {
                    "page_number": i + 1,
                    "width": float(page.mediabox.width),
                    "height": float(page.mediabox.height),
                    "rotation": int(page.rotation) if page.rotation else 0
                }
                info["pages"].append(page_info)
            
            return info
            
        except ImportError:
            # Fallback without detailed metadata
            try:
                # Use pdf2image to get basic page count
                images = convert_from_path(pdf_path, dpi=72, first_page=1, last_page=1)
                
                # Estimate total pages (this is a rough estimate)
                # In practice, we'd need to convert all pages or use another method
                return {
                    "page_count": None,  # Unknown without full conversion
                    "metadata": {},
                    "encrypted": False,
                    "pages": [{
                        "page_number": 1,
                        "width": images[0].width,
                        "height": images[0].height,
                        "rotation": 0
                    }] if images else []
                }
                
            except Exception:
                return {
                    "page_count": None,
                    "metadata": {},
                    "encrypted": False,
                    "pages": []
                }
    
    def optimize_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Apply OCR-specific optimizations to converted image.
        
        Args:
            image: PIL Image from PDF conversion
            
        Returns:
            Optimized PIL Image
        """
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply sharpening filter for text clarity
        from PIL import ImageFilter, ImageEnhance
        
        # Sharpen
        image = image.filter(ImageFilter.SHARPEN)
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance brightness if image is too dark
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def extract_text_regions(self, page_image: PageImage) -> List[dict]:
        """
        Extract potential text regions from PDF-converted image.
        
        This can be used to guide OCR processing to specific areas.
        
        Args:
            page_image: PageImage object
            
        Returns:
            List of text region dictionaries with coordinates
        """
        # Load image
        image = cv2.imread(page_image.file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        
        # Dilate to connect text characters
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and format text regions
        text_regions = []
        min_area = 100  # Minimum area for text region
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter out regions that are too wide/thin (likely not text)
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 20:  # Reasonable aspect ratio for text
                    text_regions.append({
                        "bbox": [x, y, x + w, y + h],
                        "area": area,
                        "aspect_ratio": aspect_ratio
                    })
        
        # Sort by area (largest first)
        text_regions.sort(key=lambda x: x["area"], reverse=True)
        
        return text_regions
    
    def validate_pdf(self, pdf_path: str) -> bool:
        """
        Validate that file is a readable PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if valid PDF, False otherwise
        """
        try:
            # Try to get basic info
            info = self.get_pdf_info(pdf_path)
            return True
            
        except Exception:
            return False