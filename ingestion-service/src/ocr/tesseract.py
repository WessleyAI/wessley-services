"""
Tesseract OCR provider implementation.
"""
import os
from typing import List, Dict, Any
import cv2
import numpy as np
import pytesseract
from PIL import Image

from .base import OcrProvider
from ..core.schemas import PageImage, TextSpan, OcrEngine


class TesseractProvider(OcrProvider):
    """
    Tesseract OCR provider with configurable languages and parameters.
    """
    
    def __init__(self, languages: str = "eng", config: Dict[str, Any] = None):
        """
        Initialize Tesseract provider.
        
        Args:
            languages: Language codes for Tesseract (e.g., "eng", "eng+fra")
            config: Additional Tesseract configuration parameters
        """
        self.languages = languages
        self.config = config or {}
        
        # Default Tesseract config for electrical schematics
        self.default_config = {
            # Page segmentation modes:
            # 6 = Uniform block of text
            # 11 = Sparse text. Find as much text as possible in no particular order
            # 13 = Raw line. Treat the image as a single text line
            "psm": 11,
            
            # OCR Engine modes:
            # 0 = Legacy engine only
            # 1 = Neural nets LSTM engine only  
            # 2 = Legacy + LSTM engines
            # 3 = Default, based on what is available
            "oem": 3,
            
            # Whitelist characters (useful for component labels)
            # "tessedit_char_whitelist": "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.kMΩμ",
        }
        
        # Merge user config with defaults
        self.tesseract_config = {**self.default_config, **self.config}
        
        # Verify Tesseract is available
        self._verify_tesseract()
    
    def _verify_tesseract(self) -> None:
        """Verify Tesseract is installed and accessible."""
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(f"Tesseract not found or not accessible: {e}")
    
    def _build_config_string(self) -> str:
        """Build Tesseract configuration string."""
        config_parts = []
        
        for key, value in self.tesseract_config.items():
            if key == "psm":
                config_parts.append(f"--psm {value}")
            elif key == "oem":
                config_parts.append(f"--oem {value}")
            else:
                config_parts.append(f"-c {key}={value}")
        
        return " ".join(config_parts)
    
    async def extract_text(self, page_image: PageImage) -> List[TextSpan]:
        """
        Extract text from page image using Tesseract.
        
        Args:
            page_image: Processed page image
            
        Returns:
            List of text spans with coordinates and confidence
        """
        try:
            # Load image
            image = self._load_image(page_image.file_path)
            
            # Get detailed OCR data with bounding boxes
            config_string = self._build_config_string()
            
            # Use pytesseract to get detailed data
            data = pytesseract.image_to_data(
                image,
                lang=self.languages,
                config=config_string,
                output_type=pytesseract.Output.DICT
            )
            
            # Convert to TextSpan objects
            text_spans = self._parse_tesseract_data(data, page_image.page)
            
            return text_spans
            
        except Exception as e:
            print(f"Tesseract OCR error for page {page_image.page}: {e}")
            return []
    
    def _load_image(self, file_path: str) -> np.ndarray:
        """Load image from file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Load with OpenCV (handles various formats)
        image = cv2.imread(file_path)
        if image is None:
            # Fallback to PIL
            pil_image = Image.open(file_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return image
    
    def _parse_tesseract_data(self, data: Dict[str, List], page: int) -> List[TextSpan]:
        """
        Parse Tesseract OCR data into TextSpan objects.
        
        Args:
            data: Tesseract output data dictionary
            page: Page number
            
        Returns:
            List of TextSpan objects
        """
        text_spans = []
        
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            confidence = int(data["conf"][i])
            
            # Skip empty text or very low confidence
            if not text or confidence < 0:
                continue
            
            # Extract bounding box coordinates
            x = data["left"][i]
            y = data["top"][i]
            width = data["width"][i]
            height = data["height"][i]
            
            # Convert to [x1, y1, x2, y2] format
            bbox = [float(x), float(y), float(x + width), float(y + height)]
            
            # Convert confidence from 0-100 to 0-1
            confidence_normalized = confidence / 100.0
            
            # Create TextSpan
            text_span = TextSpan(
                page=page,
                bbox=bbox,
                text=text,
                rotation=0,  # Tesseract doesn't provide rotation directly
                confidence=confidence_normalized,
                engine=OcrEngine.TESSERACT
            )
            
            text_spans.append(text_span)
        
        return text_spans
    
    @property
    def engine_name(self) -> str:
        """Return the name of this OCR engine."""
        return "tesseract"
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        try:
            langs = pytesseract.get_languages()
            return langs
        except Exception:
            return ["eng"]  # Default fallback
    
    def set_languages(self, languages: str) -> None:
        """Update language configuration."""
        self.languages = languages
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update Tesseract configuration."""
        self.tesseract_config.update(config)