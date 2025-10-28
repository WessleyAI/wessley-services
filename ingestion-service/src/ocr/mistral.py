"""
Mistral OCR provider implementation using API.
"""
import os
import base64
from typing import List, Dict, Any, Optional
import httpx
import asyncio
from PIL import Image
import io

from .base import OcrProvider
from ..core.schemas import PageImage, TextSpan, OcrEngine


class MistralProvider(OcrProvider):
    """
    Mistral OCR provider using their vision API for text extraction.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "mistral-large-latest",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize Mistral OCR provider.
        
        Args:
            api_key: Mistral API key (or from environment)
            api_url: Mistral API endpoint URL
            model: Model name to use for OCR
            max_retries: Maximum number of API retries
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.api_url = api_url or os.getenv("MISTRAL_API_URL", "https://api.mistral.ai/v1")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("Mistral API key not provided. Set MISTRAL_API_KEY environment variable.")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def extract_text(self, page_image: PageImage) -> List[TextSpan]:
        """
        Extract text from page image using Mistral Vision API.
        
        Args:
            page_image: Processed page image
            
        Returns:
            List of text spans with coordinates and confidence
        """
        try:
            # Encode image to base64
            image_b64 = await self._encode_image_to_base64(page_image.file_path)
            
            # Create OCR prompt
            prompt = self._create_ocr_prompt()
            
            # Make API request
            response_data = await self._make_api_request(image_b64, prompt)
            
            # Parse response to TextSpan objects
            text_spans = self._parse_mistral_response(response_data, page_image.page)
            
            return text_spans
            
        except Exception as e:
            print(f"Mistral OCR error for page {page_image.page}: {e}")
            return []
    
    async def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large
                max_size = 2048
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=95)
                img_bytes.seek(0)
                
                return base64.b64encode(img_bytes.read()).decode('utf-8')
                
        except Exception as e:
            raise RuntimeError(f"Failed to encode image: {e}")
    
    def _create_ocr_prompt(self) -> str:
        """Create OCR prompt for Mistral Vision API."""
        return """Extract ALL visible text from this electrical schematic image with precise coordinates.

Return results in JSON format:
{
  "text_detections": [
    {
      "text": "extracted_text",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.0-1.0
    }
  ]
}

Include component labels, values, pin numbers, and any other text."""
    
    async def _make_api_request(self, image_b64: str, prompt: str) -> Dict[str, Any]:
        """Make request to Mistral Vision API."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.api_url}/chat/completions",
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    response.raise_for_status()
                    
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
        
        raise RuntimeError(f"Mistral API request failed: {last_exception}")
    
    def _parse_mistral_response(self, response_data: Dict[str, Any], page: int) -> List[TextSpan]:
        """Parse Mistral API response into TextSpan objects."""
        text_spans = []
        
        try:
            if "choices" not in response_data or not response_data["choices"]:
                return []
            
            content = response_data["choices"][0]["message"]["content"]
            
            import json
            import re
            
            # Find JSON block
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                return self._parse_unstructured_response(content, page)
            
            data = json.loads(json_match.group())
            
            if "text_detections" in data:
                for detection in data["text_detections"]:
                    text = detection.get("text", "").strip()
                    bbox = detection.get("bbox", [])
                    confidence = detection.get("confidence", 0.5)
                    
                    if text and len(bbox) == 4:
                        text_span = TextSpan(
                            page=page,
                            bbox=[float(x) for x in bbox],
                            text=text,
                            rotation=0,
                            confidence=float(confidence),
                            engine=OcrEngine.MISTRAL
                        )
                        text_spans.append(text_span)
            
        except Exception as e:
            print(f"Failed to parse Mistral response: {e}")
            return self._parse_unstructured_response(
                response_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                page
            )
        
        return text_spans
    
    def _parse_unstructured_response(self, content: str, page: int) -> List[TextSpan]:
        """Fallback parser for unstructured response content."""
        text_spans = []
        
        import re
        patterns = [
            r'\b[RLCUQ]\d+\b',
            r'\b\d+[kKmMnNpPuU]?[ΩΩohm]*\b',
            r'\b\d+[.]?\d*[vV]\b',
            r'\b[A-Z]+\d*\b',
        ]
        
        y_offset = 100
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                text = match.group().strip()
                if text:
                    x_offset = 100 + len(text_spans) * 50
                    bbox = [x_offset, y_offset, x_offset + len(text) * 8, y_offset + 20]
                    
                    text_span = TextSpan(
                        page=page,
                        bbox=bbox,
                        text=text,
                        rotation=0,
                        confidence=0.7,
                        engine=OcrEngine.MISTRAL
                    )
                    text_spans.append(text_span)
        
        return text_spans
    
    @property
    def engine_name(self) -> str:
        """Return the name of this OCR engine."""
        return "mistral"
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()