"""
DeepSeek OCR provider implementation using API.
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


class DeepSeekProvider(OcrProvider):
    """
    DeepSeek OCR provider using their vision API for text extraction.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = "deepseek-vl",
        max_retries: int = 3,
        timeout: int = 30
    ):
        """
        Initialize DeepSeek OCR provider.
        
        Args:
            api_key: DeepSeek API key (or from environment)
            api_url: DeepSeek API endpoint URL
            model: Model name to use for OCR
            max_retries: Maximum number of API retries
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = api_url or os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable.")
        
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
        Extract text from page image using DeepSeek Vision API.
        
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
            text_spans = self._parse_deepseek_response(response_data, page_image.page)
            
            return text_spans
            
        except Exception as e:
            print(f"DeepSeek OCR error for page {page_image.page}: {e}")
            return []
    
    async def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Load and potentially resize image for API limits
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (DeepSeek may have size limits)
                max_size = 2048
                if max(img.size) > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Save to bytes
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='JPEG', quality=95)
                img_bytes.seek(0)
                
                # Encode to base64
                img_b64 = base64.b64encode(img_bytes.read()).decode('utf-8')
                
                return img_b64
                
        except Exception as e:
            raise RuntimeError(f"Failed to encode image: {e}")
    
    def _create_ocr_prompt(self) -> str:
        """
        Create OCR prompt for DeepSeek Vision API.
        
        Returns:
            Prompt string optimized for electrical schematic OCR
        """
        return """Please extract ALL visible text from this electrical schematic image. 

For each piece of text you find, provide:
1. The exact text content
2. Bounding box coordinates as [x1, y1, x2, y2] in pixels
3. Your confidence in the detection (0.0 to 1.0)

Focus on:
- Component labels (R1, C2, U3, etc.)
- Component values (10k, 100nF, 74HC04, etc.)
- Net labels and pin numbers
- Any other visible text or numbers

Return the results in this JSON format:
{
  "text_detections": [
    {
      "text": "R1",
      "bbox": [100, 200, 130, 220],
      "confidence": 0.95
    },
    {
      "text": "10k",
      "bbox": [140, 200, 170, 220], 
      "confidence": 0.88
    }
  ]
}

Be precise with bounding box coordinates and include ALL visible text, even if small or unclear."""
    
    async def _make_api_request(self, image_b64: str, prompt: str) -> Dict[str, Any]:
        """
        Make request to DeepSeek Vision API.
        
        Args:
            image_b64: Base64 encoded image
            prompt: OCR prompt
            
        Returns:
            API response data
        """
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1  # Low temperature for consistent OCR results
        }
        
        # Make request with retries
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.api_url}/chat/completions",
                    json=payload
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit
                    wait_time = 2 ** attempt  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
                    
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break
        
        raise RuntimeError(f"DeepSeek API request failed after {self.max_retries} attempts: {last_exception}")
    
    def _parse_deepseek_response(self, response_data: Dict[str, Any], page: int) -> List[TextSpan]:
        """
        Parse DeepSeek API response into TextSpan objects.
        
        Args:
            response_data: API response data
            page: Page number
            
        Returns:
            List of TextSpan objects
        """
        text_spans = []
        
        try:
            # Extract content from response
            if "choices" not in response_data or not response_data["choices"]:
                return []
            
            content = response_data["choices"][0]["message"]["content"]
            
            # Try to parse JSON from content
            import json
            import re
            
            # Find JSON block in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if not json_match:
                # Fallback: try to extract text without structured format
                return self._parse_unstructured_response(content, page)
            
            json_str = json_match.group()
            data = json.loads(json_str)
            
            # Parse text detections
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
                            engine=OcrEngine.DEEPSEEK
                        )
                        text_spans.append(text_span)
            
        except Exception as e:
            print(f"Failed to parse DeepSeek response: {e}")
            # Fallback to unstructured parsing
            return self._parse_unstructured_response(
                response_data.get("choices", [{}])[0].get("message", {}).get("content", ""),
                page
            )
        
        return text_spans
    
    def _parse_unstructured_response(self, content: str, page: int) -> List[TextSpan]:
        """
        Fallback parser for unstructured response content.
        
        Args:
            content: Response content string
            page: Page number
            
        Returns:
            List of TextSpan objects (with estimated bboxes)
        """
        text_spans = []
        
        # Extract mentioned text using regex patterns
        import re
        
        # Look for component labels and values
        patterns = [
            r'\b[RLCUQ]\d+\b',  # Component labels (R1, L2, C3, U4, Q5)
            r'\b\d+[kKmMnNpPuU]?[ΩΩohm]*\b',  # Values (10k, 100nF, etc.)
            r'\b\d+[.]?\d*[vV]\b',  # Voltages (5V, 3.3V)
            r'\b[A-Z]+\d*\b',  # General labels
        ]
        
        y_offset = 100  # Estimated Y position
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                text = match.group().strip()
                if text:
                    # Estimate bounding box (since we don't have coordinates)
                    x_offset = 100 + len(text_spans) * 50
                    bbox = [x_offset, y_offset, x_offset + len(text) * 8, y_offset + 20]
                    
                    text_span = TextSpan(
                        page=page,
                        bbox=bbox,
                        text=text,
                        rotation=0,
                        confidence=0.7,  # Lower confidence for unstructured parsing
                        engine=OcrEngine.DEEPSEEK
                    )
                    text_spans.append(text_span)
        
        return text_spans
    
    @property
    def engine_name(self) -> str:
        """Return the name of this OCR engine."""
        return "deepseek"
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()