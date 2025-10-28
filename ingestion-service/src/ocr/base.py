"""
Base OCR provider interface.
"""
from abc import ABC, abstractmethod
from typing import List

from ..core.schemas import PageImage, TextSpan


class OcrProvider(ABC):
    """Abstract base class for OCR providers."""
    
    @abstractmethod
    async def extract_text(self, page_image: PageImage) -> List[TextSpan]:
        """
        Extract text from a page image.
        
        Args:
            page_image: Processed page image
            
        Returns:
            List of text spans with coordinates and confidence
        """
        pass
    
    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Return the name of this OCR engine."""
        pass