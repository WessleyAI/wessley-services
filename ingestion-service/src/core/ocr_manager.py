"""
OCR Manager for initializing and managing OCR providers.
"""
import os
from typing import List, Dict, Any, Optional

from ..ocr.base import OcrProvider
from ..ocr.tesseract import TesseractProvider
from ..ocr.deepseek import DeepSeekProvider
from ..ocr.mistral import MistralProvider
from ..ocr.fusion import OcrFusionEngine
from .schemas import OcrEngine, PageImage, TextSpan


class OcrManager:
    """
    Manages OCR providers and provides unified interface.
    """
    
    def __init__(self):
        self.providers: Dict[str, OcrProvider] = {}
        self.fusion_engine: Optional[OcrFusionEngine] = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available OCR providers."""
        # Initialize Tesseract (always available)
        try:
            self.providers["tesseract"] = TesseractProvider()
            print("Initialized Tesseract OCR provider")
        except Exception as e:
            print(f"Failed to initialize Tesseract: {e}")
        
        # Initialize DeepSeek (if API key available)
        if os.getenv("DEEPSEEK_API_KEY"):
            try:
                self.providers["deepseek"] = DeepSeekProvider()
                print("Initialized DeepSeek OCR provider")
            except Exception as e:
                print(f"Failed to initialize DeepSeek: {e}")
        
        # Initialize Mistral (if API key available)
        if os.getenv("MISTRAL_API_KEY"):
            try:
                self.providers["mistral"] = MistralProvider()
                print("Initialized Mistral OCR provider")
            except Exception as e:
                print(f"Failed to initialize Mistral: {e}")
        
        # Initialize fusion engine if multiple providers available
        if len(self.providers) > 1:
            try:
                available_providers = list(self.providers.values())
                self.fusion_engine = OcrFusionEngine(
                    providers=available_providers,
                    fusion_strategy="confidence_weighted"
                )
                print(f"Initialized OCR fusion engine with {len(available_providers)} providers")
            except Exception as e:
                print(f"Failed to initialize fusion engine: {e}")
    
    def get_provider(self, engine_name: str) -> Optional[OcrProvider]:
        """Get OCR provider by name."""
        return self.providers.get(engine_name)
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engine names."""
        engines = list(self.providers.keys())
        if self.fusion_engine:
            engines.append("fusion")
        return engines
    
    async def extract_text(
        self, 
        page_image: PageImage, 
        engines: List[str] = None
    ) -> List[TextSpan]:
        """
        Extract text using specified engines or fusion.
        
        Args:
            page_image: Page image to process
            engines: List of engine names to use (None = all available)
            
        Returns:
            List of TextSpan objects
        """
        if engines is None:
            engines = self.get_available_engines()
        
        # If fusion is requested and available
        if "fusion" in engines and self.fusion_engine:
            return await self.fusion_engine.extract_text_fused(page_image)
        
        # If single engine requested
        if len(engines) == 1 and engines[0] in self.providers:
            provider = self.providers[engines[0]]
            return await provider.extract_text(page_image)
        
        # Multiple engines - create temporary fusion
        if len(engines) > 1:
            selected_providers = []
            for engine_name in engines:
                if engine_name in self.providers:
                    selected_providers.append(self.providers[engine_name])
            
            if selected_providers:
                temp_fusion = OcrFusionEngine(
                    providers=selected_providers,
                    fusion_strategy="confidence_weighted"
                )
                return await temp_fusion.extract_text_fused(page_image)
        
        # Fallback to first available engine
        if self.providers:
            first_provider = next(iter(self.providers.values()))
            return await first_provider.extract_text(page_image)
        
        return []
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about available providers."""
        info = {
            "available_providers": list(self.providers.keys()),
            "fusion_available": self.fusion_engine is not None,
            "total_providers": len(self.providers)
        }
        
        # Get detailed info for each provider
        for name, provider in self.providers.items():
            provider_info = {
                "engine_name": provider.engine_name,
                "status": "available"
            }
            
            # Add provider-specific info
            if hasattr(provider, 'get_supported_languages'):
                try:
                    provider_info["supported_languages"] = provider.get_supported_languages()
                except:
                    pass
            
            info[name] = provider_info
        
        return info


# Global OCR manager instance
ocr_manager = OcrManager()