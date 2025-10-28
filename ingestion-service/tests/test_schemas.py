"""
Test cases for core schemas.
"""
import pytest
from uuid import uuid4
from datetime import datetime

# Test basic imports (without external dependencies)
def test_schema_imports():
    """Test that core schemas can be imported."""
    try:
        from src.core.schemas import (
            TextSpan,
            Component,
            Net,
            Netlist,
            CreateIngestionRequest,
            IngestionStatus,
            OcrEngine,
            ComponentType,
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import schemas: {e}")


def test_enum_values():
    """Test enum definitions."""
    from src.core.schemas import IngestionStatus, OcrEngine, ComponentType
    
    # Test IngestionStatus enum
    assert IngestionStatus.QUEUED == "queued"
    assert IngestionStatus.PROCESSING == "processing"
    assert IngestionStatus.COMPLETED == "completed"
    assert IngestionStatus.FAILED == "failed"
    
    # Test OcrEngine enum
    assert OcrEngine.TESSERACT == "tesseract"
    assert OcrEngine.DEEPSEEK == "deepseek"
    assert OcrEngine.MISTRAL == "mistral"
    
    # Test ComponentType enum has expected values
    assert ComponentType.RESISTOR == "resistor"
    assert ComponentType.CAPACITOR == "capacitor"
    assert ComponentType.OPAMP == "opamp"


# Only run Pydantic tests if available
def test_pydantic_models():
    """Test Pydantic model validation (if available)."""
    try:
        from src.core.schemas import TextSpan, OcrEngine
        
        # Test valid TextSpan creation
        text_span = TextSpan(
            page=1,
            bbox=[0.0, 0.0, 100.0, 20.0],
            text="Test text",
            rotation=0,
            confidence=0.95,
            engine=OcrEngine.TESSERACT
        )
        
        assert text_span.page == 1
        assert text_span.text == "Test text"
        assert text_span.confidence == 0.95
        
    except ImportError:
        pytest.skip("Pydantic not available")


if __name__ == "__main__":
    test_schema_imports()
    test_enum_values()
    print("Basic schema tests passed!")