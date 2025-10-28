# Test Fixtures

This directory contains sample documents for testing the ingestion service.

## Structure

```
fixtures/
├── clean/          # Clean, high-quality documents
├── noisy/          # Scanned documents with noise/artifacts
├── handdrawn/      # Hand-drawn schematics
└── negative/       # Non-schematic documents (negative controls)
```

## Sample Documents

### Clean Documents (Vector-to-Raster)
- `clean_schematic_1.pdf` - Simple resistor-capacitor circuit
- `clean_schematic_2.pdf` - Op-amp amplifier circuit
- `clean_schematic_3.pdf` - Digital logic circuit
- `clean_schematic_4.pdf` - Power supply schematic
- `clean_schematic_5.pdf` - Motor control circuit

### Noisy Documents (Scanned)
- `noisy_schematic_1.pdf` - Skewed scan with shadows
- `noisy_schematic_2.pdf` - Low contrast document
- `noisy_schematic_3.pdf` - Rotated/tilted scan
- `noisy_schematic_4.pdf` - Partial occlusion
- `noisy_schematic_5.pdf` - Multiple pages with different orientations

### Hand-drawn Documents
- `handdrawn_1.pdf` - Hand-sketched circuit
- `handdrawn_2.pdf` - Whiteboard photo

### Negative Controls
- `text_document.pdf` - Plain text document
- `table_data.pdf` - Tabular data
- `photo.jpg` - Non-technical photograph

## Adding New Fixtures

When adding new test documents:

1. **Name consistently**: Use descriptive names with category prefix
2. **Include metadata**: Add corresponding `.json` files with:
   - Expected component count
   - Known text regions
   - Ground truth annotations
3. **Document purpose**: Update this README with fixture descriptions

## Ground Truth Format

For benchmarking, include ground truth files:

```json
{
  "filename": "clean_schematic_1.pdf",
  "pages": 1,
  "components": [
    {
      "id": "R1",
      "type": "resistor",
      "value": "10k",
      "bbox": [100, 200, 150, 220],
      "page": 1
    }
  ],
  "nets": [
    {
      "name": "VCC",
      "connections": [{"component": "R1", "pin": "1"}]
    }
  ],
  "text_regions": [
    {
      "text": "R1",
      "bbox": [105, 180, 115, 195],
      "page": 1
    }
  ]
}
```

## Usage in Tests

```python
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

def test_clean_schematic():
    fixture_path = FIXTURES_DIR / "clean" / "clean_schematic_1.pdf"
    # Process fixture...
```