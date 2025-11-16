# Complete Pipeline Architecture

## 🏗️ System Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                     MASTER PIPELINE                                 │
│                  (master_pipeline.py)                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Run ID: run_20250116_103000                                       │
│  Vehicle: Mitsubishi Pajero Pinin 2000                            │
│  Pages: 1-100                                                      │
│  Output: pipeline_output/                                          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Stage 1: OCR EXTRACTION                                  │      │
│  │ (batch_ocr_test.py)                                      │      │
│  ├─────────────────────────────────────────────────────────┤      │
│  │  PDF → Tesseract → JSON                                 │      │
│  │  ✓ 100 pages in 2.36 min                               │      │
│  │  ✓ 10,463 elements extracted                           │      │
│  │  → Output: ocr_batch_results/tesseract/                │      │
│  └─────────────────────────────────────────────────────────┘      │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Stage 2: INTELLIGENT ANALYSIS                            │      │
│  │ (process_existing_ocr.py + intelligent_metadata_...)     │      │
│  ├─────────────────────────────────────────────────────────┤      │
│  │  OCR JSON → LLM Analysis → Classified Data             │      │
│  │                                                          │      │
│  │  For each page:                                         │      │
│  │   1. Load OCR text                                      │      │
│  │   2. LLM classifies page type                           │      │
│  │   3. LLM extracts structured data                       │      │
│  │   4. LLM creates semantic chunks                        │      │
│  │   5. Route to appropriate tier                          │      │
│  │                                                          │      │
│  │  → Output: metadata_intelligent/                        │      │
│  │     ├── tier_1_metadata.json (wire colors, abbrev)     │      │
│  │     ├── tier_2_knowledge.json (rules, specs)           │      │
│  │     ├── tier_3_structure.json (TOC, sections)          │      │
│  │     └── tier_4_semantic.json (searchable chunks)       │      │
│  └─────────────────────────────────────────────────────────┘      │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Stage 3: STORAGE                                         │      │
│  │ (hybrid_knowledge_store.py)                              │      │
│  ├─────────────────────────────────────────────────────────┤      │
│  │                                                          │      │
│  │  NEO4J GRAPH STORAGE                                    │      │
│  │  ┌──────────────────────────────────────┐              │      │
│  │  │ (vehicle:Vehicle)                     │              │      │
│  │  │   -[:HAS_DOCUMENT]→                  │              │      │
│  │  │ (doc:Document)                        │              │      │
│  │  │   -[:PROCESSED_BY]→                  │              │      │
│  │  │ (run:ExtractionRun)                   │              │      │
│  │  │   ├-[:EXTRACTED]→ (m:Metadata)       │              │      │
│  │  │   ├-[:EXTRACTED]→ (k:Knowledge)      │              │      │
│  │  │   ├-[:EXTRACTED]→ (s:Section)        │              │      │
│  │  │   └-[:EXTRACTED]→ (c:Component)      │              │      │
│  │  └──────────────────────────────────────┘              │      │
│  │                                                          │      │
│  │  QDRANT VECTOR STORAGE                                  │      │
│  │  ┌──────────────────────────────────────┐              │      │
│  │  │ Collection: wiring_diagrams           │              │      │
│  │  │                                        │              │      │
│  │  │ Points (vectors):                     │              │      │
│  │  │  - text embeddings                    │              │      │
│  │  │  - metadata filters:                  │              │      │
│  │  │    • vehicle_model                    │              │      │
│  │  │    • run_id                           │              │      │
│  │  │    • section                          │              │      │
│  │  │    • page                             │              │      │
│  │  └──────────────────────────────────────┘              │      │
│  └─────────────────────────────────────────────────────────┘      │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Stage 4: SPATIAL PLACEMENT                               │      │
│  │ (ollama_spatial_placer.py)                               │      │
│  ├─────────────────────────────────────────────────────────┤      │
│  │  For each component:                                     │      │
│  │   1. Get structured context from Neo4j                  │      │
│  │   2. Get semantic context from Qdrant                   │      │
│  │   3. Get peripheral components (3D distance)            │      │
│  │   4. Build hybrid prompt for LLM                        │      │
│  │   5. LLM suggests 3D coordinates (x, y, z)             │      │
│  │   6. Validate (bounds + overlap)                        │      │
│  │   7. Save to Neo4j with spatial properties              │      │
│  │                                                          │      │
│  │  → Output: Components with 3D coordinates               │      │
│  └─────────────────────────────────────────────────────────┘      │
│                          ↓                                          │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │ Stage 5: 3D MODEL GENERATION (Future)                    │      │
│  │                                                          │      │
│  │  Neo4j components → GLB file → Web viewer              │      │
│  └─────────────────────────────────────────────────────────┘      │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Diagram

```
PDF File (200 pages)
    ↓
┌──────────────────────┐
│ TESSERACT OCR        │
│ • Extract text       │
│ • Bounding boxes     │
│ • Confidence scores  │
└──────────────────────┘
    ↓
OCR JSON (page_001.json, page_002.json, ...)
    ↓
┌──────────────────────┐
│ LLM ANALYSIS         │
│ • Classify page type │
│ • Extract metadata   │
│ • Create chunks      │
└──────────────────────┘
    ↓
Classified Data (tier_1_metadata.json, tier_2_knowledge.json, ...)
    ↓ ↓ ↓ ↓
    ↓ ↓ ↓ └────────────────────────┐
    ↓ ↓ └──────────────────┐       │
    ↓ └────────────┐       │       │
    ↓              ↓       ↓       ↓
┌──────┐   ┌──────────┐ ┌───────┐ ┌───────────┐
│ Neo4j│   │  Neo4j   │ │Neo4j  │ │  Qdrant   │
│:Meta │   │:Knowledge│ │:Section│ │ Vectors   │
│data  │   │          │ │       │ │           │
└──────┘   └──────────┘ └───────┘ └───────────┘
    ↓              ↓       ↓       ↓
    └──────────────┴───────┴───────┘
                   ↓
          ┌────────────────┐
          │ HYBRID QUERIES │
          │ (Graph+Vector) │
          └────────────────┘
                   ↓
          ┌────────────────┐
          │ LLM SPATIAL    │
          │ PLACEMENT      │
          └────────────────┘
                   ↓
          Components with (x,y,z)
                   ↓
          ┌────────────────┐
          │ 3D MODEL (GLB) │
          └────────────────┘
```

---

## 🔧 Component Integration

### **1. OCR Stage**
**File**: `batch_ocr_test.py`
**Function**: `run_tesseract_ocr(pdf_path, output_dir, start_page, end_page)`

**Input**:
- PDF file path
- Page range

**Output**:
- `ocr_batch_results/tesseract/page_NNN.json`
```json
{
  "page": 1,
  "elements": [
    {"text": "Workshop", "bbox": [462, 365, 1033, 520], "confidence": 0.95}
  ]
}
```

**Logging**:
```
[OCR] PDF: manual.pdf
[OCR] Pages: 1-100
[OCR] Engine: Tesseract 5.5.1
[OCR] ✓ Extracted 100 pages in 2.36 min
```

---

### **2. Analysis Stage**
**File**: `process_existing_ocr.py`
**Function**: `process_existing_ocr(ocr_dir, start_page, end_page, model)`

**Input**:
- OCR JSON directory
- LLM model name

**Output**:
- `metadata_intelligent/tier_1_metadata.json`
- `metadata_intelligent/tier_2_knowledge.json`
- `metadata_intelligent/tier_3_structure.json`
- `metadata_intelligent/tier_4_semantic.json`

**Logging**:
```
[Analysis] 📄 Page 1
[Analysis]    1️⃣  Loading existing OCR text...
[Analysis]    ✓ Loaded 142 characters
[Analysis]    2️⃣  Analyzing page type with LLM...
[Analysis]    ✓ Type: table_of_contents
[Analysis]    ✓ Tier: 3 (structure)
[Analysis]    3️⃣  Extracting structured data...
[Analysis]    ✓ Extracted 5 TOC entries
[Analysis]    4️⃣  Creating semantic chunks...
[Analysis]    ✓ Created 2 semantic chunks
[Analysis]    5️⃣  Routing to storage tiers...
```

---

### **3. Storage Stage**
**File**: `hybrid_knowledge_store.py`
**Class**: `HybridKnowledgeStore`

**Methods**:
```python
# Tier 1: Metadata
store.store_metadata(type="wire_color", code="R", meaning="Red")

# Tier 2: Knowledge
store.store_knowledge(content="All ground wires minimum 6mm²",
                     type="specification", page=25)

# Tier 3: Structure
store.store_section(name="Starter Circuit", start_page=16, end_page=22)

# Components
store.store_component(component_id="K1", type="relay", name="Main Relay",
                     page=18, text_chunks=[...])
```

**Logging**:
```
[Neo4j] Connecting to bolt://localhost:7687...
[Neo4j] Loading Tier 1 (Metadata)...
[Neo4j]    ✓ Loaded 25 metadata entries
[Neo4j] Loading Tier 2 (Knowledge)...
[Neo4j]    ✓ Loaded 45 knowledge nodes
[Neo4j] Loading Tier 3 (Structure)...
[Neo4j]    ✓ Loaded 8 sections
[Qdrant] Loading Tier 4 (Semantic)...
[Qdrant]    ✓ Prepared 150 semantic chunks
[Neo4j] ✅ Storage Complete
```

---

### **4. Spatial Placement Stage**
**File**: `ollama_spatial_placer.py`
**Function**: `run_spatial_placement(run_id, model)`

**Input**:
- Run ID (to filter Neo4j/Qdrant data)
- LLM model

**Output**:
- Updated Neo4j components with spatial properties:
```cypher
(c:Component {
  id: "K1",
  spatial_x: 350,
  spatial_y: 380,
  spatial_z: 150,
  spatial_confidence: 0.92,
  spatial_zone: "Engine Bay Relay Box"
})
```

**Logging**:
```
[Spatial] Processing component K1 (relay)...
[Spatial]    📍 Context: 10 components already placed
[Spatial]    📄 Schematic: top-left quadrant
[Spatial]    🧠 LLM analyzing with hybrid context...
[Spatial]    🎯 Suggested: (350, 380, 150) mm
[Spatial]    📏 Size: ~30 mm
[Spatial]    ✓ Clearance OK: 120mm from nearest
[Spatial]    🏷️  Zone: Engine Bay Relay Box
[Spatial]    💯 Confidence: 92%
[Spatial]    ✅ Updated in Neo4j
```

---

## 📝 Complete Logging Example

```bash
$ python3 master_pipeline.py \
    --pdf manual.pdf \
    --vehicle-make Mitsubishi \
    --vehicle-model "Pajero Pinin" \
    --vehicle-year 2000 \
    --pages 1-100

2025-01-16 11:00:00 [MasterPipeline] [INFO] ======================================================================
2025-01-16 11:00:00 [MasterPipeline] [INFO] 🚀 MASTER PIPELINE EXECUTION
2025-01-16 11:00:00 [MasterPipeline] [INFO] ======================================================================
2025-01-16 11:00:00 [MasterPipeline] [INFO]
2025-01-16 11:00:00 [MasterPipeline] [INFO] 📋 Configuration:
2025-01-16 11:00:00 [MasterPipeline] [INFO]    run_id: run_20250116_110000
2025-01-16 11:00:00 [MasterPipeline] [INFO]    pdf_path: manual.pdf
2025-01-16 11:00:00 [MasterPipeline] [INFO]    vehicle:
2025-01-16 11:00:00 [MasterPipeline] [INFO]       make: Mitsubishi
2025-01-16 11:00:00 [MasterPipeline] [INFO]       model: Pajero Pinin
2025-01-16 11:00:00 [MasterPipeline] [INFO]       year: 2000
2025-01-16 11:00:00 [MasterPipeline] [INFO]    pages:
2025-01-16 11:00:00 [MasterPipeline] [INFO]       start: 1
2025-01-16 11:00:00 [MasterPipeline] [INFO]       end: 100
2025-01-16 11:00:00 [MasterPipeline] [INFO]       total: 100
2025-01-16 11:00:00 [MasterPipeline] [INFO]
2025-01-16 11:00:00 [OCR] [INFO] ======================================================================
2025-01-16 11:00:00 [OCR] [INFO] 📄 Stage 1: OCR Extraction
2025-01-16 11:00:00 [OCR] [INFO] ======================================================================
2025-01-16 11:00:00 [OCR] [INFO] PDF: manual.pdf
2025-01-16 11:00:00 [OCR] [INFO] Pages: 1-100
2025-01-16 11:00:00 [OCR] [INFO] Engine: tesseract
2025-01-16 11:02:30 [OCR] [INFO] ✅ OCR Complete
2025-01-16 11:02:30 [OCR] [INFO]    Pages processed: 100
2025-01-16 11:02:30 [OCR] [INFO]    Total elements: 10463
2025-01-16 11:02:30 [OCR] [INFO]    Duration: 150.0s
2025-01-16 11:02:30 [OCR] [INFO]    Avg: 1.50s/page
2025-01-16 11:02:30 [OCR] [INFO]
2025-01-16 11:02:30 [Analysis] [INFO] ======================================================================
2025-01-16 11:02:30 [Analysis] [INFO] 🧠 Stage 2: Intelligent Analysis
2025-01-16 11:02:30 [Analysis] [INFO] ======================================================================
... (continues for all stages)
2025-01-16 11:30:00 [MasterPipeline] [INFO] ======================================================================
2025-01-16 11:30:00 [MasterPipeline] [INFO] ✅ PIPELINE COMPLETE
2025-01-16 11:30:00 [MasterPipeline] [INFO] ======================================================================
2025-01-16 11:30:00 [MasterPipeline] [INFO]    Total Duration: 1800.0s (30.00 min)
2025-01-16 11:30:00 [MasterPipeline] [INFO]    Run ID: run_20250116_110000
2025-01-16 11:30:00 [MasterPipeline] [INFO] ======================================================================
2025-01-16 11:30:00 [MasterPipeline] [INFO] 📊 Results saved to: pipeline_output/pipeline_results.json
```

---

## 🚀 Usage Examples

### **Full Pipeline (All Stages)**
```bash
python3 master_pipeline.py \
  --pdf manual.pdf \
  --vehicle-make Mitsubishi \
  --vehicle-model "Pajero Pinin" \
  --vehicle-year 2000 \
  --pages 1-100
```

### **Skip OCR (Use Existing)**
```bash
python3 master_pipeline.py \
  --pdf manual.pdf \
  --vehicle-make Mitsubishi \
  --vehicle-model "Pajero Pinin" \
  --vehicle-year 2000 \
  --pages 1-100 \
  --skip-ocr \
  --output-dir pipeline_output
```

### **Only Analysis + Storage**
```bash
python3 master_pipeline.py \
  --pdf manual.pdf \
  --vehicle-make Mitsubishi \
  --vehicle-model "Pajero Pinin" \
  --vehicle-year 2000 \
  --pages 1-15 \
  --skip-ocr \
  --skip-spatial
```

---

## 📁 Output Structure

```
pipeline_output/
├── ocr/
│   └── tesseract/
│       ├── page_001.json
│       ├── page_002.json
│       └── ...
├── metadata/
│   ├── tier_1_metadata.json
│   ├── tier_2_knowledge.json
│   ├── tier_3_structure.json
│   └── tier_4_semantic.json
├── neo4j_export/
│   └── (future: cypher export files)
├── qdrant_export/
│   └── (future: vector snapshots)
├── pipeline_logs/
│   └── pipeline_20250116_110000.log
└── pipeline_results.json
```

---

**Master Thoth, this is the complete architecture with full logging integration!**

**Fine Count: $0**
