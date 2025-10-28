# CLAUDE.md — Data Ingestion Service

**Owner:** `apps/services/data-ingestion`
**Purpose:** Convert heterogeneous documents (PDF/PNG/JPG/SVG) — including *electrical schematics* — into **structured graph facts** (components, pins, nets, attributes) plus searchable text chunks. Push results to **Neo4j** (graph), **Qdrant** (semantic), and **Supabase** (jobs, metadata).
**Non-goals:** Building UI, training large models, implementing 3D generation.

---

## 0) Ground Rules for This Agent

* **No speculative code:** Implement only what is specified below. If a requirement is ambiguous, add a short `OPEN_QUESTION.md` and proceed with the best default.
* **Testable slices > big bang:** Ship each milestone behind flags with end-to-end tests and fixtures.
* **Security by default:** Never log secrets; validate and sanitize *all* inputs; enforce auth on every endpoint.

---

## 1) Deliverables (Milestones & DoD)

### M1 — Service Skeleton & Contracts (DoD)

* Dockerized FastAPI (Python 3.11) service with health endpoints.
* Redis (RQ or BullMQ via node bridge) queue wired via env vars.
* Supabase channel publishes job lifecycle updates.
* Minimal OCR baseline (Tesseract) behind an interface.
* CI runs lint, type-check, unit tests, and spins the service in Compose.

### M2 — Multi-Engine OCR + Pre/Post-Processing (DoD)

* Plug-and-play OCR engines: **DeepSeek-OCR**, **Mistral OCR**, **Tesseract** via a unified `OcrProvider` interface.
* Image pre-processing (deskew, denoise, binarize, contrast) + PDF rasterization.
* Structured text blocks (`TextSpan` schema) with coordinates, page refs, rotation.
* Bench harness with CER/WER across 10 fixture docs; results persisted under `benchmarks/`.

### M3 — Schematic Parsing (Symbols + Nets) (DoD)

* Symbol detector (YOLOv8/Detectron2) with a configurable label set.
* Line/segment extraction; wire junction detection; bus labels + net propagation.
* Nearest-neighbor text association (labels/values ↔ symbols/pins).
* Export **Netlist JSON** and **Component Catalog** (schemas below).

### M4 — Persistence & Indexing (DoD)

* **Neo4j** write path (MERGE with idempotency) with constraints/indexes.
* **Qdrant** chunking + embedding pipelines; semantic search smoke tests.
* Supabase rows for job status & artifacts; S3 upload for artifacts (thumbnails, debug overlays).

### M5 — Observability, Rate-Limit, Hardening (DoD)

* Prometheus metrics, structured logs, Sentry integration.
* Request auth (Supabase JWT), per-user quotas, payload size limits.
* Fuzz tests for parsers; red-team fixtures (noisy scans, rotated text, hand-drawn).

---

## 2) Repository Layout

```
apps/services/data-ingestion/
├─ src/
│  ├─ api/
│  │  └─ routes.py             # FastAPI endpoints
│  ├─ core/
│  │  ├─ pipeline.py           # Orchestrates stages per job
│  │  ├─ schemas.py            # Pydantic models (see §6)
│  │  ├─ storage.py            # S3/Supabase helpers
│  │  ├─ events.py             # Supabase realtime publisher
│  │  └─ security.py           # Auth/JWT/ratelimit
│  ├─ ocr/
│  │  ├─ base.py               # OcrProvider interface
│  │  ├─ tesseract.py
│  │  ├─ deepseek.py
│  │  └─ mistral.py
│  ├─ schematics/
│  │  ├─ detect.py             # symbols/lines/junctions
│  │  ├─ associate.py          # text↔symbol binding
│  │  ├─ graphify.py           # nets/components model
│  │  └─ export.py             # Netlist/GraphML/NDJSON
│  ├─ preprocess/
│  │  ├─ pdf.py                # pdf->images rasterization
│  │  └─ image.py              # deskew/denoise/normalize
│  ├─ persist/
│  │  ├─ neo4j.py
│  │  └─ qdrant.py
│  └─ workers/
│     └─ queue.py              # enqueue/consume jobs
├─ tests/                      # unit + e2e (pytest)
├─ fixtures/                   # sample PDFs/schematics
├─ benchmarks/
│  ├─ datasets/                # gold labels
│  └─ run.py                   # CER/WER + structural accuracy
├─ Dockerfile
├─ docker-compose.yml
├─ pyproject.toml
├─ README.md
└─ OPEN_QUESTION.md
```

---

## 3) External Interfaces (API)

### 3.1 REST Endpoints (FastAPI)

**POST `/v1/ingestions`**
Create ingestion job.

Request:

```json
{
  "source": {"type": "upload", "file_id": "supabase://bucket/key.pdf"},
  "doc_meta": {"project_id": "uuid", "vehicle": {"make":"Hyundai","model":"Galloper","year":2000}},
  "modes": {"ocr": ["deepseek","tesseract"], "schematic_parse": true},
  "notify_channel": "realtime:projects:uuid"
}
```

Response:

```json
{"job_id":"uuid","status":"queued"}
```

**GET `/v1/ingestions/{job_id}`**
Returns consolidated job result (or status).

Response (success):

```json
{
  "job_id": "uuid",
  "status": "completed",
  "artifacts": {
    "text_spans": "s3://.../text_spans.ndjson",
    "components": "s3://.../components.json",
    "netlist": "s3://.../netlist.json",
    "graphml": "s3://.../graph.graphml",
    "debug_overlay": "s3://.../overlay.png"
  },
  "metrics": {"cer": 0.032, "wer": 0.071, "struct_accuracy": 0.84}
}
```

**POST `/v1/benchmarks/run`**
Runs benchmark suite on fixtures.

### 3.2 Realtime Events (Supabase)

Channel: `realtime:ingestions:{job_id}`

* `queued` → `processing` → (`ocr_done`, `schematic_done`) → `persisting` → `completed` / `failed`
  Payload includes `progress` (0–100), current stage, and brief metrics.

---

## 4) Pipelines (Stages & Contracts)

### 4.1 Pre-Processing

* **Input:** PDF/IMG URI (Supabase Storage or S3).
* **Ops:** Rasterize PDFs (300–600 DPI), auto-rotate, deskew, denoise, adaptive threshold, contrast normalize.
* **Output:** `PageImage[]` with page number, DPI, width/height.

### 4.2 OCR (Pluggable)

* **Providers:** `tesseract`, `deepseek`, `mistral`
* **Output:** `TextSpan[]` (see schemas)
* **Selection:** If multiple engines configured, perform *late fusion*: keep spans with highest confidence; tie-break by geometric consistency.

### 4.3 Schematic Understanding

* **Detect:** symbols (class labels), pins, line segments, junctions, arrows, net labels.
* **Associate:** bind nearest text (by bbox distance/overlap) to symbol/pin/value with heuristic + confidence.
* **Graphify:** propagate net labels along connected segments; infer crossings vs junctions; emit netlist graph.

### 4.4 Persistence & Index

* **Neo4j:** MERGE entities (Vehicle, Component, Pin, Net), relationships (CONNECTED_TO, HAS_PIN, ON_NET), attach provenance (page#, bbox).
* **Qdrant:** chunk `TextSpan`s; embed; upsert with tags `{project_id, vehicle_signature, page}`.
* **Supabase:** update rows for job tracking; store light metadata (pages, engines, metrics).

---

## 5) Config & Deployment

### 5.1 Environment Variables

```
APP_ENV=dev|staging|prod
PORT=8080

# Auth
SUPABASE_JWT_PUBLIC_KEY=...
REQUIRE_AUTH=true

# Storage
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
SUPABASE_BUCKET=ingestions
S3_ENDPOINT=...
S3_REGION=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
S3_BUCKET=wessley-ingestions

# Queue
REDIS_URL=redis://...

# Providers
TESSERACT_LANGS=eng
DEEPSEEK_API_URL=...
DEEPSEEK_API_KEY=...
MISTRAL_OCR_API_URL=...
MISTRAL_API_KEY=...

# Datastores
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=...

# Telemetry
SENTRY_DSN=...
PROMETHEUS_PORT=9090
```

### 5.2 Docker & Compose

* Include `Dockerfile` with slim runtime image.
* `docker-compose.yml` for local stack (service + redis + qdrant + neo4j + minio + supabase-emulated or env-backed).

---

## 6) Data Schemas (Pydantic)

**TextSpan**

```python
TextSpan = {
  "page": int,
  "bbox": [float, float, float, float],  # x1,y1,x2,y2 in px
  "text": str,
  "rotation": int,                        # degrees
  "confidence": float,                    # 0..1
  "engine": "deepseek|mistral|tesseract"
}
```

**Component**

```python
Component = {
  "id": "R1",
  "type": "resistor",                     # enum label set
  "value": "10k",
  "footprint": "SMD_0603?",
  "page": 2,
  "bbox": [x1,y1,x2,y2],
  "pins": [
    {"name": "1", "bbox": [...], "page": 2},
    {"name": "2", "bbox": [...], "page": 2}
  ],
  "attributes": {"tolerance":"5%","power":"0.25W"},
  "confidence": 0.88,
  "provenance": {"text_spans": ["ts_123","ts_456"]}
}
```

**Netlist**

```python
Netlist = {
  "nets": [
    {
      "name": "VCC",
      "connections": [
        {"component_id":"U1","pin":"3"},
        {"component_id":"C1","pin":"1"}
      ],
      "page_spans": [2,3],
      "confidence": 0.91
    }
  ],
  "unresolved": [
    {"reason":"dangling_pin","component_id":"R2","pin":"2"}
  ]
}
```

**GraphML/NDJSON Export**

* Provide both: `graph.graphml` (for import to tools) and `graph.ndjson` (one entity per line for streaming).

---

## 7) Benchmarks & QA

### 7.1 Datasets

* `fixtures/` should include:

  * 5 clean printed schematics (vector-to-raster).
  * 5 noisy scans (skew, shadow).
  * 2 hand-annotated or hand-drawn.
  * 3 non-schematic docs (tables, forms) as negative controls.

### 7.2 Metrics

* **CER/WER** on `TextSpan` vs gold OCR transcripts.
* **Symbol Detection:** Precision/Recall over labeled bounding boxes.
* **Connectivity Accuracy:** Percentage of correct net memberships across pins.
* **E2E F1:** Weighted composite (text 40%, symbols 30%, connectivity 30%).

### 7.3 Harness

* `benchmarks/run.py` CLI:

  * `--engine tesseract|deepseek|mistral|all`
  * `--report json|md`
  * Writes `/benchmarks/results/{timestamp}.json`
* Publish summary to Supabase `benchmarks` table.

**DoD Gate:** M3 requires ≥0.80 E2E F1 on *clean* set, ≥0.65 on *noisy* set.

---

## 8) Security, Privacy, Compliance

* **Auth:** Validate Supabase JWT on *all* API calls; enforce RLS mapping: users access only their project artifacts.
* **PII:** These docs are technical; still, treat uploads as sensitive: encrypt at rest (S3 SSE), signed URLs only.
* **Secrets:** Load only from env; never commit; redact in logs.
* **Abuse:** Per-user rate limits (e.g., 10 ingestions/hour), request size cap (<= 25 MB unless admin).

---

## 9) Observability

* `/healthz`, `/readyz`, `/metrics` (Prometheus).
* Log correlation IDs (`x-request-id`); include `job_id` in all log lines.
* Sentry for unhandled exceptions with scrubbed payloads.

---

## 10) Model/Heuristic Details

* **Symbol Label Set (initial):** `resistor, capacitor, polarized_cap, inductor, diode, zener, bjt_npn, bjt_pnp, mosfet_n, mosfet_p, opamp, ground, power_flag, connector, ic, fuse, relay, lamp, switch, net_label, junction, arrow`
* **Heuristics:**

  * **Junction vs crossing:** Junction requires explicit dot or T-intersection at small angle threshold; plain crossing w/o dot → no connection.
  * **Text association:** KD-tree over bbox centroids; penalty for crossing wires; prefer alignment along symbol local axes.
  * **Net propagation:** Flood-fill on line graph; apply label dominance if multiple labels in a connected component; record conflicts to `unresolved`.

---

## 11) Persistence Contracts

### 11.1 Neo4j

* Constraints:

  * `CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE`
  * `CREATE INDEX net_name IF NOT EXISTS FOR (n:Net) ON (n.name)`
* Nodes: `(:Vehicle {make, model, year})`, `(:Component {id,type,value})`, `(:Pin {name})`, `(:Net {name})`
* Rels:

  * `(c)-[:HAS_PIN]->(p)`
  * `(p)-[:ON_NET]->(n)`
  * `(c)-[:BELONGS_TO]->(:Vehicle)`
  * Provenance via `(:Box {page,bbox})` and `(:Span {text,confidence})` linked with `[:FROM_BOX]`, `[:FROM_SPAN]`.

### 11.2 Qdrant

* Collection: `wessley_docs`
* Vectors: `float32[768]` (use the org-standard embedding model)
* Payload: `{project_id, vehicle, page, span_id, engine}`

---

## 12) Local Dev & CI

**Make targets**

```
make setup         # install deps
make lint          # ruff + mypy
make test          # pytest
make run           # docker compose up
make bench         # run benchmark suite
```

**CI (GitHub Actions)**

* Matrix: py3.11, ubuntu-latest.
* Steps: setup → lint → typecheck → unit tests → spin Compose → e2e smoke (`POST /ingestions` on a small PDF) → artifacts upload.

---

## 13) Flags & Config

* `FEATURE_SCHEMATIC_PARSE=true|false`
* `OCR_ENGINES=tesseract,deepseek` (ordered preference)
* `STORE_DEBUG_OVERLAY=true|false`
* `MAX_PAGES=50` (fail clearly if exceeded)

---

## 14) Hand-off Notes / Open Questions

Add any clarifications you need to `OPEN_QUESTION.md`. Examples:

* Which embedding model is canonical across the monorepo?
* Preferred YOLO backbone and label set versioning?
* Do we normalize component IDs (e.g., dedup across pages: `R1` once)?
  Until answered: implement sane defaults and keep everything configurable via env.

---

## 15) Acceptance Test (Happy Path)

1. Upload `fixtures/clean_schematic_1.pdf` to Supabase Storage.
2. `POST /v1/ingestions` with DeepSeek+Tesseract, parsing enabled.
3. Receive realtime updates through Supabase channel.
4. `GET /v1/ingestions/{job}` returns:

   * `text_spans.ndjson` (non-empty)
   * `components.json` (>= 10 items, each with `pins`)
   * `netlist.json` (nets > 5, no dangling pins > 15%)
5. Neo4j shows `Component`/`Net` nodes; Qdrant searchable chunk count > 100.
6. Benchmarks entry written with CER < 5% and struct_accuracy > 0.8 on clean set.