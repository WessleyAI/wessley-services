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

# PATCH — Integrated Semantic Search & Learning Loop (Automotive-Electronics Focus)

Scope: Keep everything inside the Data Ingestion service (no new microservices). Add a robust semantic layer and a continual-learning loop that improves symbol detection, OCR fusion, and schematic understanding as more documents are processed.

## 0) Goals

Build a growing domain model for automotive electronics (components, pins, nets, harnesses, locations, failure modes).

Better answers over time: every newly ingested manual/schematic boosts retrieval, parsing accuracy, and reasoning.

Zero downtime iteration: new models/heuristics are trained offline and atomically promoted via a registry version.

## 1) Domain Ontology & Object Model (v0.1)

Define a minimal automotive electronics ontology to normalize everything we extract:

Entities

Vehicle { make, model, year, market }

System { name, e.g., “Starting”, “Lighting”, “Fuel Pump” }

Component { id/ref, type(enum), value, rating, location_hint }

Pin { name, index, electrical_role(enum: power, ground, signal, can_h, can_l) }

Net { name, voltage_hint, bus(enum: CAN, LIN, K-line) }

Connector { code, cavity_count, location_hint }

Ground { code, chassis_location }

Fuse/Relay { rating, slot, panel }

Wire { gauge, color_code, route_hint }

Relations

(Component)-[:HAS_PIN]->(Pin)

(Pin)-[:ON_NET]->(Net)

(Component)-[:BELONGS_TO]->(System)

(Component)-[:IN_VEHICLE]->(Vehicle)

(Connector)-[:HOSTS_PIN]->(Pin)

(Ground)-[:RETURNS_NET]->(Net)

(Fuse|Relay)-[:PROTECTS]->(Net|Component)

Keep ontology extensible; don’t overfit early.

## 2) Semantic Layer (Embedded)
### 2.1 Multi-Vector Indexing Strategy

Dense embeddings: two fields per chunk

semantic_text (OCR spans merged by layout; 256–512 tokens)

symbolic_context (symbol types, nets, nearby components serialized as text)

Sparse (lexical) index: postgres tsvector (BM25-like) for exact terms: “J/B No.1”, “IG1”, “EFI MAIN”, “GND-108”.

Hybrid search: topk_dense ∪ topk_sparse → cross-encoder re-ranker (tiny).

### 2.2 Chunking Rules

Prefer layout blocks (same font/size, proximity) over fixed tokens.

Attach structured anchors to each chunk: {vehicle_signature, page, bbox, component_ids[], net_names[], system}.

### 2.3 Payload (Qdrant/pgvector)
{
  "id": "span_8f…",
  "project_id": "uuid",
  "vehicle": {"make":"Hyundai","model":"Galloper","year":2000},
  "page": 3,
  "bbox": [x1,y1,x2,y2],
  "text": "Starter relay circuit operation...",
  "symbolic_context": "relay K1, net IG1, fuse F10 30A, ground G102",
  "component_ids": ["K1","F10"],
  "net_names": ["IG1","BATT"],
  "embedding_sem": [ ... ],
  "embedding_sym": [ ... ],
  "ts_lex": "starter & relay & ig1 & f10"
}

### 2.4 API

GET /v1/search?q=...&project_id=...&vehicle=...&filters=...
Returns merged hits: spans + components + nets (w/ reasons).

## 3) Learning Loop (Continual)
### 3.1 Data Sources (growing corpora)

Parsed schematics (images + derived labels)

Workshop manuals (PDF → OCR chunks)

Fuse/relay charts (tables → normalized)

Harness layout pages (diagrams → location hints)

User corrections / adjudications from UI (later)

### 3.2 Signals Collected Automatically

OCR late-fusion disagreements

Symbol detector false positives/negatives (via geometry + net consistency checks)

Net propagation conflicts (label collisions, dangling pins)

Search click-through / dwell time (when integrated with web app)

### 3.3 Training Tasks (no human labels required at first)

Self-supervised text:

Contrastive learning across duplicate spans (same paragraph, different scans), and between text vs symbolic_context.

Weakly-supervised symbol detection:

Bootstrap small golden set; augment with geometry rules (junction vs crossing, pin counts).

Sequence tagging for electrical tokens:

BIO tagging of NET, COMP_ID, RATING, FUSE_SLOT, GROUND_CODE based on regex+rules → use as noisy labels.

Graph consistency losses:

Penalize component types whose pins consistently violate learned role priors (e.g., op-amp must have power pins on VCC/VEE).

Distillation:

Use a strong VLM offline to label 100–500 pages/month; distill into efficient runtime models.

### 3.4 Model Zoo (lightweight & swappable)

Embeddings: sentence-transformer-class (384–768 dims) fine-tuned on manuals (contrastive).

Re-ranker: cross-encoder tiny (≤ 50M params).

Tagger: BiLSTM/CRF or tiny Transformer for token tagging (electrical entities).

Detector: YOLOv8n/s or D2-R50 for symbols; class set versioned: symset_v1.

Heuristics: rule packs with weights (learnable thresholds).

### 3.5 Curriculum

Start with clean printed schematics (vector-to-raster).

Add noisy scans (deskewed).

Add photos of manuals (phone glare etc.).

Add multilingual (labels often English, but region terms differ).

# 3.6 Versioning & Promotion

Track via model_registry (Supabase):

{
  "name": "symbol_detector",
  "version": "1.3.2",
  "s3_uri": "s3://…/detector_v1.3.2.pt",
  "metrics": {"mAP":0.78,"net_F1":0.72},
  "stage": "staging|prod",
  "changelog": "junction rule fix; pin IO prior"
}


Blue/Green load by tag CURRENT_PROD_*.

Auto-rollback if error budget exceeded (see §6).

## 4) Quality & Evaluation
### 4.1 Metrics

OCR: CER/WER overall + for rotated/condensed fonts.

Symbols: mAP@[.5:.75], per-class recall (pay attention to junction, net_label).

Connectivity: Net membership accuracy, dangling-pin rate, cross-short false alarms.

Search: NDCG@10, Recall@50, MRR (per query template: “Where is starter relay?”, “Which fuse protects ECU?”).

E2E F1: 0.4Text + 0.3Symbols + 0.3*Connectivity.

### 4.2 Regression Suite

benchmarks/run.py --full

Clean vs Noisy subsets

Reports saved under benchmarks/results/{date}.json and summarized to Markdown.

### 4.3 Error Mining

Nightly job flags:

Low-confidence nets with high impact systems (Starting, Charging).

Consistent mismatches between tagger and detector (e.g., token says “F18” but no fuse box adjacency).

Candidate pages for manual adjudication.

## 5) Knowledge Consolidation
### 5.1 Normalization Tables (Supabase)

component_types(id, name, pin_roles[], typical_values[])

fuse_panels(vehicle_signature, panel_name, slot_map jsonb)

grounds(vehicle_signature, code, location_hint)

wires(color_code, gauge_awg, typical_role)

### 5.2 Graph Sanity Constraints (checked before Neo4j write)

Junctions must create ≥3 incident segments.

Net labels must not create cycles with conflicting voltages.

Power nets must reach at least one fuse/relay unless marked direct_batt.

## 6) Ops & Safety

Shadow evaluation: new model versions run in shadow for X% of jobs and log deltas only.

Guardrails: if E2E F1 drops by >5% vs baseline over last 100 pages → auto-rollback.

Resource control: detectors on CPU-first; batch on GPU when present; rate-limit per tenant.

Privacy/licensing: store features only for proprietary manuals if required; retain raw pages behind signed URLs.

## 7) Implementation Tasks (Patch Adds)

Semantic Module

 src/semantic/embed.py (multi-head embeddings)

 src/semantic/search.py (hybrid, rerank)

 GET /v1/search returns mixed hits with reasons

 Migrate payload schema (add symbolic_context, component_ids, net_names, ts_lex)

Learning Jobs

 cron/weekly_eval.py → runs full bench, writes model_registry

 cron/self_train_text.py → contrastive fine-tune embeddings on mined positives

 cron/weak_label_tokens.py → generate noisy BIO labels from rules

 cron/detector_refresh.py → fine-tune YOLO on error-mined patches

Model Registry & Hot-Load

 src/core/models.py → download by tag, warm-up, health checks

 Supabase trigger to signal model_version_change → hot-reload

Consistency Validators

 src/schematics/validate.py → pre-Neo4j checks

 Failure → artifacts to artifacts/triage/… with overlay PNG

Bench & Telemetry

 Expand benchmarks/run.py for NDCG/MRR + E2E F1

 Prometheus counters for: CER moving avg, net_accuracy, search_ndcg

## 8) Sample Config
semantic:
  dense_model: "wsl/auto-elec-embed-v0.3"
  dense_dim: 512
  re_ranker: "wsl/auto-elec-rerank-tiny"
  use_pgvector: false
  qdrant_collection: "wessley_docs"
  topk_dense: 50
  topk_sparse: 50
  topk_final: 20

learning:
  enable_cron: true
  weekly_eval_cron: "0 3 * * 1"
  detector_finetune_quota_hours: 2
  shadow_eval_sample: 0.2

## 9) Example: Re-ranker Input (for transparency)
[QUERY]  "where is the starter relay"
[DOC1]   "Starter relay (K1) located at engine bay fuse/relay box, slot R3..."
[DOC2]   "Wiring: IG1 feeds coil of starter relay via clutch switch..."
[DOC3]   "Fuse F10 (30A) protects starter motor circuit..."


Cross-encoder scores these; top-k returned with component/net anchors.

## 10) Roadmap (90 days)

Week 0–2: Implement semantic module + hybrid search; add telemetry & basic eval.

Week 3–5: Introduce token tagger + weak labels; grow ontology tables; first shadow promotions.

Week 6–8: Detector fine-tune on mined errors; add junction vs crossing robustness; E2E F1 ≥ 0.80 clean, ≥ 0.70 noisy.

Week 9–12: Curriculum on multi-brand manuals; location hints; start failure-mode queries (“no crank: check K1, F10, G102”).