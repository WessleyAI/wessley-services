# Open Questions

This document tracks clarifications needed for the ingestion service implementation.

## Current Open Questions

### 1. Embedding Model Selection
**Question**: Which embedding model is canonical across the monorepo?
**Impact**: Affects Qdrant vector dimensions and semantic search compatibility
**Current Default**: Using float32[768] as specified in CLAUDE.md
**Resolution Needed**: Confirm if this aligns with other services

### 2. YOLO Model Configuration
**Question**: Preferred YOLO backbone and label set versioning strategy?
**Impact**: Component detection accuracy and model deployment size
**Current Default**: Planning to use YOLOv8 with configurable label set
**Resolution Needed**: Model weights source and versioning approach

### 3. Component ID Normalization
**Question**: Do we normalize component IDs across pages (e.g., deduplicate "R1" references)?
**Impact**: Graph storage schema and component uniqueness
**Current Default**: Treating each page component as separate entity
**Resolution Needed**: Cross-page component identity resolution strategy

### 4. Authentication Integration
**Question**: Specific Supabase JWT configuration and RLS policy structure?
**Impact**: User isolation and permission management
**Current Default**: Basic JWT validation placeholder
**Resolution Needed**: Complete auth flow and policy definitions

### 5. Error Recovery Strategy
**Question**: How should the service handle partial failures (e.g., OCR succeeds but schematic parsing fails)?
**Impact**: Job status reporting and artifact availability
**Current Default**: All-or-nothing job completion
**Resolution Needed**: Partial success handling approach

### 6. Real-time Channel Naming
**Question**: Standard convention for Supabase realtime channel naming across services?
**Impact**: Client subscription patterns and namespace organization
**Current Default**: `realtime:ingestions:{job_id}` format
**Resolution Needed**: Confirm naming convention consistency

### 7. Artifact Storage Strategy
**Question**: S3 bucket organization and lifecycle policies for generated artifacts?
**Impact**: Storage costs and data retention
**Current Default**: Simple `/jobs/{job_id}/` prefix structure
**Resolution Needed**: Retention policies and cleanup schedules

### 8. Performance Benchmarking
**Question**: Standard datasets and metrics for cross-service performance comparison?
**Impact**: Service optimization targets and SLA definitions
**Current Default**: Custom fixture-based benchmarks
**Resolution Needed**: Shared benchmarking infrastructure

## Resolved Questions

None yet - this section will track resolved clarifications.

---

## Notes

- All questions have sensible defaults implemented to unblock development
- Configuration is externalized via environment variables where possible
- Architecture supports changing decisions without major refactoring

## Next Steps

1. Review questions with product/architecture team
2. Update implementation based on decisions
3. Document final approaches in README.md
4. Move resolved items to "Resolved Questions" section