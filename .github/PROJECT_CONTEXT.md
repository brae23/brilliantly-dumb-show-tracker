# SYSTEM CONTEXT: The Brilliantly Dumb Show - Podcast RAG Pipeline

## 1. High-Level Overview
* **Goal:** Automated RAG pipeline for podcast archives (YouTube audio -> WhisperX -> pgvector -> LLM search).
* **Architecture:** Monorepo with static Astro frontend, FastAPI Query API, Fargate ingestion, and Terraform IaC.
* **Current Execution Mode:** Local development mode (running scripts via Python/`uv` directly; Docker not yet configured).

## 2. Tech Stack & Monorepo Structure
* **Monorepo Layout:**
  * `services/ingestion/`: WhisperX ASR, Pyannote diarization, semantic chunking scripts.
  * `services/query_api/`: FastAPI service handling vector similarity queries and LLM prompt generation.
  * `apps/web/`: Astro static frontend (SSG) hosted on AWS S3 + CloudFront.
  * `infrastructure/`: Terraform IaC for provisioning RDS PostgreSQL (pgvector), ECS Fargate, S3, CloudFront, Lambda, and IAM.
* **Environment & Package Management:** Python 3.11+, managed strictly via `uv` (`uv run`, `pyproject.toml`).
* **Models & APIs:**
  * ASR/Diarization: WhisperX (CPU/PyTorch) + Pyannote.
  * Embeddings: OpenAI `text-embedding-3-small` (1536-dim).
  * LLM Inference: OpenAI `gpt-4o-mini` (RAG generation & speaker identity resolution).
  * Database: PostgreSQL + `pgvector` extension (HNSW index post-load).

## 3. Data & Schema Standards
* **Local Raw Staging:** Raw WhisperX output saved as JSON in `/data/samples/` to decouple transcription from chunking iterations.
* **Chunking Strategy:** `HybridSemanticChunker` (`window_size=1`, `breakpoint_percentile=80.0`, `max_tokens=400`).
* **Database Schema Layout:**
  * `chunks`: `id`, `episode_id`, `text_content` (pure spoken dialogue, NO speaker tags), `embedding` (`vector(1536)`).
  * `metadata`: `speaker_name` (relational `VARCHAR`), `start_time`, `end_time`, `youtube_timestamp_url`.

## 4. Key Architectural Decisions (DO NOT OVERRIDE)
1. **Decoupled Relational Metadata:** Spoken dialogue is embedded without speaker prefixes. Speaker names are stored in separate relational columns to allow async SQL updates without re-embedding text vectors.
2. **2-Stage Backfill Strategy:** Test parameters and run sweeps on local raw JSONs (*Sample & Lock*) prior to running the full production archive backfill.
3. **Frontend Architecture:** Astro (Static Site Generation / SSG) deployed to S3/CloudFront. Zero egress/bandwidth costs under expected volume (< 150k monthly requests).

## 5. Current State & Active Focus
* **Status:** Local dev mode (no Docker setup yet). Python env managed with `uv`.
* **Completed:** Audio download logic, WhisperX diarization pipeline design, 2-Stage backfill strategy, and master plan documentation.
* **Active Focus:** Implementing `Semantic Split Ratio` logging (`semantic` vs. `max_tokens` splits) and setting up parameter sweep scripts on 3 golden sample JSONs.