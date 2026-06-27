# Stage 2 Project Understanding

## Current Repository State

| Property | Value |
|----------|-------|
| Repository URL | https://github.com/hainalstefan-coder/DeepFake_Detector |
| Local path | /Users/patronumiu78/Desktop/projects/DeepFake_Detector_Agent |
| Branch | main (single branch) |
| HEAD commit | f08acd9 — "Initial commit: Mandrake Deepfake Detector" |
| HEAD date | 2026-01-15 |
| Commits total | 1 |
| Working tree | clean |
| Uncommitted changes | none |

## Main Goal of the Project

Build a multi-agent deepfake detection system for static face images. The design is inspired by the Mandrake fault-tolerant decentralized multiagent system [10.1007/s10458-021-09540-8]. The system runs several independent detection agents concurrently, streams intermediate evidence via WebSocket, applies fault-tolerance patterns (Remind, Checkpoint, Continue), and aggregates outputs into a final REAL/FAKE verdict with a structured report.

## Current Architecture

```
app/
├── main.py                 # FastAPI application, lifespan, CORS, static mount
├── config.py               # Pydantic Settings loader (YAML-backed)
├── models/
│   └── schemas.py          # TaskMessage, ResultMessage, ErrorMessage, AgentState,
│                            # AggregatedResult, JobState, WebSocketEvent
├── core/
│   ├── agent_base.py       # BaseAgent (async interface, timeout, error isolation)
│   ├── message_bus.py      # Async pub/sub, idempotence, job-based event queues
│   └── orchestrator.py     # Mandrake orchestrator (Remind/Checkpoint/Continue)
├── agents/
│   ├── registry.py         # Auto-discovery + decorator-based registration
│   ├── cnn_classifier.py   # EfficientNet-B0 binary classifier
│   ├── vit_classifier.py   # ViT-B/16 attention-based classifier
│   ├── frequency.py        # FFT/DCT + MLP (GAN frequency artifacts)
│   ├── embedding_anomaly.py# Custom CNN encoder + ArcFace-style similarity
│   └── face_xray.py        # BoundaryDetector + edge/heuristic scoring
├── preprocessing/
│   └── face_pipeline.py    # Simplified center-crop face extractor (PIL-only)
├── storage/
│   └── job_store.py        # SQLite + in-memory persistence (Checkpoint)
├── api/
│   ├── routes.py           # REST: /analyze, /result/{id}, /jobs, /health
│   └── websocket.py        # WebSocket per job, event streaming
└── static/
│   ├── index.html          # Frontend UI
│   ├── styles.css          # Vercel dark theme
│   └── app.js              # Live agent status cards
config/
└── config.yaml             # Timeouts, weights, thresholds, preproc, storage, server
data/
└── jobs.db                 # SQLite database (initialized at runtime)
tests/
├── test_aggregation.py     # Weighted avg, quorum, uncertainty, threshold tests
└── test_message_bus.py     # Pub/sub, idempotence, error isolation tests
```

## Main Modules

| Module | Role |
|--------|------|
| `app.main` | Application bootstrap, lifespan, routing, static serving |
| `app.config` | YAML + defaults via Pydantic Settings |
| `app.core.agent_base` | Abstract base agent with `initialize`/`process`/`run` |
| `app.core.orchestrator` | Task dispatch, timeout monitor, retry, checkpoint, quorum finalization, aggregation |
| `app.core.message_bus` | Topic-based async pub/sub, idempotent message handling, WebSocket event queues |
| `app.agents.*` | 5 concrete detectors, each with lazy model loading |
| `app.preprocessing.face_pipeline` | Simplified face crop (center-crop heuristics) |
| `app.storage.job_store` | Persist JobState to SQLite or memory |
| `app.api.routes` | REST endpoints |
| `app.api.websocket` | Per-job WebSocket stream |

## Execution Flow

1. Client uploads image to `POST /api/analyze`.
2. File saved to job directory.
3. `FacePreprocessor` crops face (simplified center-crop).
4. Orchestrator creates `JobState`, dispatches `TaskMessage` to all 5 agents concurrently via `asyncio.create_task`.
5. Each agent runs `process(task)` inside `BaseAgent.run()` with timeout + error isolation.
6. On success → `on_agent_result` → Checkpoint (persist) → emit `agent_completed`.
7. On failure → `on_agent_error` → Remind (retry if under `max_retries`) or mark ERROR.
8. Background `_monitor_timeouts` enforces timeout unless agent finishes naturally.
9. When `completed >= quorum` or `pending == 0` → Continue → `aggregate_results` → final verdict.
10. Final event streamed via WebSocket; REST `GET /api/result/{id}` returns full state.

## Detection Agents

| Agent | Technique | Input size | Details |
|-------|-----------|------------|---------|
| `CNNClassifierAgent` | EfficientNet-B0 + binary head | 224×224 | ImageNet pretrained, head randomly initialized |
| `ViTClassifierAgent` | ViT-B/16 + binary head | 224×224 | ImageNet pretrained, head randomly initialized |
| `FrequencyAgent` | 2D FFT radial + angular profile + MLP | 256-d vector | Heuristic HF-ratio bias |
| `EmbeddingAnomalyAgent` | Custom CNN encoder (512-d) + ArcFace cosine similarity | 112×112 | Zero reference embedding (random init) |
| `FaceXrayLikeAgent` | BoundaryDetector CNN + edge heuristics | 128×128 | Custom CNN + color imbalance scoring |

## Deepfake Detection Pipeline

Input image → face crop → 5 parallel detectors → weighted aggregation (quorum 3/5) → verdict (REAL/FAKE) with confidence and uncertainty.

## User Interaction Flow

- FastAPI serves a Vercel-inspired dark UI at `/`.
- Upload triggers real-time updates via `/ws/{job_id}`.
- Events: `job_started`, `agent_started`, `agent_completed`, `agent_error`, `agent_retry`, `job_completed`.
- Polling fallback via `GET /api/result/{job_id}` and `GET /api/jobs`.

## Dependencies

- Python 3.11+
- FastAPI 0.104+, uvicorn
- PyTorch 2.1+, torchvision, timm
- facenet-pytorch, insightface (declared but not used in code)
- opencv-python (declared but `face_pipeline.py` uses PIL only)
- numpy, scipy, Pillow
- aiosqlite, PyYAML, pydantic-settings
- pytest, pytest-asyncio, httpx

## Existing Documentation

- `README.md` — Project overview, quick start, API docs, configuration.
- Inline docstrings — All modules include explanatory docstrings.
- `config/config.yaml` — Default parameters with comments.
- No `docs/` directory.
- No `papers/` directory or bibliography export present in the repository.

## Existing Dissertation Files

| File | Location |
|------|----------|
| Stage I dissertation | `Disertatie/Dissertation_Progress_Stefan_Hainal.docx` |
| First task prompt | `Disertatie/FIRST_TASK.md` |
| Today’s mission | `Disertatie/SOUL.md` |

The dissertation Part I (read via `python zipfile` extraction) describes an 8-component pipeline (5 detectors, 2 verifiers, 1 aggregator) and a semester-based roadmap.

## What Is Clear

1. The repository is a functioning FastAPI + WebSocket multi-agent prototype.
2. Mandrake patterns (Remind, Checkpoint, Continue) are implemented in the orchestrator.
3. Five detection agents are implemented as `BaseAgent` subclasses.
4. Aggregation logic (weighted average, uncertainty, quorum) is implemented and unit-tested.
5. Message bus and job store have unit tests.
6. No trained model weights are included — all heads are randomly initialized.
7. Face preprocessing is a simplified center-crop; no real MTCNN/retinaface usage.
8. `insightface` and `opencv-python` are declared but not actually used in `face_pipeline.py`.
9. The verifier modules referenced in the dissertation Part I are absent from the code.
10. The `/papers` folder and `outputs/papers_bibliography.csv` referenced in the dissertation are not present in the repository.
11. Git history contains only the initial commit; there are no tracked incremental commits after January 15, 2026.

## What Is Unclear

1. Whether the verifier modules were ever implemented outside the tracked git history or were planned only.
2. Whether the dissertation’s description of "two verifier modules" refers to conceptual design or implemented code.
3. The baseline around February 11 cannot be identified via Git because there are no commits on that date or after it.
4. Whether the user worked on the code locally and forgot to commit, or whether the project was delivered as a single commit.
5. Whether the initial commit date (Jan 15) corresponds to the Stage 1 delivery, or whether Stage 1 was delivered in February as a separate artifact not captured in git.
6. Current performance metrics (no benchmark results, no evaluation data present).
7. Whether the system has been tested end-to-end.

## Evidence Log

| Evidence | Source |
|----------|--------|
| Mandrake design documented in README | `README.md` lines 1–275 |
| 5 detection agents implemented | `app/agents/*.py` |
| Orchestrator with Remind/Checkpoint/Continue | `app/core/orchestrator.py` lines 1–590 |
| WebSocket streaming | `app/api/websocket.py`, `app/core/message_bus.py` |
| Tests present | `tests/test_aggregation.py`, `tests/test_message_bus.py` |
| Single git commit | `git log --oneline -1` |
| No verifier modules found | absence of `app/verifiers/` or similar |
| No papers/ directory | `find . -maxdepth 3 -type d` |
