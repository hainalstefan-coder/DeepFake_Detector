# Stage 2 Git Forensics

## Baseline Commit

- **Hash:** `f08acd97c7252cf9893361c33d28f8c509d0d135`
- **Date:** 2026-01-15 23:00:57 +0200
- **Message:** Initial commit: Mandrake Deepfake Detector
- **Why selected:** The repository contains exactly one commit. It is also the closest ancestor to the expected Stage 1 delivery window. Because no commits exist around February 11, this initial commit is both the Stage 1 baseline and the current HEAD.

## Current HEAD

- **Hash:** `f08acd97c7252cf9893361c33d28f8c509d0d135`
- **Date:** 2026-01-15 23:00:57 +0200
- **Message:** Initial commit: Mandrake Deepfake Detector
- **Branch:** `main` (only local + remote branch)

## Commit Count After Baseline

0 commits after the baseline.

## Changed Files Summary

There are no commits after the baseline. The diff between baseline and HEAD is empty.

## Commit Timeline

| # | Commit | Date | Message | Files | Category | Dissertation relevance |
|---|--------|------|---------|-------|----------|-------------------------|
| 1 | f08acd9 | 2026-01-15 | Initial commit: Mandrake Deepfake Detector | ~30 | Core pipeline + infrastructure | Defines entire project state |

## Functional Groups

Because there is only one commit, all changes belong to that single initial snapshot. The functional groups below describe the snapshot itself, not post-baseline evolution.

### 1. Core detection pipeline
- **Files:** `app/agents/cnn_classifier.py`, `app/agents/vit_classifier.py`, `app/agents/frequency.py`, `app/agents/embedding_anomaly.py`, `app/agents/face_xray.py`, `app/agents/registry.py`
- **What changed:** Five detector agents were introduced with pretrained backbones (EfficientNet-B0, ViT-B/16, custom FrequencyMLP, custom EmbeddingEncoder, custom BoundaryDetector) plus an auto-discovery registry.
- **Evidence:** `app/agents/*.py` present in initial commit.

### 2. Model integration
- **Files:** `requirements.txt`, `app/config.py`
- **What changed:** PyTorch 2.1, timm, facenet-pytorch, insightface declared. `get_device()` prefers CUDA with CPU fallback.
- **Evidence:** `requirements.txt` lines 1–39; `app/config.py` lines 101–111.

### 3. Data preprocessing
- **Files:** `app/preprocessing/face_pipeline.py`
- **What changed:** Simplified center-crop face extractor using PIL only. MTCNN declared in config but not implemented.
- **Evidence:** `app/preprocessing/face_pipeline.py` lines 21–180.

### 4. UI / UX
- **Files:** `app/static/index.html`, `app/static/styles.css`, `app/static/app.js`
- **What changed:** Vercel-inspired dark theme with live agent status cards.
- **Evidence:** Files present in `app/static/`.

### 5. API / backend
- **Files:** `app/main.py`, `app/api/routes.py`, `app/api/websocket.py`
- **What changed:** FastAPI app with `/api/analyze`, `/api/result/{id}`, `/api/jobs`, `/api/health`, and `/ws/{job_id}`.
- **Evidence:** `app/main.py` lines 56–98; `app/api/routes.py` lines 33–183; `app/api/websocket.py` lines 53–111.

### 6. Agentic workflow
- **Files:** `app/core/orchestrator.py`, `app/core/message_bus.py`, `app/models/schemas.py`
- **What changed:** Mandrake-inspired orchestrator with Remind (timeout + retry), Checkpoint (immediate job_store.update_job), and Continue (quorum 3 of 5). Async pub/sub message bus with idempotence and per-job WebSocket queues.
- **Evidence:** `app/core/orchestrator.py` lines 1–590; `app/core/message_bus.py` lines 1–161.

### 7. Evaluation / testing
- **Files:** `tests/test_aggregation.py`, `tests/test_message_bus.py`
- **What changed:** Unit tests for weighted average, quorum detection, uncertainty, verdict threshold, message bus pub/sub + idempotence + error isolation.
- **Evidence:** `tests/test_aggregation.py` lines 1–200; `tests/test_message_bus.py` lines 1–143.

### 8. Build / environment
- **Files:** `requirements.txt`, `config/config.yaml`, `.gitignore`
- **What changed:** Dependency list, runtime config with weights and thresholds, ignores for models/data/IDE.
- **Evidence:** `requirements.txt`; `config/config.yaml`.

### 9. Dissertation-related files
- **Files:** None added in repository. The repository contains no `/papers` directory and no `outputs/papers_bibliography.csv`, although the dissertation text references them.
- **Evidence:** `find . -maxdepth 3 -type d` shows no `papers` or `outputs` directories.

## Detailed Findings

1. **Stage 1 baseline and HEAD are identical.** There are no commits after the initial delivery. Any technical progress made since February 11 is not tracked in the Git history.
2. **Verifier modules are absent.** The existing dissertation describes "two verifier modules" (Evidence and Robustness). The repository contains only the 5 detectors and the aggregator inside the orchestrator. There is no `verifiers/` directory, no separate verifier class, and no distinct "verifier" concept in the schemas.
3. **Pretrained weights are not included.** Head initializations are random (`nn.init.xavier_uniform_`, `nn.init.zeros_`). No `.pt`, `.pth`, or `.onnx` files are tracked.
4. **Face preprocessing is a placeholder.** The face pipeline assumes the face is centered. Real face detection (MTCNN, RetinaFace) is declared in config but not implemented.
5. **Dependencies contain unused libraries.** `insightface`, `opencv-python`, `facenet-pytorch` are listed but not imported in the active preprocessing pipeline.

## Evidence

| Command | Result |
|---------|--------|
| `git log --oneline -50` | 1 commit only |
| `git log --since="2026-02-01" --until="2026-02-20" --oneline` | (empty) |
| `git branch -a` | `* main` |
| `find . -maxdepth 3 -type f \| sort` | Confirms file list above |
| `git show --stat HEAD` | Shows full tree in single commit |
