# Stage 2 Evidence Log

Tracked claims with their support level.

| # | Claim | Evidence type | File/commit | Confidence | Notes |
|---|-------|---------------|-------------|------------|-------|
| 1 | Repository is a Mandrake-style multi-agent deepfake detector | README + code | `README.md`; `app/core/orchestrator.py` | High | README title and docstrings explicitly state this |
| 2 | 5 detection agents implemented | Code | `app/agents/registry.py` lines 70–80 | High | Registry lists exactly 5 classes |
| 3 | Orchestrator implements Remind, Checkpoint, Continue | Code | `app/core/orchestrator.py` lines 6–8, 171–225, 230–305, 358–414 | High | Docstrings and logic match Mandrake concepts |
| 4 | Async message bus with idempotence | Code | `app/core/message_bus.py` lines 24–94 | High | `_processed_messages` set and explicit idempotence docs |
| 5 | Unit tests exist | Code | `tests/test_aggregation.py`; `tests/test_message_bus.py` | High | Both files present |
| 6 | Weighted aggregation uses configurable weights | Code | `app/core/orchestrator.py` lines 452–473; `config/config.yaml` lines 11–16 | High | Config model and aggregation loop visible |
| 7 | Quorum-based finalization | Code | `app/core/orchestrator.py` lines 376–414 | High | `quorum` compared against `completed` count |
| 8 | Pretrained backbones are used (ImageNet), heads randomized | Code | `app/agents/cnn_classifier.py` lines 45–55; `app/agents/vit_classifier.py` lines 44–53 | High | `pretrained=True` + manual head init |
| 9 | No trained deepfake-specific weights included | Absence + code | `.gitignore` lines 58–61; `app/agents/*.py` | High | `*.pt`, `*.pth`, `*.onnx` ignored |
| 10 | Face preprocessing is center-crop heuristics | Code | `app/preprocessing/face_pipeline.py` lines 42–66 | High | No MTCNN/RetinaFace logic implemented |
| 11 | `insightface` and `opencv-python` are declared but unused | Code | `requirements.txt` lines 17–23; `app/preprocessing/face_pipeline.py` | High | No imports of cv2 or insightface in active code |
| 12 | WebSocket streaming per job | Code | `app/api/websocket.py` lines 53–111; `app/main.py` lines 77–80 | High | `/ws/{job_id}` endpoint wired |
| 13 | SQLite persistence for jobs | Code | `app/storage/job_store.py` lines 57–69 | High | `aiosqlite` schema creation visible |
| 14 | Two verifier modules from dissertation are not implemented | Absence | Directory tree under `app/` | High | No `verifiers/` directory, no verifier class |
| 15 | Dissertation-referenced `/papers` directory is absent | Absence | `find . -name papers` | High | No papers folder in repo |
| 16 | Dissertation-referenced `outputs/papers_bibliography.csv` is absent | Absence | `find . -name papers_bibliography.csv` | High | Not found |
| 17 | Git history contains exactly 1 commit | Git | `git log --oneline` | High | Single initial commit |
| 18 | No commits exist around February 11 | Git | `git log --since="2026-02-01" --until="2026-02-20" --oneline` | High | Empty output |
| 19 | Current HEAD and Stage 1 baseline are the same commit | Git | `git rev-parse HEAD` + date check | High | Only one commit in repo |
| 20 | System supports static-image analysis only | Code | `app/models/schemas.py` lines 27–31, 34–46; README | High | `Modality` enum prepared for VIDEO/AUDIO but not used |
| 21 | Timeout + retry logic exists (Remind) | Code | `app/core/orchestrator.py` lines 249–277, 321–356 | High | Configurable `timeout_seconds` and `max_retries=1` |
| 22 | Frontend is a single-page HTML/JS app | Code | `app/static/index.html`, `app/static/app.js` | High | Static mount in `app/main.py` |
