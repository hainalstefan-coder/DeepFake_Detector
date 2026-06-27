# Stage 2 Next Steps

## Immediate next steps for dissertation

| Priority | Task | Why it matters | Evidence/gap | Suggested branch |
|----------|------|----------------|--------------|------------------|
| P0 | Verify end-to-end pipeline locally | Prove the prototype actually runs from upload to verdict | No integration test, no sample output in repo | `verify/e2e-pipeline` |
| P0 | Implement Evidence verifier | Matches Part I design; enables cross-detector consistency scoring | `app/verifiers/` absent; dissertation Part I describes 2 verifiers | `feature/verifier-evidence` |
| P0 | Implement Robustness verifier | Matches Part I design; enables perturbation stability checks | same as above | `feature/verifier-robustness` |
| P1 | Replace random model heads with trained weights | Without weights, detection scores are meaningless | `nn.init.xavier_uniform_` in all agents; no `.pt` files | `train/cnn-vit-heads` |
| P1 | Add real face detector (MTCNN or RetinaFace) | Current center-crop is non-robust | `app/preprocessing/face_pipeline.py` lines 42–66 | `preprocessing/real-face-detector` |
| P1 | Clean unused dependencies | Prevents environment failures on macOS | `requirements.txt` lists `insightface`, `opencv-python`, `facenet-pytorch` but unused | `chore/clean-dependencies` |
| P1 | Create evaluation dataset + splits | Required for AUC/F1/accuracy | No `data/` splits or `papers/` references present in repo | `data/curate-dataset` |
| P2 | Run ablation and robustness suite | Directly from Part I roadmap | No ablation scripts or stress-test harness | `eval/ablations-robustness` |
| P2 | Add reproducibility artifacts (seeds, configs) | Dissertation defense requirement | Roadmap Semester 2 mentions this | `docs/reproducibility` |
| P2 | Expand frontend for real-time charts | Improves demonstration quality | `app/static/app.js` is minimal | `feature/ui-enhancements` |
| P3 | Add video support (frame extraction + temporal modeling) | Part of roadmap Semester 3 | `Modality` enum prepared but unused | `feature/video-support` |
| P3 | Add audio signals (spectral, speaker consistency) | Part of roadmap Semester 3 | Not started | `feature/audio-support` |
| P3 | Implement metric randomization / security module | Part of roadmap Semester 4 | Not started | `feature/security-extensions` |
| P3 | Implement blockchain-anchored reporting prototype | Part of roadmap Semester 4 | Not started | `feature/blockchain-anchoring` |
| P3 | Port to macOS / Metal (MPS) | User now on M2 Max with 96 GB RAM | `get_device()` prefers CUDA; macOS not supported | `port/macos-metal` |

## Immediate next steps for repository

1. **Add a `docs/` directory** with README-style architecture diagrams and setup instructions.
2. **Add `.gitkeep` placeholders** in `data/jobs/` and `data/uploads/` to keep the directory structure in Git.
3. **Add CI workflow** (`GitHub Actions`) for tests and linting.
4. **Add a `papers/` directory** with the referenced literature corpus and export the bibliography CSV.
5. **Add a `CHANGELOG.md`** to make incremental progress visible even without fine-grained commits.

## Experiments needed

- Train classifiers on a labeled deepfake dataset and report:
  - per-agent AUC / EER / F1 / accuracy;
  - calibrated confidence (temperature scaling or isotonic regression);
  - cross-dataset generalization numbers.
- Ablation:
  - remove each agent in turn and measure impact on final accuracy;
  - perturb inputs (JPEG Q=[10, 30, 70], resize [0.5x, 0.75x, 1.5x], Gaussian blur);
  - measure drop in AUC / confidence.
- Verifier impact:
  - compare weighted-average-only aggregator vs. weighted + Evidence verifier vs. weighted + Robustness verifier vs. both.

## Validation needed

- End-to-end test with at least 20 real and 20 fake images from a held-out split.
- WebSocket streaming validated against simulated slow agents (artificial latency injection).
- Face-preprocessing failure mode tested on off-center and multi-face images.
- Unit tests expanded to cover the orchestrator timeout monitor and full job lifecycle.

## Figures/screenshots needed

- Mermaid architecture diagram (data flow from upload through agents to UI).
- Screenshot of the live UI with agent cards mid-analysis.
- WebSocket event log snippet showing real-time streaming.
- Table of detector specifications (model, input size, signal family, current weight).
- Table of test coverage matrix.
- Roadmap table mapping Part I semesters to actual repository components.

## macOS migration tasks

- Replace `torch.cuda.is_available()` branch with a device priority: `mps` > `cuda` > `cpu`.
- Update `requirements.txt` to track macOS-specific pins if needed.
- Remove `insightface` and `opencv-python` or replace them with lightweight alternatives.
- Add macOS test matrix to CI.

## Risks before final dissertation

| Risk | Mitigation |
|------|-----------|
| No trained weights | Allocate fixed time for training; if insufficient, clearly scope dissertation to the prototype + methodology only |
| Verifier gap remains | Treat verifier implementation as the critical milestone; defer multimodal/security if needed |
| Git history gap | Document the single-commit reality explicitly in Stage II; avoid claiming incremental development |
| Bibliography corpus missing | Create `papers/` and export CSV immediately |
| Frontend flakiness | Test with `curl` + WebSocket client before relying on UI for defense demo |
| Environment divergence (Mac vs. Windows) | Use `pyproject.toml` or `uv` to freeze dependencies; test on target machine early |

## Priority list

| Priority | Task | Evidence/gap |
|----------|------|--------------|
| P0 | Proof-of-concept end-to-end run | Prototype unverified |
| P0 | Implement missing verifiers | Dissertation/plan mismatch |
| P1 | Train model heads | Random initialization |
| P1 | Real face detection | Center-crop heuristic |
| P1 | Dependency cleanup | Environment risk |
| P1 | Evaluation dataset | No metrics available |
| P2 | Ablation + robustness studies | Roadmap requirement |
| P2 | Reproducibility package | Dissertation defense requirement |
| P3 | Multimodal extensions | Roadmap Semester 3 |
| P3 | Security extensions | Roadmap Semester 4 |
| P3 | macOS port | User hardware change |
