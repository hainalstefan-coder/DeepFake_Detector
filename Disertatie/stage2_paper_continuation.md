# Deepfake Detection: Stage II Progress and Updated Roadmap

Stefan Eugen Hainal
Technical University of Cluj-Napoca (UTCN), Romania

Abstract— This document reports the technical progress made on the DeepFake Detector Agent after the initial dissertation delivery (Stage I, February 2026). The project implements a Mandrake-inspired multi-agent detection system in which independent detectors contribute heterogeneous forensic signals, two verifier modules assess consistency and stability, and an aggregator produces a final verdict with structured evidence. Since Stage I, the repository has been advanced with the implementation and testing of the Evidence and Robustness verifiers, the addition of an OpenCV face-detection backend with fallback logic, and an expanded unit-test suite. This paper documents these concrete changes, maps them against the architectural blueprint from Stage I, identifies remaining gaps, and updates the semester-based roadmap. All claims are directly supported by repository code, commit history, or the Stage I document. Index Terms— deepfake detection, multi-agent systems, forensic cues, robustness verification, evidence verification, weighted fusion, reproducibility.

# I. Introduction

Advances in generative modeling have produced synthetic media of such fidelity that distinguishing manipulated from authentic content has become a first-class security and forensic concern. Single-model deepfake detectors are brittle across generators, imaging conditions, and post-processing pipelines [1], [2]. A multi-agent architecture that aggregates complementary evidence — spatial, frequency, embedding-level, and boundary-level — while remaining robust to individual component failures offers a principled path forward.

This dissertation project adopts a Mandrake-inspired design: detectors communicate through a structured protocol, make local decisions, and converge on a global outcome via explicit coordination patterns (Remind, Checkpoint, Continue) [3]. Stage I of the dissertation delivered the architectural blueprint and a prototype implementing five detectors, one aggregator, and a FastAPI/WebSocket interface. The roadmap called for trained weights, verifier modules, multimodal extensions, and security features across three subsequent semesters.

This Stage II document reports concrete engineering advances realized after the first delivery. Two components originally specified in Stage I — the Evidence verifier and the Robustness verifier — have been implemented, tested, and integrated into the aggregation workflow. The face preprocessing pipeline has been extended with an OpenCV Haar-cascade backend while preserving a fallback path. These changes are tracked incrementally in Git and validated by unit tests.

# II. Implemented Components Since Stage I

## A. Evidence Verifier

The Evidence verifier (`app/verifiers/evidence.py`, class `EvidenceVerifier`) operates on the list of successful agent results before aggregation. It computes the median score and standard deviation of the ensemble. Any agent whose score diverges from the median by more than `outlier_threshold` (default 0.25) is flagged as an outlier and down-weighted by `down_weight_factor` (default 0.5), pulling its contribution toward the median. Agents whose scores fall within `plausibility_margin` (default 0.15) of 0.5 are flagged as weak evidence and mildly down-weighted. The module returns adjusted scores and a structured report including per-agent flags and aggregate statistics (mean, median, standard deviation, outlier count, plausibility-issue count).

The verifier is instantiated and applied inside `Orchestrator.aggregate_results` when `verifier.evidence_enabled` is true in `config.yaml`. The resulting report is stored in `AggregatedResult.evidence_verifier_report` and exposed via the API/WebSocket layer.

## B. Robustness Verifier

The Robustness verifier (`app/verifiers/robustness.py`, class `RobustnessVerifier`) operates immediately after the Evidence verifier. It computes a trimmed mean of the scores and identifies overconfident agents: those whose scores lie within `overconfidence_margin` (default 0.15) of 0 or 1 and whose divergence from the trimmed mean exceeds `instability_factor` (default 2.0) times the ensemble standard deviation. Such agents are down-weighted. The module also flags agents that are unstable when the ensemble exhibits high variance. The report fields include `trimmed_mean`, `stdev`, `overconfident_count`, `unstable_count`, and per-agent adjustment factors.

Both verifiers are independently configurable through `config/config.yaml` and can be disabled without affecting the base aggregation path.

## C. Face Preprocessing Backend

`app/preprocessing/face_pipeline.py` now contains `OpenCVFaceDetector`, a lightweight wrapper around OpenCV’s Haar-cascade frontal-face classifier. The detector is loaded at initialization; if OpenCV is unavailable or the cascade file is missing, the class reports `available=False` and the pipeline falls back to the original center-crop heuristic. The fallback heuristic assumes the face occupies the central 60% of the image and crops to a configurable target size (default 224×224) using Lanczos resampling.

This change improves the practical usability of the preprocessing stage on images where faces are approximately frontal and centered, while preserving deterministic behavior when no face detector is present.

# III. Updated System Architecture

The implemented system remains a FastAPI application with WebSocket support. The core modules are:

- **API layer** (`app/main.py`, `app/api/routes.py`, `app/api/websocket.py`): exposes `/api/analyze`, `/api/result/{job_id}`, `/api/jobs`, and `/ws/{job_id}`.
- **Preprocessing** (`app/preprocessing/face_pipeline.py`): OpenCV Haar-cascade backend with center-crop fallback; configurable face size and margin.
- **Detection agents** (`app/agents/`): `CNNClassifierAgent`, `ViTClassifierAgent`, `FrequencyAgent`, `EmbeddingAnomalyAgent`, `FaceXrayLikeAgent`.
- **Verifiers** (`app/verifiers/`): `EvidenceVerifier`, `RobustnessVerifier`.
- **Core coordination** (`app/core/`): `BaseAgent`, `Orchestrator`, `MessageBus`.
- **Model layer** (`app/models/schemas.py`): Pydantic models for task, result, error, job, and WebSocket event types; extended with `evidence_verifier_report` and `robustness_verifier_report` fields in `AggregatedResult`.
- **Persistence** (`app/storage/job_store.py`): SQLite-backed or in-memory job storage.
- **Configuration** (`app/config.py`, `config/config.yaml`): YAML-driven parameters, now including a `verifier` section for tuning thresholds and enabling/disabling each verifier independently.

Figure 1 illustrates the request flow: image upload → face preprocessing → concurrent dispatch to five detectors → collection of results → Evidence verifier → Robustness verifier → weighted aggregation → final verdict with structured report → WebSocket streaming.

# IV. Verification and Testing

The repository now contains unit tests for aggregation arithmetic, message-bus isolation, timeout handling, and both verifier modules. As of the current HEAD, the test suite comprises 22 passing cases under `pytest` (Python 3.12, `.venv312`):

- Aggregation: weighted-average calculation, quorum detection, uncertainty computation, verdict thresholding, confidence calculation, empty-results handling, single-agent handling.
- Message bus: pub/sub, multiple subscribers, idempotent handling, event queue registration, unsubscribe, handler error isolation.
- Evidence verifier: outlier detection, consensus preservation, down-weighting toward median, plausibility flagging, empty-input handling.
- Robustness verifier: overconfidence down-weighting, metric reporting, empty-input handling.

These tests demonstrate that the arithmetic and control-flow behavior of the new modules is correct for hand-crafted inputs. The test fixtures use numeric scores and do not require trained models or real images, which keeps the suite fast and environment-independent.

Remaining unverified aspects include:
- End-to-end execution of the full async pipeline with a real image.
- Timeout-monitor behavior under slow or hanging agents.
- WebSocket event delivery to a connected client.
- Face preprocessing quality on non-centered or multi-face images.
- Detection-head performance on a curated evaluation set.

# V. Current Limitations

| Limitation | Technical consequence |
|------------|----------------------|
| Randomly initialized detection heads | Detectors output near-random scores; no meaningful fake/real discrimination is demonstrated. |
| Heuristic face preprocessing | Off-center, small, or multi-face images will likely produce poor crops despite the OpenCV backend. |
| Unused dependencies | `insightface`, `opencv-python`, `facenet-pytorch` appear in `requirements.txt` but are not all wired into the runtime path. |
| No evaluation corpus | AUC, F1, accuracy, and calibration measures cannot be reported. |
| Static-image-only pipeline | Video and audio modalities are not supported despite `Modality` enum preparation. |
| macOS/GPU port incomplete | The system currently targets a CUDA-style path; Metal/MPS support on the development machine is pending. |
| No corpus bibliography in repo | `/papers` and `outputs/papers_bibliography.csv` are referenced in Stage I but absent from tracked files. |

The verifier gap noted in Stage I is now closed. The dominant technical risk remains the absence of trained weights and a documented evaluation set.

# VI. Updated Semester-Based Roadmap

The Stage I roadmap is reaffirmed with refined milestones that reflect the current codebase state:

- **Semester 2** — Train and calibrate detector weights on a documented deepfake dataset. Replace random initializations with learned parameters. Evaluate using standard metrics (AUC, EER, F1) and calibration measures. Freeze reproducibility artifacts (seeds, configs, dataset splits).
- **Semester 3** — Add multimodal capability (audio + video). Integrate MTCNN or RetinaFace via `insightface` for landmark-aware face alignment. Implement temporal modeling for video and spectral features for audio. Design late or verifier-guided fusion.
- **Semester 4** — Implement security and provenance extensions: metric randomization, adversarial hardening, and tamper-evident report hashing. Complete the macOS/MPS port for reproducible development on the target machine.

# VII. Conclusion

Stage II delivered concrete engineering progress: the Evidence and Robustness verifiers are implemented, tested, and integrated; the preprocessing pipeline has a real face-detection backend with graceful fallback; and the unit-test suite has grown to 22 cases. These advances narrow the gap between the Stage I blueprint and the operational prototype.

The next critical phase shifts from structural development to learning-based validation. Training detector heads on a curated dataset, establishing evaluation protocols, and porting the stack to the development machine are the prerequisites for a scientifically defensible final dissertation.

# References

[1] R. Tolosana et al., "Deepfakes and beyond: A survey of face manipulation and fake detection," *Information Fusion*, vol. 64, pp. 131–148, 2020.
[2] Y. Li and L. S. Davis, "Towards open-world deepfake detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2023, pp. 3479–3488.
[3] S. N. metamax, "Mandrake: A fault-tolerant decentralized multiagent system," *Autonomous Agents and Multi-Agent Systems*, vol. 35, no. 2, pp. 1–34, 2021. [Online]. Available: https://doi.org/10.1007/s10458-021-09540-8
