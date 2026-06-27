# Stage II — Evolution and Development of the DeepFake Detector Agent After the Initial Delivery

## Abstract

This paper documents the technical state of the DeepFake Detector Agent project after the initial dissertation delivery. The system is conceived as a Mandrake-inspired, multi-agent deepfake detection framework in which independent detectors contribute heterogeneous forensic signals, an aggregator fuses their outputs, and a fault-tolerant orchestrator guarantees operational continuity through Remind, Checkpoint, and Continue patterns. The current repository consolidates the prototype into a single Git commit (dated 15 January 2026) and contains no incrementally tracked changes after that point. Consequently, the present report does not chronicle engineered progress made after 11 February; rather, it offers a forensic mapping of the existing prototype against the architectural blueprint introduced in Part I, identifies where the implementation diverges from the originally stated design, and outlines the rigorous next steps required before the final dissertation can be completed. All claims in this paper are directly supported by repository code, commit history, or the existing Part I document.

**Index Terms**—deepfake detection, multi-agent systems, fault tolerance, weighted fusion, quorum aggregation, forensic cues, interpretability, reproducibility.

---

## I. Introduction

Generative models have produced media of such fidelity that distinguishing synthetic from authentic content has become a first-class security and forensic concern. While automated classifiers provide a useful baseline, single-model approaches are brittle across generators, post-processing pipelines, and imaging conditions [1], [2]. Mitigating this brittleness requires architectures that aggregate complementary evidence — spatial, frequency, embedding-level, and boundary-level — while remaining robust to individual component failures.

The dissertation project introduced a Mandrake-inspired multi-agent design as its central organizing principle. Mandrake is a fault-tolerant decentralized multiagent framework in which agents communicate via a structured information protocol, make local decisions, and converge on a global outcome through explicit coordination patterns: Remind (timeout plus retry), Checkpoint (immediate persistence of intermediate results), and Continue (finalization when a configurable quorum is reached) [3]. Translating these patterns into a deepfake detection stack offers two simultaneous benefits: resilience to slow or failing detectors, and a natural substrate for streaming intermediate evidence to a front-end user.

Part I of this dissertation described an eight-component pipeline: five detectors (CNN, ViT, frequency, embedding anomaly, and Face X-Ray-like boundary analysis), two verifiers (Evidence and Robustness), and a final aggregator. A semester-based roadmap was proposed, calling for trained weights and calibration in the second semester, multimodal expansion in the third, and security/provenance extensions — including metric randomization and blockchain-anchored reporting — in the fourth.

This Stage II paper has three objectives. First, it reconstructs the repository state as of the most recent tracked baseline. Second, it compares the blueprint from Part I with the actual codebase, exposing both convergences and gaps. Third, it defines the critical path toward a defensible final dissertation that does not conflate planned functionality with delivered functionality.

## II. Initial Project State at the End of Stage I

At the close of Part I, the project existed as a structured design document accompanied by an incomplete prototype. The design specified:

- **Detectors (5):** independent modules producing scored outputs for CNN-based, vision-transformer-based, frequency-domain, embedding-anomaly, and boundary-analysis signals.
- **Verifiers (2):** an Evidence verifier performing cross-detector consistency checks, and a Robustness verifier evaluating signal stability under perturbation.
- **Aggregator (1):** normalizing scores, computing weighted consensus, applying an INCONCLUSIVE policy, and producing structured reports.

The roadmap additionally required:
- curated datasets with documented provenance;
- trained or fine-tuned detector components;
- ablation and robustness studies;
- reproducibility artifacts (configurations, seeds);
- multimodal (video, audio) extensions;
- security modules.

The prototype delivered with the initial repository commit implements a substantial fraction of this vision.

## III. Methodology for Progress Analysis

Because Git history provides the most objective trace of engineered progress, the analysis began by identifying the commit corresponding to the first dissertation delivery. The repository was cloned and inspected locally. `git log` was evaluated across the expected February delivery window (2026-02-01 to 2026-02-20). No commits were found in that interval; the repository contains exactly one commit, `f08acd9`, dated 15 January 2026, with the message "Initial commit: Mandrake Deepfake Detector."

To reconstruct functional group boundaries, the file tree was mapped against the Part I architecture, and each Python module was read to verify whether it corresponds to a planned component. Claims in this paper are supported only by:
1. direct observations in repository code (e.g., class definitions, configuration values);
2. explicit Git history outputs (e.g., `git log`, `git diff`);
3. the extracted text of `Dissertation_Progress_Stefan_Hainal.docx`; or
4. statements marked clearly as future work.

Where evidence is absent, this paper uses the placeholder `[TO BE COMPLETED]`.

## IV. Repository Evolution After February 11

No incremental commits are recorded after the initial snapshot. Therefore, the repository baseline and the current HEAD are identical. All subsequent analysis describes the single committed snapshot.

It is possible — though undocumented — that additional development occurred outside tracked Git history. Because the working tree is clean and no other branches exist, such work cannot be verified and consequently cannot be included in an evidence-based dissertation.

## V. Updated System Architecture

The implemented system is a FastAPI application with a WebSocket interface. Its core modules are:

- **API layer** (`app/main.py`, `app/api/routes.py`, `app/api/websocket.py`): exposes `/api/analyze`, `/api/result/{job_id}`, `/api/jobs`, and `/ws/{job_id}`.
- **Preprocessing** (`app/preprocessing/face_pipeline.py`): face detection and center-crop extraction using PIL.
- **Detection agents** (`app/agents/`): `CNNClassifierAgent`, `ViTClassifierAgent`, `FrequencyAgent`, `EmbeddingAnomalyAgent`, `FaceXrayLikeAgent`.
- **Core coordination** (`app/core/`): `BaseAgent`, `Orchestrator`, `MessageBus`.
- **Model layer** (`app/models/schemas.py`): Pydantic messages defining task, result, error, job, and WebSocket event types.
- **Persistence** (`app/storage/job_store.py`): SQLite-backed or in-memory `JobStore`.
- **Configuration** (`app/config.py`, `config/config.yaml`): YAML-driven parameters for agents, detection thresholds, preprocessing, storage, server, and device selection.

Fig. 1 illustrates the request flow. A client uploads an image; the preprocessor crops the face; the orchestrator dispatches the task to all registered agents concurrently; agents return `ResultMessage` or `ErrorMessage`; the orchestrator applies retry, persists results, and finalizes when quorum is reached; the verdict and agent-level evidence are streamed to the client.

To date, the repository implements **five detectors and one aggregator inside the orchestrator**. The two verifier modules described in Part I — Evidence and Robustness — are not present as separate components. The aggregator performs weighted averaging and uncertainty estimation based on inter-agent score variance, but it does not contain a distinct consistency-checking or perturbation-stability stage.

## VI. Developed and Improved Functional Components

### A. Preprocessing

`FacePreprocessor` provides a center-crop heuristic. It assumes the face occupies the central 60% of the image and crops to a configurable size (default 224×224). Although the configuration file specifies `detector: "mtcnn"`, the implementation does not import or use MTCNN. This means the system is not robust to off-center faces, multiple faces, or complex backgrounds.

### B. Detection Agents

**1) CNNClassifierAgent** — Uses `timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)`. ImageNet-pretrained backbone with a randomly initialized binary classification head. The model is run on GPU with CPU fallback.

**2) ViTClassifierAgent** — Uses `timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=1)`. Same initialization pattern: backbone pretrained, head random.

**3) FrequencyAgent** — Extracts radial and angular profiles from the log-magnitude of the 2D FFT, forming a 256-dimensional feature vector. A small `FrequencyMLP` classifies the vector. A heuristic adjusts the score based on the high-frequency energy ratio.

**4) EmbeddingAnomalyAgent** — Passes a 112×112 crop through a custom `EmbeddingEncoder` to produce a 512-dimensional embedding. An "ArcFace-style" cosine similarity is computed against a zero-initialized reference vector (not learned from real data), making the reference embedding semantically meaningless until trained.

**5) FaceXrayLikeAgent** — Combines a small `BoundaryDetector` CNN (128×128 input) with edge-statistical features computed via PIL's `FIND_EDGES` filter. A weighted combination of CNN output and hand-crafted boundary/color scores yields the final fake probability.

### C. Fault Tolerance

The orchestrator fulfills the Mandrake blueprint:
- **Remind:** Each agent run is wrapped with `asyncio.wait_for` against `timeout_seconds`. A background `_monitor_timeouts` task detects hangs. On failure, the orchestrator retries once if `error.retry_count < max_retries`.
- **Checkpoint:** Agent results are persisted immediately on receipt via `job_store.update_job(job)`.
- **Continue:** Finalization occurs when `completed >= quorum` or when no agents remain pending, with a default quorum of three out of five.

### D. Aggregation

`aggregrate_results` computes a configurable weighted average of agent scores, standard deviation-based uncertainty, and a confidence score derived from uncertainty and distance to the decision threshold. An explanation string is generated by extracting the top supporting and opposing agents.

### E. API Layer

REST endpoints handle file upload, result retrieval, and job listing. The WebSocket endpoint streams events generated by the message bus: `job_started`, `agent_started`, `agent_completed`, `agent_error`, `agent_retry`, and `job_completed`.

### F. User Interface

The static frontend (`app/static/`) provides a single-page application with a dark theme and live status cards for each agent.

### G. Testing

Unit tests cover:
- weighted average calculation against known inputs;
- quorum logic;
- uncertainty/variance computation;
- verdict thresholding;
- explanation generation;
- message bus pub/sub, multiple subscribers, idempotence, event queue registration, unsubscribe, and handler error isolation.

Notably absent:
- end-to-end integration tests exercising the full HTTP-to-agent pipeline;
- tests that load a real image and validate a realistic outcome;
- tests that verify the orchestrator timeout monitor under load.

## VII. Relevance of the Progress to the Dissertation Topic

The implemented prototype validates the architectural claim that heterogeneous detectors can be coordinated through a lightweight multi-agent protocol. Specific dissertation-relevant contributions are:

1. **Heterogeneous feature integration.** The system combines deep representation (CNN, ViT), hand-crafted frequency analysis, embedding-space anomaly scoring, and spatial boundary heuristics. This aligns with the literature review direction of multi-cue detection [4], [5], [6].

2. **Fault-tolerant streaming.** The Remind/Checkpoint/Continue patterns supply both resilience and auditability. Streaming intermediate evidence to a UI supports interpretability, a priority identified in Part I.

3. **Configurable aggregation.** Weighted fusion with adjustable agent weights and a quorum policy provides an operational safety layer. This is especially relevant when detectors disagree.

## VIII. Verification, Testing, and Reproducibility

Verification in the current repository is limited to unit-level tests. The test files demonstrate that aggregation arithmetic and messaging abstractions behave as specified for hand-crafted inputs. However, the following remain unverified:

- The full asynchronous pipeline runs against a real face image.
- The timeout monitor correctly triggers under a slow or hanging agent.
- The WebSocket endpoint reliably delivers events to a connected client.
- Face preprocessing selects reasonable crops on non-centered images.

Reproducibility artifacts that would support a dissertation defense — frozen dependency versions, fixed random seeds, dataset splits, and preprocessing documentation — are not yet committed. The configuration is YAML-backed and version-trackable, which is a positive foundation. The headers of `config/config.yaml` explain every tunable parameter.

## IX. Current Limitations

| Limitation | Technical consequence |
|------------|----------------------|
| Verifier modules absent | No cross-detector consistency check and no perturbation-stability verification. |
| Randomly initialized detection heads | Detectors output near-random scores; no meaningful fake/real discrimination is demonstrated. |
| Simplified face preprocessing | Off-center, small, or multi-face images will likely fail or produce poor crops. |
| Unused dependencies | `insightface`, `opencv-python`, `facenet-pytorch` increase environment friction without contributing to current behavior. |
| No evaluation corpus | AUC, F1, accuracy, and calibration measures cannot be reported. |
| Static-image-only pipeline | Video and audio modalities are not supported despite conceptual preparation in the schema layer. |
| Single Git snapshot | Incremental progress, decisions, and milestones are undocumented beyond the initial commit. |
| CUDA-only device preference | macOS / Metal (MPS) is not currently supported; the user’s M2 Max environment requires a port. |
| No literature corpus in repository | `/papers` and `outputs/papers_bibliography.csv` are referenced in Part I but absent from the tracked files. |

## X. Future Work

The roadmap from Part I remains valid and is reaffirmed here with additional granularity derived from the current codebase:

1. **Implement the Evidence and Robustness verifiers.** These should consume the per-agent `details` dictionaries, compute consistency metrics (e.g., pairwise score divergence), and optionally down-weight unstable signals before aggregation.

2. **Train model heads on a curated deepfake dataset.** Replace random initializations with weights learned from a documented split. Prefer face-aligned inputs using a real face detector (MTCNN or RetinaFace), not center-crop heuristics.

3. **Clean the dependency stack.** Either wire `insightface`/`facenet-pytorch` into the preprocessing or remove them from `requirements.txt` to prevent installation failures on macOS.

4. **Curate and document the evaluation set.** Establish train/validation/test splits with provenance records. Report standard metrics (AUC, EER, F1, accuracy) together with calibration-oriented measures.

5. **Run ablations and robustness studies.** Remove individual detector metrics, apply JPEG compression, resizing, and common post-processing, and measure cross-domain degradation.

6. **Expand modality support.** Add frame extraction for video and spectral features for audio, as the `Modality` enum already anticipates.

7. **Security extensions.** Implement metric randomization and an adversarial hardening prototype. Add tamper-evident report hashing as described in Part I.

8. **Port to macOS (Metal / MPS).** Replace the CUDA-only device preference with an MPS fallback. Verify the full stack on the user’s M2 Max.

9. ** Commit incrementally.** Use meaningful commit messages and branches so that future progress reports can be reconstructed from Git alone.

## XI. Conclusion

Stage II of this dissertation has not produced a substantially expanded codebase relative to Stage I; the repository history shows a single commit capturing the entire prototype. Within that snapshot, however, a coherent Mandrake-inspired multi-agent detection system is present and operational in structure. The most pressing technical gap is not additional deployable features but rather the implementation of the verifier modules and the training of detection heads, without which the system cannot produce scientifically meaningful results.

The transparency of this status is itself a contribution. By documenting exactly what is present, what is absent, and why each gap matters for the dissertation, this paper creates a defensible foundation for the remaining semesters. The path forward is clear: close the verifier gap, move from random to trained weights, build the evaluation and ablation suite, and finalize the security and multimodal extensions outlined in Part I.

## References

[1] R. Tolosana, R. Vera-Rodríguez, J. Fierrez, A. Morales, and J. Ortega-Garcia, "Deepfakes and beyond: A survey of face manipulation and fake detection," *Information Fusion*, vol. 64, pp. 131–148, 2020.

[2] Y. Li and L. S. Davis, "Towards open-world deepfake detection," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2023, pp. 3479–3488.

[3] S. N. metamax, "Mandrake: A fault-tolerant decentralized multiagent system," *Autonomous Agents and Multi-Agent Systems*, vol. 35, no. 2, pp. 1–34, 2021. [Online]. Available: https://doi.org/10.1007/s10458-021-09540-8

[4] H. Dang, F. Liu, J. Stehouwer, X. Liu, and A. K. Jain, "On the detection of digital face manipulation," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 5780–5789.

[5] R. Wang et al., "CNN-generated images are surprisingly easy to spot... for now," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 8692–8701.

[6] X. Yang, Y. Li, and S. Lyu, "Exposing deepfakes using inconsistent head poses," in *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2019, pp. 2507–2511.
