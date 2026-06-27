# Outline — Stage 2 Dissertation Progress Paper

## Proposed Title

Stage II — Evolution and Development of the DeepFake Detector Agent After the Initial Delivery

## Objective of Stage 2

- Describe what existed at the first dissertation delivery (January 15 / February 11 baseline).
- Document what changed in the repository from that point until the present.
- Explain technical and academic significance of observed progress.
- Identify gaps between the dissertation’s stated architecture and the actual repository code.
- Provide a transparent next-work roadmap aligned with the semester-based plan from Part I.

## I. Introduction

- Context: rising synthetic media threat and need for robust, interpretable detection.
- Brief restatement of the problem and of the Mandrake-inspired design choice.
- Purpose of Stage II: bridge between the Part I framework and the next implementation milestones.

## II. Initial Project State at the End of Stage I

- Summarize the 8-component architecture as planned in Part I.
- List the five detector families, two verifier modules, and the aggregator.
- State the intended Mandrake patterns (Remind, Checkpoint, Continue).
- Note the planned semester roadmap.

## III. Methodology for Progress Analysis

- Repository archaeology: baseline commit identification, diff analysis, commit grouping.
- Limitations of Git-based progress tracking (single-commit repository).
- File-by-file mapping of implemented components against the Part I specification.

## IV. Repository Evolution After February 11

- **Finding:** No incremental commits exist after the initial delivery.
- Explanation of what this means for progress reconstruction.
- Present the full snapshot as the current state.

## V. Updated System Architecture

- Present the actual architecture observed in the code.
- Note the discrepancy: 5 detectors + aggregator implemented; 2 verifiers absent.
- Data-flow diagram (Mermaid or textual) showing input → preprocessing → agents → orchestrator → verdict → UI.

## VI. Developed and Improved Functional Components

- **Preprocessing:** Center-crop face extractor vs. planned MTCNN/retinaface.
- **Detection Agents:** CNN, ViT, Frequency, Embedding, FaceXray — describe each concisely.
- **Fault Tolerance:** Remind (timeout + retry), Checkpoint (SQLite persistence), Continue (quorum 3/5).
- **Aggregation:** Weighted average, uncertainty-variance confidence, thresholded verdict, explanation generator.
- **API Layer:** REST + WebSocket endpoints.
- **UI Layer:** Vercel-themed frontend with live agent cards.
- **Testing:** pub/sub, idempotence, weighted average, quorum, uncertainty unit tests.

## VII. Relevance of the Progress to the Dissertation Topic

- Connection to multi-agent fault-tolerant design (Mandrake paper).
- Relevance of heterogeneous detector families (representation, frequency, embedding, boundary).
- Role of streaming evidence for interpretability.
- Importance of quorum-based aggregation for operational safety.

## VIII. Verification, Testing, and Reproducibility

- Current test coverage and its limits.
- Missing validation: no end-to-end integration test, no dataset, no benchmark.
- Reproducibility status: config is versioned; weights are not trained.
- Planned reproducibility artifacts (configs, seeds) from Roadmap Semester 2.

## IX. Current Limitations

- Verifier modules not implemented.
- Pretrained weights are random — no meaningful detection capability demonstrated.
- Face preprocessing is heuristic, not detection-based.
- No literature corpus (`/papers`, bibliography CSV) in the repository.
- No evaluation data or metrics.
- No macOS port (CUDA-only device preference).
- No multimodal extension.

## X. Future Work

- Implement missing verifier modules (Evidence, Robustness).
- Replace random head initializations with trained weights.
- Curate dataset splits and document provenance.
- Run ablations and robustness suite.
- Add real face detection (MTCNN/RetinaFace).
- Remove unused dependencies or wire them in.
- Expand to video (temporal consistency, motion/physiology).
- Add audio signals (spectral artifacts, speaker consistency).
- Add security extensions (metric randomization, adversarial hardening).
- Add tamper-evident reporting (blockchain anchoring prototype).
- Port to macOS / Metal (MPS) for M2 Max development.

## XI. Conclusion

- Stage II reveals a functional prototype with solid architectural foundations but a significant gap between planned and implemented functionality.
- The next priority is closing the verifier gap and moving from prototype to validated system.

## Proposed Tables

- Table 1: Detector comparison (model, input size, signal family).
- Table 2: Mandrake pattern implementation status.
- Table 3: Unit test coverage matrix.
- Table 4: Dissertation roadmap alignment — implemented vs. planned.

## Proposed Figures

- Figure 1: System architecture diagram (Mermaid flowchart).
- Figure 2: Data-flow sequence for a single analysis job.
- Figure 3: Git timeline (will be minimal; can be a table if no branching history).
- Figure 4: Roadmap Gantt-style (text table) mapping semesters to components.

## Items [TO BE COMPLETED]

- Actual benchmark results after evaluation is run.
- Trained model performance metrics.
- User study or qualitative feedback on UI.
- Conference/hackathon award confirmation (if applicable).
- Final hardware port (macOS / Metal) verification.
- Bibliography CSV export from `/papers`.
