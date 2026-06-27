# Stage 2 Review Checklist

## Content Accuracy

- [ ] Every technical claim maps to a repository file, commit, or existing dissertation text.
- [ ] No invented metrics (AUC, F1, accuracy, training duration).
- [ ] No hallucinated benchmark results.
- [ ] No fabricated dataset sizes or sample counts.
- [ ] No unsupported "robust", "highly accurate", or "state-of-the-art" claims.
- [ ] Missing facts are marked `[TO BE COMPLETED]`.

## Repository Alignment

- [ ] The git forensics section accurately reflects the single-commit reality.
- [ ] The baseline commit date and hash are correct.
- [ ] File paths and line numbers referenced in the paper match actual repository content.
- [ ] The discrepancy between the dissertation Part I "8-component pipeline" and the actual codebase (5 detectors + aggregator, no verifiers) is explicitly acknowledged.

## IEEE Style Compliance

- [ ] Sections follow Part I numbering and hierarchy.
- [ ] Abstract is IEEE-style (~150–250 words, structured).
- [ ] Index Terms are included.
- [ ] Citations use IEEE numeric style `[1]`, `[2]`, etc.
- [ ] Figures and tables have IEEE-style captions (e.g., "Fig. 1. Architecture.")
- [ ] Terminology is consistent with Part I (agents, aggregator, verifier, Mandrake patterns).
- [ ] English is formal, concise, and free of slang.

## Dissertation Structure

- [ ] I. Introduction
- [ ] II. Initial Project State at the End of Stage I
- [ ] III. Methodology for Progress Analysis
- [ ] IV. Repository Evolution After February 11
- [ ] V. Updated System Architecture
- [ ] VI. Developed and Improved Functional Components
- [ ] VII. Relevance of the Progress to the Dissertation Topic
- [ ] VIII. Verification, Testing, and Reproducibility
- [ ] IX. Current Limitations
- [ ] X. Future Work
- [ ] XI. Conclusion

## Future Work Separation

- [ ] Planned but unimplemented features are clearly labeled as Future Work, not as delivered.
- [ ] The roadmap table aligns with Part I Semester 2–4 plans.
- [ ] macOS port is mentioned as Future Work unless repository evidence proves otherwise.
- [ ] No part-written or half-implemented feature is presented as complete.

## Evidence Log Quality

- [ ] Every stakeholder-visible claim in the paper has a corresponding row in `stage2_evidence_log.md`.
- [ ] Confidence levels are set honestly (High / Medium / Low / Missing).
- [ ] The evidence log references exact files and line ranges where applicable.

## Internal Consistency

- [ ] The paper draft, outline, progress summary, and evidence log do not contradict each other.
- [ ] Detector names, model names, and acronyms are identical in all documents.
- [ ] Config values (`timeout_seconds`, `max_retries`, `quorum`, weights) match between `config.yaml`, `app/config.py`, and the dissertation text.

## Minimal Editing Readiness

- [ ] The draft requires only minor copy-editing (spacing, punctuation) to be usable in the dissertation.
- [ ] Section headers can be copy-pasted into the main `.docx` template.
- [ ] Tables and figures are provided in formats suitable for insertion (Markdown or Mermaid).

## Outstanding Placeholders

- [ ] `[TO BE COMPLETED]` search performed and all remaining placeholders documented in the outline.
- [ ] No placeholder text remains in the final paper draft without an accompanying explanation in the outline or evidence log.
