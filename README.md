# EchoShield v2 (offline) — proof-carrying answers

EchoShield v2 is a fully offline, multilingual chatbot that runs locally (no network calls).
It improves robustness and auditability via:

- **self-consensus decoding** (multi-candidate generation + agreement scoring)
- optional **PAA** (paraphrase-and-agree) using offline back-translation
- optional **C3** (counterfactual consistency) to reduce ambiguity
- optional **critic/governor** to reduce redundancy and improve clarity
- a local **BM25 knowledge base** (KB) and **proof-carrying answers (PCA)**
  - each sentence can be tagged: `(kb: supported)`, `(kb: contradicted)`, `(kb: unverified)`
  - evidence is retrieved from the local KB
  - optionally verified with an offline **NLI entailment** model (MNLI)
- **prompt-injection shield** (removes common hijack patterns + invisible unicode)
- **offline safety filter** (regex + optional MNLI zero-shot)

> no internet is used at runtime: all `transformers` loads use `local_files_only=True`.

---

## Requirements

### Python
- Python 3.9+

### Python packages
- torch
- transformers
- langdetect

Install (on a connected machine if needed):
```bash
pip install torch transformers langdetect
