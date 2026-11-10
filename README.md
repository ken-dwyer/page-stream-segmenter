# Page Stream Segmenter (PSS)

> **Autonomous, Low-Latency Multi-Page Document Boundary Detection & Splitting Engine**  
> Evaluated on the DocSplit and TABME++ benchmarks. Designed for 100% private, edge execution on consumer NVIDIA GPUs (8 GB VRAM) with zero cloud dependencies.

---

## 1. Overview

**Page Stream Segmentation (PSS)** solves a ubiquitous enterprise document AI challenge: ingesting continuous, unsorted multi-page scanned PDF streams (e.g., mailroom document batches, court discovery packets, loan packages, insurance claims) and splitting them into discrete, atomic subdocuments.

Splitting errors incur severe real-world costs:
* **False Negatives (Missed Splits)** merge distinct statutory, legal, or medical records, creating severe compliance, data privacy (HIPAA/GDPR), and transactional errors.
* **False Positives (Over-Splits)** shatter multi-page contracts, invoices, or forms into orphaned fragments, breaking downstream indexing, ERP ingestion, and RAG pipelines.

`page-stream-segmenter` solves this using a **Cascaded Waterfall Architecture** that resolves over 90% of transitions in under 2 milliseconds, escalating only ambiguous transitions to local edge language models.

---

## 2. Architecture: Cascaded Waterfall Pipeline

```
Page Stream Input (t-1 -> t)
           │
           ▼
┌──────────────────────────────────────┐
│ Stage 1: Explicit Structural Rules   │──[Page 1 of N, "PAGE 1"]──► Direct Decision (<0.01 ms)
└──────────────────────────────────────┘
           │ (Unresolved ~45%)
           ▼
┌──────────────────────────────────────┐
│ Stage 1.5: NuExtract-v1.5 Gate       │──[Doc ID shift / Title shift]──► Pass to Stage 3 Review
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Stage 2: Supervised GBDT (30 Feats)  │──[P < 0.40: High Confidence CONT]──► Fast-Path (<2 ms)
└──────────────────────────────────────┘
           │ (Ambiguous P in [0.40, 0.70] ~10%)
           ▼
┌──────────────────────────────────────┐
│ Stage 3: Edge LLM Adjudicator        │──[Gemma-4-E4B / Gemma-26B MoE]──► Final Boundary Decision
└──────────────────────────────────────┘
```

### Stage Breakdown:
1. **Stage 1 (Structural Fast-Path)**: Evaluates deterministic page numbering regexes (`Page 1 of 5`, `PAGE 1`, `Sheet 1`). Resolves ~50% to 55% of all pages with 0% error in microseconds.
2. **Stage 1.5 (NuExtract-v1.5 Semantic Discontinuity Gate)**: Extracts structured document headers (`document_type`, `document_title`, `document_id`). Includes two critical heuristics:
   - **Sequential Bates Stamp Filter**: Detects incrementing legal discovery stamps (`HCOR-000735 -> HCOR-000736`) and suppresses false splits.
   - **Active Document Title Tracking**: Locks repeated multi-page document headers as continuations.
   - Detects true ID or title transitions between identical document classes that fool spatial models.
3. **Stage 2 (Supervised 30-Feature GBDT Classifier)**: Computes full-text and header TF-IDF cosine similarities, vertical whitespace layout, font line heights, and grammar continuity. Calibrated with an asymmetric threshold (`P >= 0.40`) to aggressively prioritize Recall over Precision.
4. **Stage 3 (Edge LLM Adjudicator)**: Powered by `Gemma-4-E4B-it` (or distributed `Gemma-4-26B-A4B MoE`). Uses domain-specific anti-split prompts (`prompts/adjudicator_v2_anti_split.txt`) to adjudicate complex ambiguous seams (exhibits, ledgers, multi-page form continuations).

---

## 3. Evaluation Metrics Engine

The engine computes five rigorous evaluation metrics across three structural granularities:

* **Boundary Precision, Recall, and F1**: Binary transition-level classification between page $(t-1)$ and page $t$. Evaluates raw split detection accuracy.
* **Document-Level Exact Match F1 (Doc Exact F1)**: Requires both the exact start page and exact end page of an extracted document to match ground truth. Highly punitive: a single boundary mistake penalizes multiple documents.
* **Packet Straight-Through Processing (Packet STP %)**: Percentage of multi-page packets segmented with 100% perfection across all transitions. Direct measure of business automation ROI.
* **Beeferman's $P_k$ Error Rate**: Moving window probe metric ($k = \text{doc\_len} / 2$). Measures probability that two pages distance $k$ apart are misclassified. Rewards near misses.
* **WindowDiff Error Rate (Pevzner & Hearst)**: Corrects $P_k$'s false negative blindness by penalizing when the exact number of boundaries within window $k$ differs between prediction and truth.

---

## 4. Benchmark Results

Evaluated on the 15-packet `DocSplit` sequential benchmark:

| Pipeline Configuration | Latency / Transition | Boundary Recall | Doc Exact F1 | Packet STP % | WindowDiff Error |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1 (Rules Only)** | < 0.01 ms | 70.0% | 40.0% | 20.0% | 15.6% |
| **Stage 1 + Stage 2 (GBDT)** | 1.8 ms | 90.0% | 63.2% | 40.0% | 8.9% |
| **Waterfall + Gemma-4-E4B (v1 Prompt)** | 420 ms | 93.3% | 76.9% | 60.0% | 6.2% |
| **Waterfall + Gemma-4-E4B (v2 Anti-Split)** | 480 ms | 96.7% | 84.6% | 73.3% | 4.8% |
| **Waterfall + Stage 1.5 NuExtract + Gemma** | 560 ms | **100.0%** | **90.3%** | **80.0%** | **3.1%** |

*(Detailed leaderboard and packet breakdowns available in [PSS_DOCSPLIT_LEADERBOARD.md](PSS_DOCSPLIT_LEADERBOARD.md)).*

---

## 5. Hardware Sizing & Local Deployment

### Single Laptop (NVIDIA RTX 4060 8 GB VRAM)
Both models are served concurrently using `llama-server.exe` with CUDA offloading:
* **Port 8080**: `gemma-4-E4B-it-Q5_K_M.gguf` (3,350 MiB VRAM)
* **Port 8081**: `nuextract-v1.5-Q8_0.gguf` (2,900 MiB VRAM)
* **Total VRAM**: ~6,250 MiB out of 8,188 MiB (~1.9 GB headroom).

### Distributed Multi-GPU Cluster (`llama.cpp` RPC)
For serving larger models (e.g. `Gemma-4-26B-A4B-it-UD-IQ3_S.gguf`, 11.28 GB):
```powershell
python pss_pipeline.py --rpc 192.168.1.91:50052 --model Gemma-4-26B-A4B-IT-MoE ...
```
Automatically distributes tensor layers across the local GPU and a remote worker over Gigabit LAN.

---

## 6. Installation & Quickstart

### Setup
```bash
git clone https://github.com/your-username/page-stream-segmenter.git
cd page-stream-segmenter
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Splitting a Scanned PDF Packet
```bash
pss-split --input ./scripts/test_data/sample_multi_doc_packet.pdf --output-dir ./output/
```

### Running Benchmark Tournaments
```bash
# Verify GBDT + Rule performance
python run_pss_benchmark_tournament.py --dataset docsplit --dry-run

# Run full tournament with local LLM adjudicator
python run_pss_benchmark_tournament.py --dataset docsplit --server-url http://127.0.0.1:8080/v1
```

### Local Arize Phoenix Observability
Start the Phoenix telemetry server:
```bash
python -m phoenix.server.main serve
```
Open `http://127.0.0.1:6006` in your browser. Every page boundary evaluation is recorded with token inputs, GBDT probabilities, and LLM reasoning spans.

---

## 7. Dataset Management Policy

> [!NOTE]
> Benchmark evaluation datasets (`data/*.json`, `data/*.pdf`, `data/*.parquet`) are excluded from git version control via `.gitignore`.
> To populate the datasets locally, use the provided loader script:
> ```bash
> python prepare_pss_benchmarks.py
> ```

---

## 8. License

Apache 2.0
