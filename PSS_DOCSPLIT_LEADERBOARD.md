# ✂️ Page Stream Segmentation (PSS) & DocSplit-Seq Benchmark Leaderboard

**Hardware:** Dual-PC Cluster: Laptop RTX 4060 (8GB) + Desktop hpenvy GTX 1660 Super (6GB) via `llama.cpp` RPC  
**Models Evaluated:** Distributed Gemma-26B MoE (`Gemma-4-26B-A4B-it-UD-IQ3_S.gguf`), Gemma-4-E4B-IT, Gemma-4-E2B-IT, Qwen2.5-7B, LFM3-1.7B, Supervised 30F GBDT  
**Benchmarks Evaluated:**
1. **TABME++ (`bevaya/TABMEpp`)** — Official UCSF archival business document folders with Microsoft Azure 2D OCR.
2. **DocSplit-Poly-Seq (`RVL-CDIP-N-MP`)** — Multi-category contiguous packets (invoices, resumes, budgets, memos, letters, forms, etc.).
3. **DocSplit-Mono-Seq (`RVL-CDIP-N-MP`)** — Single-category contiguous packets (multiple multi-page documents of the *exact same class* concatenated back-to-back).

---

## 1. Official TABME++ Benchmark (`bevaya/TABMEpp`)
* **Test Suite Size:** 10 packets/folders | 99 documents | 224 pages

| Rank | Algorithm / Model | Paradigm | Boundary F1 (%) | Boundary P / R (%) | Doc Exact Match F1 (%) | Packet STP (%) | $P_k$ Err (%) ↓ | WindowDiff (%) ↓ | ARI (%) | V-Measure (%) | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Supervised 30-Feature Spatial+Continuity GBDT** | Era 3: Supervised Pairwise ML | **90.32%** | 86.6 / 94.4 | **77.67%** | 20.0% | 7.00% | 13.87% | 85.03% | 95.07% | 5.1s |
| 2 | **Two-Stage Hybrid (GBDT + Distributed Gemma-26B MoE)** | Era 4+: 2-Stage Hybrid Ensemble | **89.12%** | 82.7 / 96.6 | **74.18%** | 20.0% | 6.11% | 15.59% | 87.31% | 95.15% | 189.6s |
| 3 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, LFM3-1.7B (Q8_0))** | Era 4+: 2-Stage Hybrid Ensemble | **85.06%** | 87.1 / 83.2 | **67.01%** | 0.0% | 8.28% | 21.64% | 81.47% | 92.78% | 26.2s |
| 4 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Gemma-4-E2B-IT (Q5_K_M))** | Era 4+: 2-Stage Hybrid Ensemble | **85.06%** | 87.1 / 83.2 | **67.01%** | 0.0% | 8.50% | 21.41% | 81.38% | 92.75% | 24.3s |
| 5 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Qwen2.5-7B-Instruct (Q6_K))** | Era 4+: 2-Stage Hybrid Ensemble | **85.06%** | 87.1 / 83.2 | **67.01%** | 10.0% | 8.50% | 20.52% | 81.97% | 92.86% | 64.3s |
| 6 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Gemma-4-E4B-IT (Q5_K_M))** | Era 4+: 2-Stage Hybrid Ensemble | **84.57%** | 86.0 / 83.2 | **65.64%** | 0.0% | 8.50% | 21.86% | 81.34% | 92.68% | 44.7s |
| 7 | **Holistic Windowed LLM (Distributed Gemma-26B MoE)** | Era 4: Holistic Zero-Shot LLM | **79.80%** | 71.1 / 91.0 | **64.57%** | 10.0% | 15.22% | 28.76% | 58.82% | 90.20% | 195.3s |
| 8 | **Unsupervised TF-IDF + Layout Shift** | Era 2: Unsupervised IR | **77.93%** | 66.9 / 93.3 | **56.65%** | 0.0% | 15.02% | 34.04% | 61.55% | 89.21% | 0.2s |
| 9 | **Two-Stage Pairwise Signature LLM (Distributed Gemma-26B MoE)** | Era 4+: 2-Stage Pairwise LLM | **77.16%** | 70.4 / 85.4 | **52.53%** | 0.0% | 15.55% | 33.87% | 61.92% | 88.64% | 189.6s |
| 10 | **Two-Stage Pairwise Signature LLM (Gemma-4-E4B-IT (Q5_K_M))** | Era 4+: 2-Stage Pairwise LLM | **76.38%** | 69.1 / 85.4 | **53.88%** | 0.0% | 14.66% | 33.02% | 66.53% | 88.77% | 44.7s |
| 11 | **Two-Stage Pairwise Signature LLM (Gemma-4-E2B-IT (Q5_K_M))** | Era 4+: 2-Stage Pairwise LLM | **73.20%** | 67.6 / 79.8 | **47.66%** | 0.0% | 16.69% | 37.78% | 59.91% | 86.97% | 24.3s |
| 12 | **Holistic Windowed LLM (Gemma-4-E4B-IT (Q5_K_M))** | Era 4: Holistic Zero-Shot LLM | **71.57%** | 63.5 / 82.0 | **53.57%** | 20.0% | 18.37% | 35.08% | 58.01% | 86.97% | 23.4s |
| 13 | **Two-Stage Pairwise Signature LLM (Qwen2.5-7B-Instruct (Q6_K))** | Era 4+: 2-Stage Pairwise LLM | **70.27%** | 67.7 / 73.0 | **43.90%** | 0.0% | 18.96% | 37.83% | 60.40% | 85.80% | 64.3s |
| 14 | **Two-Stage Pairwise Signature LLM (LFM3-1.7B (Q8_0))** | Era 4+: 2-Stage Pairwise LLM | **67.52%** | 77.9 / 59.5 | **42.94%** | 0.0% | 21.30% | 37.79% | 60.85% | 83.87% | 26.2s |
| 15 | **Holistic Windowed LLM (Qwen2.5-7B-Instruct (Q6_K))** | Era 4: Holistic Zero-Shot LLM | **56.76%** | 47.4 / 70.8 | **33.06%** | 0.0% | 33.95% | 58.06% | 31.21% | 79.94% | 36.0s |
| 16 | **Holistic Windowed LLM (LFM3-1.7B (Q8_0))** | Era 4: Holistic Zero-Shot LLM | **30.38%** | 34.8 / 27.0 | **13.48%** | 0.0% | 61.75% | 71.80% | 13.04% | 44.24% | 9.3s |
| 17 | **Holistic Windowed LLM (Gemma-4-E2B-IT (Q5_K_M))** | Era 4: Holistic Zero-Shot LLM | **21.36%** | 78.6 / 12.4 | **13.01%** | 0.0% | 56.37% | 61.13% | 4.54% | 17.95% | 7.7s |
| 18 | **Rules + Pagination/Header Heuristic** | Era 1: Heuristic / Regex | **19.23%** | 66.7 / 11.2 | **3.23%** | 0.0% | 56.72% | 60.67% | 17.33% | 35.52% | 0.0s |

---

## 2. DocSplit-Poly-Seq Benchmark (Multi-Category Contiguous Packets)
* **Test Suite Size:** 10 packets/folders | 60 documents | 175 pages

| Rank | Algorithm / Model | Paradigm | Boundary F1 (%) | Boundary P / R (%) | Doc Exact Match F1 (%) | Packet STP (%) | $P_k$ Err (%) ↓ | WindowDiff (%) ↓ | ARI (%) | V-Measure (%) | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 🏆 **Holistic Windowed LLM (Distributed Gemma-26B MoE)** | Era 4: Holistic Zero-Shot LLM | **94.85%** | 97.9 / 92.0 | **87.18%** | **50.0%** | **3.73%** | **6.51%** | **96.18%** | **97.84%** | 141.8s |
| 2 | **Two-Stage Hybrid (GBDT + Distributed Gemma-26B MoE)** | Era 4+: 2-Stage Hybrid Ensemble | **89.11%** | 88.2 / 90.0 | **77.69%** | **50.0%** | 6.19% | 11.94% | 90.09% | 95.10% | 163.4s |
| 3 | **Supervised 30-Feature Spatial+Continuity GBDT** | Era 3: Supervised Pairwise ML | **86.87%** | 87.8 / 86.0 | **73.95%** | **50.0%** | 7.91% | 13.80% | 86.77% | 93.22% | 1.8s |
| 4 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, LFM3-1.7B (Q8_0))** | Era 4+: 2-Stage Hybrid Ensemble | **83.67%** | 85.4 / 82.0 | **66.10%** | 40.0% | 9.41% | 19.26% | 84.95% | 92.66% | 20.2s |
| 5 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Gemma-4-E2B-IT (Q5_K_M))** | Era 4+: 2-Stage Hybrid Ensemble | **83.67%** | 85.4 / 82.0 | **66.10%** | 40.0% | 9.41% | 19.26% | 84.95% | 92.66% | 21.4s |
| 6 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Gemma-4-E4B-IT (Q5_K_M))** | Era 4+: 2-Stage Hybrid Ensemble | **83.67%** | 85.4 / 82.0 | **66.10%** | 40.0% | 9.41% | 19.26% | 84.95% | 92.66% | 40.6s |
| 7 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Qwen2.5-7B-Instruct (Q6_K))** | Era 4+: 2-Stage Hybrid Ensemble | **83.67%** | 85.4 / 82.0 | **66.10%** | 40.0% | 9.41% | 19.26% | 84.95% | 92.66% | 57.5s |
| 8 | **Two-Stage Pairwise Signature LLM (Distributed Gemma-26B MoE)** | Era 4+: 2-Stage Pairwise LLM | **78.90%** | 72.9 / 86.0 | **55.81%** | 20.0% | 16.68% | 26.14% | 73.81% | 90.20% | 163.4s |
| 9 | **Holistic Windowed LLM (Gemma-4-E4B-IT (Q5_K_M))** | Era 4: Holistic Zero-Shot LLM | **74.14%** | 65.2 / 86.0 | **52.94%** | 0.0% | 16.80% | 26.09% | 78.09% | 89.44% | 17.8s |
| 10 | **Unsupervised TF-IDF + Layout Shift** | Era 2: Unsupervised IR | **71.32%** | 58.2 / 92.0 | **46.98%** | 0.0% | 21.20% | 38.66% | 63.21% | 86.02% | 0.2s |
| 11 | **Two-Stage Pairwise Signature LLM (Gemma-4-E4B-IT (Q5_K_M))** | Era 4+: 2-Stage Pairwise LLM | **74.14%** | 65.2 / 86.0 | **45.59%** | 0.0% | 18.10% | 33.26% | 66.68% | 87.71% | 40.6s |
| 12 | **Two-Stage Pairwise Signature LLM (Gemma-4-E2B-IT (Q5_K_M))** | Era 4+: 2-Stage Pairwise LLM | **69.42%** | 59.1 / 84.0 | **39.72%** | 0.0% | 25.35% | 38.32% | 57.40% | 84.24% | 21.4s |
| 13 | **Two-Stage Pairwise Signature LLM (Qwen2.5-7B-Instruct (Q6_K))** | Era 4+: 2-Stage Pairwise LLM | **63.08%** | 51.2 / 82.0 | **32.00%** | 0.0% | 29.61% | 49.09% | 48.30% | 81.27% | 57.5s |
| 14 | **Two-Stage Pairwise Signature LLM (LFM3-1.7B (Q8_0))** | Era 4+: 2-Stage Pairwise LLM | **48.48%** | 49.0 / 48.0 | **25.21%** | 0.0% | 39.76% | 46.65% | 44.09% | 73.66% | 20.2s |
| 15 | **Holistic Windowed LLM (Qwen2.5-7B-Instruct (Q6_K))** | Era 4: Holistic Zero-Shot LLM | **53.54%** | 44.2 / 68.0 | **24.49%** | 0.0% | 38.29% | 54.58% | 38.85% | 76.72% | 24.7s |
| 16 | **Holistic Windowed LLM (Gemma-4-E2B-IT (Q5_K_M))** | Era 4: Holistic Zero-Shot LLM | **30.00%** | 90.0 / 18.0 | **20.00%** | 0.0% | 42.92% | 44.92% | 24.17% | 32.95% | 5.7s |
| 17 | **Rules + Pagination/Header Heuristic** | Era 1: Heuristic / Regex | **35.48%** | 91.7 / 22.0 | **17.07%** | 0.0% | 44.73% | 45.75% | 22.09% | 40.18% | 0.0s |
| 18 | **Holistic Windowed LLM (LFM3-1.7B (Q8_0))** | Era 4: Holistic Zero-Shot LLM | **20.00%** | 60.0 / 12.0 | **5.00%** | 0.0% | 51.84% | 52.61% | 17.00% | 32.83% | 5.6s |

---

## 3. DocSplit-Mono-Seq Benchmark (Same-Category Contiguous Packets — Hardest Case)
* **Test Suite Size:** 10 packets/folders | 50 documents | 187 pages

| Rank | Algorithm / Model | Paradigm | Boundary F1 (%) | Boundary P / R (%) | Doc Exact Match F1 (%) | Packet STP (%) | $P_k$ Err (%) ↓ | WindowDiff (%) ↓ | ARI (%) | V-Measure (%) | Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 🏆 **Holistic Windowed LLM (Distributed Gemma-26B MoE)** | Era 4: Holistic Zero-Shot LLM | **88.89%** | 80.0 / 100.0 | **80.00%** | **50.0%** | **7.41%** | **10.78%** | **89.05%** | **94.71%** | 147.0s |
| 2 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Gemma-4-E2B-IT (Q5_K_M))** | Era 4+: 2-Stage Hybrid Ensemble | **77.89%** | 67.3 / 92.5 | **62.61%** | 40.0% | 9.59% | 17.29% | 86.48% | 91.61% | 19.7s |
| 3 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Gemma-4-E4B-IT (Q5_K_M))** | Era 4+: 2-Stage Hybrid Ensemble | **77.89%** | 67.3 / 92.5 | **62.61%** | 40.0% | 9.59% | 17.29% | 86.48% | 91.61% | 38.0s |
| 4 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, Qwen2.5-7B-Instruct (Q6_K))** | Era 4+: 2-Stage Hybrid Ensemble | **77.89%** | 67.3 / 92.5 | **62.61%** | 40.0% | 9.59% | 17.29% | 86.48% | 91.61% | 54.0s |
| 5 | **Two-Stage Hybrid (GBDT + Distributed Gemma-26B MoE)** | Era 4+: 2-Stage Hybrid Ensemble | **76.77%** | 64.4 / 95.0 | **62.18%** | 40.0% | 9.97% | 19.40% | 85.35% | 91.20% | 150.7s |
| 6 | **Two-Stage Hybrid (Signature LLM + 30F GBDT, LFM3-1.7B (Q8_0))** | Era 4+: 2-Stage Hybrid Ensemble | **77.42%** | 67.9 / 90.0 | **61.95%** | 40.0% | 9.97% | 18.44% | 86.39% | 91.34% | 23.0s |
| 7 | **Supervised 30-Feature Spatial+Continuity GBDT** | Era 3: Supervised Pairwise ML | **78.26%** | 69.2 / 90.0 | **60.71%** | 30.0% | 11.72% | 19.95% | 83.59% | 90.79% | 1.8s |
| 8 | **Holistic Windowed LLM (Gemma-4-E4B-IT (Q5_K_M))** | Era 4: Holistic Zero-Shot LLM | **75.56%** | 68.0 / 85.0 | **58.18%** | 0.0% | 17.13% | 25.42% | 69.39% | 83.71% | 17.4s |
| 9 | **Two-Stage Pairwise Signature LLM (Distributed Gemma-26B MoE)** | Era 4+: 2-Stage Pairwise LLM | **73.08%** | 59.4 / 95.0 | **56.45%** | 30.0% | 14.68% | 25.12% | 78.22% | 89.62% | 150.7s |
| 10 | **Two-Stage Pairwise Signature LLM (Gemma-4-E4B-IT (Q5_K_M))** | Era 4+: 2-Stage Pairwise LLM | **75.73%** | 61.9 / 97.5 | **55.28%** | 30.0% | 9.34% | 23.79% | 85.17% | 91.27% | 38.0s |
| 11 | **Two-Stage Pairwise Signature LLM (LFM3-1.7B (Q8_0))** | Era 4+: 2-Stage Pairwise LLM | **66.67%** | 65.8 / 67.5 | **47.52%** | 20.0% | 17.97% | 24.57% | 72.17% | 84.22% | 23.0s |
| 12 | **Two-Stage Pairwise Signature LLM (Gemma-4-E2B-IT (Q5_K_M))** | Era 4+: 2-Stage Pairwise LLM | **68.52%** | 54.4 / 92.5 | **40.63%** | 0.0% | 18.62% | 32.56% | 70.49% | 86.47% | 19.7s |
| 13 | **Unsupervised TF-IDF + Layout Shift** | Era 2: Unsupervised IR | **65.55%** | 49.4 / 97.5 | **38.85%** | 0.0% | 16.72% | 38.24% | 72.58% | 86.18% | 0.3s |
| 14 | **Rules + Pagination/Header Heuristic** | Era 1: Heuristic / Regex | **51.72%** | 83.3 / 37.5 | **35.90%** | 10.0% | 32.89% | 35.16% | 38.52% | 51.10% | 0.0s |
| 15 | **Two-Stage Pairwise Signature LLM (Qwen2.5-7B-Instruct (Q6_K))** | Era 4+: 2-Stage Pairwise LLM | **62.81%** | 46.9 / 95.0 | **32.62%** | 0.0% | 22.24% | 39.72% | 66.07% | 84.23% | 54.0s |
| 16 | **Holistic Windowed LLM (Qwen2.5-7B-Instruct (Q6_K))** | Era 4: Holistic Zero-Shot LLM | **51.61%** | 38.1 / 80.0 | **26.39%** | 0.0% | 34.38% | 48.30% | 48.74% | 77.45% | 25.9s |
| 17 | **Holistic Windowed LLM (LFM3-1.7B (Q8_0))** | Era 4: Holistic Zero-Shot LLM | **47.22%** | 53.1 / 42.5 | **21.74%** | 0.0% | 35.48% | 42.68% | 52.21% | 66.99% | 6.4s |
| 18 | **Holistic Windowed LLM (Gemma-4-E2B-IT (Q5_K_M))** | Era 4: Holistic Zero-Shot LLM | **42.31%** | 91.7 / 27.5 | **19.44%** | 0.0% | 37.64% | 40.29% | 34.25% | 41.74% | 6.2s |
