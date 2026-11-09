"""Page Stream Segmentation (PSS) & DocSplit-Seq Benchmark Tournament.

Evaluates 5 distinct algorithms across 2 major benchmarks (3 testbeds):
  1. Official TABME++ (`bevaya/TABMEpp` - UCSF Archival Business Folders with Azure 2D OCR)
  2. DocSplit-Poly-Seq (`RVL-CDIP-N-MP` Multi-Category Contiguous Packets)
  3. DocSplit-Mono-Seq (`RVL-CDIP-N-MP` Single-Category Contiguous Packets - Same-Type Back-to-Back)

Algorithms Evaluated:
  1. Era 1: Rule-Based Pagination & Header Heuristic (`Rules-Heuristic`)
  2. Era 2: Unsupervised TF-IDF Lexical & Header Continuity Scorer (`TFIDF-Continuity`)
  3. Era 3: Supervised 32-Feature Spatial + Continuity GBDT (`Supervised-GBDT-32F`)
  4. Holistic Windowed Zero-Shot LLM Prompting (`DocSplit-Holistic-LLM`) across local GGUF models
  5. Best-Practice Two-Stage Pairwise Signature + Entity/Pagination Continuity Pipeline
     (`TwoStage-Pairwise-LLM` & `TwoStage-Hybrid-Ensemble`) across local GGUF models

Metrics Computed:
  - Boundary Precision, Recall, F1 (page transition level)
  - Document-Level Exact Match F1 (Slot/Segment F1: exact [start_page, end_page] match)
  - Straight-Through Processing (STP %: percentage of folders/packets with 0 errors)
  - P_k Error Rate (Beeferman et al., 1999 - lower is better)
  - WindowDiff Error Rate (Pevzner & Hearst, 2002 - lower is better)
  - Adjusted Rand Index (ARI) & V-Measure (DocSplit paper clustering metrics)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.cluster import adjusted_rand_score, v_measure_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent
BIN_DIR = ROOT_DIR / "bin" / "llama-cuda"
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TABMEPP_FILE = DATA_DIR / "tabmepp_benchmark.json"
DOCSPLIT_FILE = DATA_DIR / "docsplit_seq_benchmark.json"
RESULTS_FILE = RESULTS_DIR / "pss_docsplit_tournament_results.json"
LEADERBOARD_FILE = ROOT_DIR / "PSS_DOCSPLIT_LEADERBOARD.md"

LOCAL_MODELS = [
    {
        "id": "lfm3-1.7b",
        "name": "LFM3-1.7B (Q8_0)",
        "file": "LFM3-1.7B-Q8_0.gguf",
        "ngl": 99,
        "ctx": 8192,
    },
    {
        "id": "gemma-4-e2b",
        "name": "Gemma-4-E2B-IT (Q5_K_M)",
        "file": "gemma-4-E2B-it-Q5_K_M.gguf",
        "ngl": 99,
        "ctx": 8192,
    },
    {
        "id": "gemma-4-e4b",
        "name": "Gemma-4-E4B-IT (Q5_K_M)",
        "file": "gemma-4-E4B-it-Q5_K_M.gguf",
        "ngl": 99,
        "ctx": 8192,
    },
    {
        "id": "qwen2.5-7b",
        "name": "Qwen2.5-7B-Instruct (Q6_K)",
        "file": "Qwen2.5-7B-Instruct-Q6_K.gguf",
        "ngl": 99,
        "ctx": 8192,
    },
]


# ============================================================================
# 1. STANDARD PSS & DOCSPLIT EVALUATION METRICS
# ============================================================================

def boundaries_to_spans(y_flags: List[int]) -> List[Tuple[int, int]]:
    """Convert 1-indexed binary boundary flags [1, 0, 1, ...] into 1-indexed (start, end) spans."""
    if not y_flags:
        return []
    spans = []
    start = 1
    for i in range(1, len(y_flags)):
        if y_flags[i] == 1:
            spans.append((start, i))
            start = i + 1
    spans.append((start, len(y_flags)))
    return spans


def boundaries_to_cluster_ids(y_flags: List[int]) -> List[int]:
    """Convert binary boundary flags into monotonically increasing cluster IDs per page."""
    cid = 0
    clusters = []
    for i, flag in enumerate(y_flags):
        if i == 0 or flag == 1:
            cid += 1
        clusters.append(cid)
    return clusters


def compute_pk_and_windowdiff(gt_flags: List[int], pred_flags: List[int]) -> Tuple[float, float]:
    """Compute Beeferman P_k and Pevzner-Hearst WindowDiff error rates (lower is better)."""
    n = len(gt_flags)
    if n <= 2:
        err = 0.0 if gt_flags == pred_flags else 1.0
        return err, err

    gt_clusters = boundaries_to_cluster_ids(gt_flags)
    pred_clusters = boundaries_to_cluster_ids(pred_flags)
    num_docs = max(1, sum(gt_flags))
    avg_len = n / float(num_docs)
    k = max(2, min(n - 1, int(round(avg_len / 2.0))))

    pk_errors = 0
    wd_errors = 0
    num_windows = n - k
    if num_windows <= 0:
        return 0.0, 0.0

    for i in range(num_windows):
        j = i + k
        gt_same = gt_clusters[i] == gt_clusters[j]
        pred_same = pred_clusters[i] == pred_clusters[j]
        if gt_same != pred_same:
            pk_errors += 1

        # Count boundaries strictly between i and j (i.e. indices i+1 .. j)
        gt_b = sum(gt_flags[i + 1 : j + 1])
        pred_b = sum(pred_flags[i + 1 : j + 1])
        if gt_b != pred_b:
            wd_errors += 1

    return pk_errors / float(num_windows), wd_errors / float(num_windows)


def evaluate_predictions(packets: List[Dict[str, Any]], pred_flags_per_packet: List[List[int]]) -> Dict[str, float]:
    """Compute comprehensive PSS & DocSplit metrics across a list of packets."""
    tp = fp = fn = tn = 0
    doc_tp = 0
    total_gt_docs = 0
    total_pred_docs = 0
    perfect_packets = 0

    pk_list = []
    wd_list = []
    ari_list = []
    vmeas_list = []

    for pkt, pred_flags in zip(packets, pred_flags_per_packet):
        gt_flags = [int(pg["is_first_page"]) for pg in pkt["pages"]]
        gt_flags[0] = 1
        pred_flags = list(pred_flags)
        pred_flags[0] = 1

        # Transition-level binary metrics (pages 2..N, index 1..N-1)
        for g, p in zip(gt_flags[1:], pred_flags[1:]):
            if g == 1 and p == 1:
                tp += 1
            elif g == 0 and p == 1:
                fp += 1
            elif g == 1 and p == 0:
                fn += 1
            else:
                tn += 1

        if gt_flags == pred_flags:
            perfect_packets += 1

        # Document-level exact span match (Segment/Slot F1)
        gt_spans = set(boundaries_to_spans(gt_flags))
        pred_spans = set(boundaries_to_spans(pred_flags))
        doc_tp += len(gt_spans.intersection(pred_spans))
        total_gt_docs += len(gt_spans)
        total_pred_docs += len(pred_spans)

        # P_k and WindowDiff
        pk, wd = compute_pk_and_windowdiff(gt_flags, pred_flags)
        pk_list.append(pk)
        wd_list.append(wd)

        # Clustering metrics (ARI & V-Measure from DocSplit paper)
        gt_c = boundaries_to_cluster_ids(gt_flags)
        pred_c = boundaries_to_cluster_ids(pred_flags)
        if len(gt_c) > 1:
            ari_list.append(float(adjusted_rand_score(gt_c, pred_c)))
            vmeas_list.append(float(v_measure_score(gt_c, pred_c)))

    b_prec = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    b_rec = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0

    d_prec = doc_tp / float(total_pred_docs) if total_pred_docs > 0 else 0.0
    d_rec = doc_tp / float(total_gt_docs) if total_gt_docs > 0 else 0.0
    d_f1 = 2 * d_prec * d_rec / (d_prec + d_rec) if (d_prec + d_rec) > 0 else 0.0

    stp = perfect_packets / float(len(packets)) if packets else 0.0

    return {
        "boundary_precision": round(b_prec * 100, 2),
        "boundary_recall": round(b_rec * 100, 2),
        "boundary_f1": round(b_f1 * 100, 2),
        "doc_exact_precision": round(d_prec * 100, 2),
        "doc_exact_recall": round(d_rec * 100, 2),
        "doc_exact_f1": round(d_f1 * 100, 2),
        "packet_stp_pct": round(stp * 100, 2),
        "pk_error": round(float(np.mean(pk_list)) * 100, 2) if pk_list else 0.0,
        "windowdiff_error": round(float(np.mean(wd_list)) * 100, 2) if wd_list else 0.0,
        "adjusted_rand_index": round(float(np.mean(ari_list)) * 100, 2) if ari_list else 0.0,
        "v_measure": round(float(np.mean(vmeas_list)) * 100, 2) if vmeas_list else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_packets": len(packets),
        "perfect_packets": perfect_packets,
        "total_gt_docs": total_gt_docs,
    }


# ============================================================================
# 2. RICH PAGE SIGNATURE & PAIRWISE FEATURE EXTRACTION
# ============================================================================

PAGE_ONE_RE = re.compile(
    r"(?:\bpage\s*1\b|\bpg\.?\s*1\b|\b1\s*(?:of|/)\s*\d+\b|^\s*-\s*1\s*-\s*$)",
    re.I | re.M,
)
PAGE_CONT_RE = re.compile(
    r"(?:\bpage\s*([2-9]|\d{2,})\b|\bpg\.?\s*([2-9]|\d{2,})\b|\b([2-9]|\d{2,})\s*(?:of|/)\s*\d+\b|^\s*-\s*([2-9]|\d{2,})\s*-\s*$|\(\s*cont(?:inued|\.?)\s*\)|\bcontinued\s+on\s+next\s+page\b|\bcontinued\s+from\b)",
    re.I | re.M,
)
TITLE_START_RE = re.compile(
    r"\b(memorandum|memo\s+to|curriculum\s+vitae|resume|invoice|purchase\s+order|table\s+of\s+contents|"
    r"executive\s+summary|abstract|dear\s+(?:mr|ms|mrs|dr|sir|madam)|minutes\s+of|agenda|"
    r"cover\s+sheet|transmittal|facsimile|fax\s+cover|press\s+release|news\s+release|"
    r"questionnaire|survey|specification|annual\s+report|financial\s+statement|balance\s+sheet)\b",
    re.I,
)
CLOSING_END_RE = re.compile(
    r"\b(sincerely|respectfully\s+submitted|best\s+regards|cordially|yours\s+truly|"
    r"approved\s+by|signature|total\s+amount\s+due|grand\s+total|cc:|attachments?:|enclosures?:)\b",
    re.I,
)
ID_TOKEN_RE = re.compile(
    r"\b(?:[A-Z]{2,5}[-#/]\d{3,10}|\d{3,5}[-/]\d{3,8}|#[0-9]{4,10}|\b(?:19|20)\d{2}\b|\b\d{9,10}\b)"
)
PROPER_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z]{2,15}(?:\s+[A-Z][A-Za-z]{2,15}){1,2}\b")


def extract_explicit_page_num(pg: Dict[str, Any]) -> Optional[int]:
    """Detect explicit page number in corner numbers or top/bottom lines."""
    candidates = [c["text"] for c in pg.get("corner_numbers", [])]
    top_txts = [t["text"] for t in pg.get("top_lines", [])[:3]]
    bot_txts = [b["text"] for b in pg.get("bottom_lines", [])[-3:]]

    for txt in candidates + top_txts + bot_txts:
        txt_s = txt.strip()
        m1 = re.search(r"\b(?:page|pg\.?)\s*(\d{1,3})\b", txt_s, re.I)
        if m1:
            val = int(m1.group(1))
            if 1 <= val <= 200:
                return val
        m2 = re.search(r"\b(\d{1,3})\s*(?:of|/)\s*(\d{1,3})\b", txt_s, re.I)
        if m2:
            val, tot = int(m2.group(1)), int(m2.group(2))
            if 1 <= val <= tot <= 200:
                return val
        m3 = re.match(r"^-?\s*(\d{1,2})\s*-?$", txt_s)
        if m3:
            val = int(m3.group(1))
            if 1 <= val <= 99:
                return val
    return None


def extract_page_signature(pg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 1 deterministic structural & entity signature extractor per page."""
    full_text = pg.get("full_text", "")
    lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()]
    top_text = "\n".join(lines[:6])
    bottom_text = "\n".join(lines[-5:]) if lines else ""

    exp_num = extract_explicit_page_num(pg)
    has_page_one_marker = bool(PAGE_ONE_RE.search(top_text) or PAGE_ONE_RE.search(bottom_text) or exp_num == 1)
    has_cont_marker = bool(
        PAGE_CONT_RE.search(top_text)
        or PAGE_CONT_RE.search(bottom_text)
        or (exp_num is not None and exp_num >= 2)
    )
    has_title_trigger = bool(TITLE_START_RE.search(top_text))
    has_closing_trigger = bool(CLOSING_END_RE.search(bottom_text))

    # Extract key identifiers (codes, reference numbers, years, uppercase headers)
    ids = set(ID_TOKEN_RE.findall(full_text[:1500]))
    entities = set(PROPER_ENTITY_RE.findall(top_text + "\n" + bottom_text))
    # First line tokens (running header check)
    first_line_words = set(re.findall(r"[a-zA-Z]{3,}", lines[0].lower())) if lines else set()
    top3_words = set(re.findall(r"[a-zA-Z]{3,}", "\n".join(lines[:3]).lower())) if lines else set()

    # Uppercase ratio in top 3 lines (strong signal of document title / letterhead)
    top3_str = " ".join(lines[:3])
    alpha_chars = [c for c in top3_str if c.isalpha()]
    upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / float(len(alpha_chars)) if alpha_chars else 0.0

    # Does the page start mid-sentence (lowercase letter or continuation punctuation)?
    starts_mid_sentence = False
    if lines:
        first_clean = lines[0].lstrip("0123456789.- )")
        if first_clean and first_clean[0].islower():
            starts_mid_sentence = True

    # Does the page end mid-sentence (no terminal punctuation on last line)?
    ends_mid_sentence = False
    if lines:
        last_clean = lines[-1].strip()
        if len(last_clean) > 35 and last_clean[-1] not in ".!?:;\"')]}":
            ends_mid_sentence = True

    return {
        "exp_page_num": exp_num,
        "has_page_one_marker": has_page_one_marker,
        "has_cont_marker": has_cont_marker,
        "has_title_trigger": has_title_trigger,
        "has_closing_trigger": has_closing_trigger,
        "ids": ids,
        "entities": entities,
        "first_line_words": first_line_words,
        "top3_words": top3_words,
        "upper_ratio": round(upper_ratio, 4),
        "starts_mid_sentence": starts_mid_sentence,
        "ends_mid_sentence": ends_mid_sentence,
        "top_text": top_text[:350],
        "bottom_text": bottom_text[:300],
        "char_len": len(full_text),
        "num_lines": pg.get("num_lines", len(lines)),
        "min_y1": float(pg.get("min_y1", 0.05)),
        "max_y3": float(pg.get("max_y3", 0.90)),
        "bottom_whitespace": float(pg.get("bottom_whitespace", 0.10)),
        "top_whitespace": float(pg.get("top_whitespace", 0.05)),
        "first_line_height": float(pg.get("first_line_height", 0.02)),
        "max_top_line_height": float(pg.get("max_top_line_height", 0.02)),
        "centered_top_lines": int(pg.get("centered_top_lines", 0)),
    }


def compute_packet_tfidf_similarities(pages: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Compute adjacent-page full-text and top-header TF-IDF cosine similarities."""
    n = len(pages)
    if n <= 1:
        return [0.0], [0.0]

    texts = [p.get("full_text", "") or "empty_page" for p in pages]
    headers = [
        "\n".join([ln.strip() for ln in (p.get("full_text", "") or "").splitlines()[:5]]) or "empty_header"
        for p in pages
    ]
    try:
        vec_full = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
        mat_full = vec_full.fit_transform(texts)
        full_sims = [0.0] + [float(cosine_similarity(mat_full[i - 1], mat_full[i])[0, 0]) for i in range(1, n)]
    except Exception:
        full_sims = [0.0] * n

    try:
        vec_hdr = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        mat_hdr = vec_hdr.fit_transform(headers)
        hdr_sims = [0.0] + [float(cosine_similarity(mat_hdr[i - 1], mat_hdr[i])[0, 0]) for i in range(1, n)]
    except Exception:
        hdr_sims = [0.0] * n

    return full_sims, hdr_sims


def extract_pairwise_feature_vector(
    sig_prev: Dict[str, Any],
    sig_curr: Dict[str, Any],
    full_sim: float,
    hdr_sim: float,
    docs_since_last_split: int = 1,
) -> List[float]:
    """Build a 30-dimensional multimodal (2D layout + pagination + lexical + entity) feature vector."""
    n_prev = sig_prev["exp_page_num"]
    n_curr = sig_curr["exp_page_num"]

    is_seq_num = 1.0 if (n_prev is not None and n_curr is not None and n_curr == n_prev + 1) else 0.0
    is_num_2 = 1.0 if (n_curr == 2) else 0.0
    is_num_gt1 = 1.0 if (n_curr is not None and n_curr >= 2) else 0.0
    is_num_reset_1 = 1.0 if (n_curr == 1 and n_prev != 0) else 0.0

    # Jaccard overlaps
    def jaccard(s1: Set[str], s2: Set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        return len(s1.intersection(s2)) / float(len(s1.union(s2)))

    first_line_jaccard = jaccard(sig_prev["first_line_words"], sig_curr["first_line_words"])
    top3_jaccard = jaccard(sig_prev["top3_words"], sig_curr["top3_words"])
    id_jaccard = jaccard(sig_prev["ids"], sig_curr["ids"])
    entity_jaccard = jaccard(sig_prev["entities"], sig_curr["entities"])

    # Conflicting IDs: both pages have IDs, and their intersection is empty
    conflicting_ids = 1.0 if (sig_prev["ids"] and sig_curr["ids"] and not sig_prev["ids"].intersection(sig_curr["ids"])) else 0.0

    len_prev = max(1.0, float(sig_prev["char_len"]))
    len_curr = max(1.0, float(sig_curr["char_len"]))
    len_ratio = len_curr / len_prev

    return [
        float(full_sim),
        float(hdr_sim),
        float(first_line_jaccard),
        float(top3_jaccard),
        float(id_jaccard),
        float(entity_jaccard),
        float(conflicting_ids),
        float(is_seq_num),
        float(is_num_2),
        float(is_num_gt1),
        float(is_num_reset_1),
        1.0 if sig_curr["has_page_one_marker"] else 0.0,
        1.0 if sig_curr["has_cont_marker"] else 0.0,
        1.0 if sig_curr["has_title_trigger"] else 0.0,
        1.0 if sig_prev["has_closing_trigger"] else 0.0,
        1.0 if sig_curr["starts_mid_sentence"] else 0.0,
        1.0 if sig_prev["ends_mid_sentence"] else 0.0,
        float(sig_curr["upper_ratio"]),
        float(sig_prev["upper_ratio"]),
        float(sig_prev["bottom_whitespace"]),
        float(sig_curr["top_whitespace"]),
        float(sig_curr["bottom_whitespace"]),
        float(sig_curr["first_line_height"]),
        float(sig_curr["max_top_line_height"]),
        float(sig_curr["max_top_line_height"] - sig_prev["max_top_line_height"]),
        float(sig_curr["centered_top_lines"]),
        float(sig_prev["num_lines"]) / 50.0,
        float(sig_curr["num_lines"]) / 50.0,
        min(5.0, len_ratio),
        float(min(10, docs_since_last_split)) / 10.0,
    ]


# ============================================================================
# 3. ALGORITHMS 1, 2, AND 3 (HEURISTIC, TF-IDF CONTINUITY, SUPERVISED GBDT)
# ============================================================================

def run_algorithm_1_rules_heuristic(packets: List[Dict[str, Any]]) -> List[List[int]]:
    """Algorithm 1 (Era 1): Rule-Based Pagination, Header & Whitespace Heuristic."""
    preds_all = []
    for pkt in packets:
        pages = pkt["pages"]
        sigs = [extract_page_signature(p) for p in pages]
        flags = [1]
        for i in range(1, len(pages)):
            sp, sc = sigs[i - 1], sigs[i]
            # Rule 1: Explicit continuation number or marker -> CONTINUE (0)
            if sc["has_cont_marker"] or (
                sp["exp_page_num"] is not None
                and sc["exp_page_num"] is not None
                and sc["exp_page_num"] == sp["exp_page_num"] + 1
            ):
                flags.append(0)
                continue
            # Rule 2: Mid-sentence continuation -> CONTINUE (0)
            if sc["starts_mid_sentence"] and sp["ends_mid_sentence"]:
                flags.append(0)
                continue
            # Rule 3: Explicit Page 1 marker or title trigger or previous page had closing + whitespace
            score = 0.0
            if sc["has_page_one_marker"]:
                score += 1.5
            if sc["has_title_trigger"]:
                score += 0.9
            if sp["has_closing_trigger"]:
                score += 0.7
            if sp["bottom_whitespace"] > 0.35:
                score += 0.6
            if sc["top_whitespace"] > 0.12 and sc["centered_top_lines"] >= 2:
                score += 0.5
            if sc["first_line_words"] and sp["first_line_words"] == sc["first_line_words"]:
                score -= 0.8  # Repeated running header
            flags.append(1 if score >= 0.85 else 0)
        preds_all.append(flags)
    return preds_all


def run_algorithm_2_tfidf_continuity(packets: List[Dict[str, Any]]) -> List[List[int]]:
    """Algorithm 2 (Era 2): Unsupervised TF-IDF Cosine Similarity + Layout Discontinuity."""
    preds_all = []
    for pkt in packets:
        pages = pkt["pages"]
        sigs = [extract_page_signature(p) for p in pages]
        full_sims, hdr_sims = compute_packet_tfidf_similarities(pages)
        flags = [1]
        for i in range(1, len(pages)):
            sp, sc = sigs[i - 1], sigs[i]
            # Combine lexical continuity with pagination/layout shift
            cont_score = 0.65 * full_sims[i] + 0.35 * hdr_sims[i]
            if sc["has_cont_marker"] or (sc["exp_page_num"] is not None and sc["exp_page_num"] >= 2):
                cont_score += 0.35
            if sc["has_page_one_marker"] or sc["has_title_trigger"]:
                cont_score -= 0.25
            if sp["bottom_whitespace"] > 0.35:
                cont_score -= 0.18
            if sp["ids"] and sc["ids"] and not sp["ids"].intersection(sc["ids"]):
                cont_score -= 0.20
            flags.append(0 if cont_score >= 0.19 else 1)
        preds_all.append(flags)
    return preds_all


def build_supervised_dataset(packets: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    X_rows = []
    y_rows = []
    for pkt in packets:
        pages = pkt["pages"]
        sigs = [extract_page_signature(p) for p in pages]
        full_sims, hdr_sims = compute_packet_tfidf_similarities(pages)
        run_len = 1
        for i in range(1, len(pages)):
            feat = extract_pairwise_feature_vector(sigs[i - 1], sigs[i], full_sims[i], hdr_sims[i], run_len)
            label = int(pages[i]["is_first_page"])
            X_rows.append(feat)
            y_rows.append(label)
            run_len = 1 if label == 1 else run_len + 1
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


def train_and_run_algorithm_3_supervised_gbdt(
    train_packets: List[Dict[str, Any]],
    test_packets: List[Dict[str, Any]],
) -> Tuple[List[List[int]], List[List[float]]]:
    """Algorithm 3 (Era 3): Supervised 30-Feature Multimodal Spatial + Continuity GBDT + LR Ensemble."""
    X_train, y_train = build_supervised_dataset(train_packets)
    gbdt = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        random_state=42,
    )
    gbdt.fit(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    lr.fit(X_train_scaled, y_train)

    preds_all = []
    probs_all = []
    for pkt in test_packets:
        pages = pkt["pages"]
        sigs = [extract_page_signature(p) for p in pages]
        full_sims, hdr_sims = compute_packet_tfidf_similarities(pages)
        flags = [1]
        pkt_probs = [1.0]
        run_len = 1
        for i in range(1, len(pages)):
            feat = np.array(
                [extract_pairwise_feature_vector(sigs[i - 1], sigs[i], full_sims[i], hdr_sims[i], run_len)],
                dtype=np.float32,
            )
            p_gbdt = float(gbdt.predict_proba(feat)[0, 1])
            p_lr = float(lr.predict_proba(scaler.transform(feat))[0, 1])
            prob = 0.75 * p_gbdt + 0.25 * p_lr
            pred = 1 if prob >= 0.50 else 0
            flags.append(pred)
            pkt_probs.append(prob)
            run_len = 1 if pred == 1 else run_len + 1
        preds_all.append(flags)
        probs_all.append(pkt_probs)
    return preds_all, probs_all


# ============================================================================
# 4. LOCAL LLAMA-SERVER HARNESS FOR ALGORITHMS 4 & 5
# ============================================================================

class LlamaServerManager:
    def __init__(self, model_cfg: Dict[str, Any], port: int = 8089):
        self.model_cfg = model_cfg
        self.port = port
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = f"http://127.0.0.1:{port}"

    def start(self) -> None:
        subprocess.run(
            ["taskkill", "/F", "/IM", "llama-server.exe"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1.5)
        server_exe = BIN_DIR / "llama-server.exe"
        model_path = MODELS_DIR / self.model_cfg["file"]
        cmd = [
            str(server_exe),
            "-m",
            str(model_path),
            "--port",
            str(self.port),
            "--host",
            "127.0.0.1",
            "-ngl",
            str(self.model_cfg.get("ngl", 99)),
            "-c",
            str(self.model_cfg.get("ctx", 8192)),
            "--flash-attn",
            "on",
            "-np",
            "1",
            "--reasoning-format",
            "none",
            "-rea",
            "off",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for health check
        for _ in range(90):
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError(f"llama-server failed to start for {self.model_cfg['name']}")

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(f"{self.base_url}/v1/chat/completions", json=payload, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception as e:
            return f"ERROR: {e}"


def run_algorithm_4_holistic_window_llm(
    server: LlamaServerManager,
    packets: List[Dict[str, Any]],
    window_size: int = 12,
) -> List[List[int]]:
    """Algorithm 4 (DocSplit Holistic Prompting Baseline):
    Feeds multi-page windows into the LLM and asks which pages start a new document.
    """
    sys_prompt = (
        "You are a document packet splitting expert. Given raw page snippets from a multi-page PDF packet, "
        "identify which page numbers are the FIRST page of a new independent document. "
        "Reply ONLY with a JSON object: {\"start_pages\": [1, ...]} containing the 1-based page numbers that start a new document."
    )
    preds_all = []
    for pkt in packets:
        pages = pkt["pages"]
        n = len(pages)
        flags = [0] * n
        flags[0] = 1
        step = max(1, window_size - 1)
        for w_start in range(0, n, step):
            w_end = min(n, w_start + window_size)
            if w_end - w_start <= 1 and w_start > 0:
                break
            lines_prompt = []
            for idx in range(w_start, w_end):
                pg = pages[idx]
                raw_txt = (pg.get("full_text", "") or "").replace("\n", " ")
                head = raw_txt[:160]
                tail = raw_txt[-100:] if len(raw_txt) > 160 else ""
                lines_prompt.append(f"Page {idx+1}: {head} ... {tail}")
            user_prompt = (
                f"Pages {w_start+1} to {w_end} of a document packet are shown below. "
                f"Which page numbers in {w_start+1}..{w_end} start a NEW independent document?\n\n"
                + "\n".join(lines_prompt)
                + "\n\nOutput JSON only: {\"start_pages\": [...]}"
            )
            resp = server.chat(sys_prompt, user_prompt, max_tokens=120)
            found_starts = set()
            m_json = re.search(r"\"start_pages\"\s*:\s*\[([^\]]*)\]", resp)
            if m_json:
                for num_s in re.findall(r"\d+", m_json.group(1)):
                    val = int(num_s)
                    if w_start + 1 <= val <= w_end:
                        found_starts.add(val)
            else:
                for num_s in re.findall(r"\b\d+\b", resp):
                    val = int(num_s)
                    if w_start + 1 <= val <= w_end:
                        found_starts.add(val)
            for s_pg in found_starts:
                if s_pg > w_start + 1 or w_start == 0:
                    flags[s_pg - 1] = 1
        flags[0] = 1
        preds_all.append(flags)
    return preds_all


def run_algorithm_5_twostage_pairwise_llm(
    server: LlamaServerManager,
    packets: List[Dict[str, Any]],
    gbdt_probs_per_packet: List[List[float]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Algorithm 5 (Our Best-Practice Two-Stage Pairwise Signature + Continuity Pipeline):
    Returns:
      - 5A (`TwoStage-Pairwise-LLM`): Stage 1 PageSignature + Continuity Signals evaluated in pairwise batches by the LLM.
      - 5B (`TwoStage-Selective-Hybrid`): Fast-path 30F Spatial GBDT + 1-on-1 Pairwise Signature LLM adjudication on uncertain boundaries.
    """
    batch_sys_prompt = (
        "You are an expert Page Stream Segmentation classifier using Stage-1 Page Signatures and Continuity Signals.\n"
        "For each adjacent page pair (Page t-1 -> Page t), decide if Page t starts a NEW document (SPLIT) or continues the same document (CONTINUE).\n"
        "Rules:\n"
        "- Explicit Page# >= 2 (e.g. 2, 3) or sequential page numbers (1->2, 2->3) or mid-sentence continuation -> CONTINUE.\n"
        "- Explicit Page# = 1, new title/memo/invoice/resume header, high bottom whitespace on Page(t-1), or conflicting document IDs with low lexical similarity -> SPLIT.\n"
        "Output ONLY a JSON object mapping the target page number to \"SPLIT\" or \"CONTINUE\", e.g. {\"2\": \"SPLIT\", \"3\": \"CONTINUE\"}."
    )

    single_sys_prompt = (
        "You are an expert Page Stream Segmentation adjudicator resolving an ambiguous page transition (Page t-1 -> Page t).\n"
        "Carefully compare Page(t-1)'s footer/closing and IDs with Page(t)'s header/title, pagination, and lexical similarity.\n"
        "Reply ONLY with a JSON object: {\"decision\": \"SPLIT\" or \"CONTINUE\", \"confidence\": 0.5 to 1.0}"
    )

    pure_llm_preds = []
    hybrid_preds = []

    for pkt, pkt_gbdt_probs in zip(packets, gbdt_probs_per_packet):
        pages = pkt["pages"]
        n = len(pages)
        sigs = [extract_page_signature(p) for p in pages]
        full_sims, hdr_sims = compute_packet_tfidf_similarities(pages)

        flags_llm = [1] + [0] * (n - 1)
        flags_hybrid = [1] + [0] * (n - 1)

        # First handle deterministic pagination fast-paths and collect transitions for batch pairwise LLM
        pending_indices = []
        for i in range(1, n):
            sp, sc = sigs[i - 1], sigs[i]
            n_prev, n_curr = sp["exp_page_num"], sc["exp_page_num"]
            if n_prev is not None and n_curr is not None and n_curr == n_prev + 1:
                flags_llm[i] = 0
                flags_hybrid[i] = 0
            elif n_curr is not None and n_curr >= 2 and not sc["has_page_one_marker"]:
                flags_llm[i] = 0
                flags_hybrid[i] = 0
            elif sc["has_page_one_marker"] and not sc["has_cont_marker"]:
                flags_llm[i] = 1
                flags_hybrid[i] = 1
            else:
                pending_indices.append(i)

        # 5A: Evaluate pending transitions in batches of 6 pairs using Structured Pairwise Signatures
        batch_size = 6
        llm_pair_decisions: Dict[int, int] = {}
        for b_start in range(0, len(pending_indices), batch_size):
            batch_idxs = pending_indices[b_start : b_start + batch_size]
            pair_blocks = []
            for i in batch_idxs:
                sp, sc = sigs[i - 1], sigs[i]
                shared_ids = sorted(list(sp["ids"].intersection(sc["ids"])))[:4]
                conflict_ids = bool(sp["ids"] and sc["ids"] and not sp["ids"].intersection(sc["ids"]))
                top_prev = sp["top_text"].replace("\n", " | ")[:110]
                bot_prev = sp["bottom_text"].replace("\n", " | ")[:110]
                top_curr = sc["top_text"].replace("\n", " | ")[:140]
                pair_blocks.append(
                    f"Transition to Page {i+1} (from Page {i}):\n"
                    f"  Page {i} (t-1): Lines={sp['num_lines']}, BotWhitespace={sp['bottom_whitespace']:.2f}, "
                    f"Closing={sp['has_closing_trigger']}, EndsMidSent={sp['ends_mid_sentence']}, Top={top_prev!r}, Bot={bot_prev!r}\n"
                    f"  Page {i+1} (t): Lines={sc['num_lines']}, TopWhitespace={sc['top_whitespace']:.2f}, "
                    f"TitleTrigger={sc['has_title_trigger']}, ContMarker={sc['has_cont_marker']}, StartsMidSent={sc['starts_mid_sentence']}, "
                    f"UpperRatio={sc['upper_ratio']:.2f}, Top={top_curr!r}\n"
                    f"  Continuity: TextSim={full_sims[i]:.2f}, HeaderSim={hdr_sims[i]:.2f}, SharedIDs={shared_ids}, ConflictingIDs={conflict_ids}"
                )
            user_prompt = (
                "Classify each transition below as \"SPLIT\" or \"CONTINUE\":\n\n"
                + "\n\n".join(pair_blocks)
                + "\n\nOutput JSON mapping page number to SPLIT or CONTINUE:"
            )
            resp = server.chat(batch_sys_prompt, user_prompt, max_tokens=140)
            for i in batch_idxs:
                pg_str = str(i + 1)
                m = re.search(rf"\"{pg_str}\"\s*:\s*\"(SPLIT|CONTINUE)\"", resp, re.I)
                if m:
                    dec = 1 if m.group(1).upper() == "SPLIT" else 0
                else:
                    # Fallback heuristic if model omitted key
                    dec = 1 if pkt_gbdt_probs[i] >= 0.5 else 0
                flags_llm[i] = dec
                llm_pair_decisions[i] = dec

        # 5B: Selective Hybrid Adjudicator (1-on-1 LLM reasoning on uncertain GBDT transitions)
        for i in pending_indices:
            gbdt_p = pkt_gbdt_probs[i]
            if 0.25 <= gbdt_p <= 0.75:
                sp, sc = sigs[i - 1], sigs[i]
                shared_ids = sorted(list(sp["ids"].intersection(sc["ids"])))[:5]
                prev_only_ids = sorted(list(sp["ids"] - sc["ids"]))[:5]
                curr_only_ids = sorted(list(sc["ids"] - sp["ids"]))[:5]
                u_prompt = (
                    f"Transition Page {i} -> Page {i+1} (Spatial Classifier Prior Split Prob = {gbdt_p:.2f}):\n"
                    f"Page {i} (t-1): Lines={sp['num_lines']}, BotWhitespace={sp['bottom_whitespace']:.2f}, "
                    f"Closing={sp['has_closing_trigger']}, EndsMidSent={sp['ends_mid_sentence']}\n"
                    f"  Top: {sp['top_text'][:180]!r}\n  Bottom: {sp['bottom_text'][:180]!r}\n"
                    f"Page {i+1} (t): Lines={sc['num_lines']}, TopWhitespace={sc['top_whitespace']:.2f}, "
                    f"TitleTrigger={sc['has_title_trigger']}, StartsMidSent={sc['starts_mid_sentence']}, UpperRatio={sc['upper_ratio']:.2f}\n"
                    f"  Top: {sc['top_text'][:220]!r}\n"
                    f"Signals: LexicalSim={full_sims[i]:.3f}, HeaderSim={hdr_sims[i]:.3f}, "
                    f"SharedIDs={shared_ids}, PrevOnlyIDs={prev_only_ids}, CurrOnlyIDs={curr_only_ids}\n"
                    f"Output JSON: {{\"decision\": \"SPLIT\" or \"CONTINUE\", \"confidence\": 0.5..1.0}}"
                )
                resp = server.chat(single_sys_prompt, u_prompt, max_tokens=64)
                resp_u = resp.upper()
                llm_split = 1 if ("\"SPLIT\"" in resp_u or (resp_u.find("SPLIT") != -1 and resp_u.find("CONTINUE") == -1)) else 0
                m_conf = re.search(r"\"confidence\"\s*:\s*([0-9.]+)", resp)
                conf = float(m_conf.group(1)) if m_conf else 0.70
                conf = max(0.5, min(0.95, conf))
                llm_prob = conf if llm_split == 1 else (1.0 - conf)
                # Combine calibrated GBDT probability (0.65) + 1-on-1 LLM adjudicator (0.35) + batch pairwise vote
                batch_vote = float(llm_pair_decisions.get(i, llm_split))
                fused_p = 0.62 * gbdt_p + 0.26 * llm_prob + 0.12 * batch_vote
                flags_hybrid[i] = 1 if fused_p >= 0.50 else 0
            else:
                flags_hybrid[i] = 1 if gbdt_p >= 0.50 else 0

        pure_llm_preds.append(flags_llm)
        hybrid_preds.append(flags_hybrid)

    return pure_llm_preds, hybrid_preds


def generate_markdown_leaderboard(results: Dict[str, Any]) -> str:
    lines = [
        "# ✂️ Page Stream Segmentation (PSS) & DocSplit-Seq Benchmark Leaderboard",
        "",
        "**Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)  ",
        "**Benchmarks Evaluated:**",
        "1. **TABME++ (`bevaya/TABMEpp`)** — Official UCSF archival business document folders with Microsoft Azure 2D OCR.",
        "2. **DocSplit-Poly-Seq (`RVL-CDIP-N-MP`)** — Multi-category contiguous packets (invoices, resumes, budgets, memos, letters, forms, etc.).",
        "3. **DocSplit-Mono-Seq (`RVL-CDIP-N-MP`)** — Single-category contiguous packets (multiple multi-page documents of the *exact same class* concatenated back-to-back).",
        "",
        "---",
        "",
    ]

    suite_titles = {
        "tabmepp": "## 1. Official TABME++ Benchmark (`bevaya/TABMEpp`)",
        "docsplit_poly_seq": "## 2. DocSplit-Poly-Seq Benchmark (Multi-Category Contiguous Packets)",
        "docsplit_mono_seq": "## 3. DocSplit-Mono-Seq Benchmark (Same-Category Contiguous Packets — Hardest Case)",
    }

    for suite_key, title in suite_titles.items():
        if suite_key not in results:
            continue
        suite_data = results[suite_key]
        meta = suite_data["metadata"]
        algos = suite_data["algorithms"]

        lines.append(title)
        lines.append(
            f"* **Test Suite Size:** {meta['num_packets']} packets/folders | "
            f"{meta['total_docs']} documents | {meta['total_pages']} pages"
        )
        lines.append("")
        lines.append(
            "| Rank | Algorithm / Model | Paradigm | Boundary F1 (%) | Boundary P / R (%) | Doc Exact Match F1 (%) | Packet STP (%) | $P_k$ Err (%) ↓ | WindowDiff (%) ↓ | ARI (%) | V-Measure (%) | Time (s) |"
        )
        lines.append(
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        # Sort by Document Exact Match F1 then Boundary F1
        sorted_algos = sorted(
            algos,
            key=lambda x: (x["metrics"]["doc_exact_f1"], x["metrics"]["boundary_f1"]),
            reverse=True,
        )
        for rank, entry in enumerate(sorted_algos, 1):
            m = entry["metrics"]
            lines.append(
                f"| {rank} | **{entry['name']}** | {entry['paradigm']} | "
                f"**{m['boundary_f1']:.2f}%** | {m['boundary_precision']:.1f} / {m['boundary_recall']:.1f} | "
                f"**{m['doc_exact_f1']:.2f}%** | {m['packet_stp_pct']:.1f}% | "
                f"{m['pk_error']:.2f}% | {m['windowdiff_error']:.2f}% | "
                f"{m['adjusted_rand_index']:.2f}% | {m['v_measure']:.2f}% | {entry['elapsed_sec']:.1f}s |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llms", action="store_true", help="Run only non-LLM algorithms first")
    parser.add_argument("--max-llm-packets", type=int, default=12, help="Number of packets per suite for LLM eval")
    args = parser.parse_args()

    print("Loading prepared benchmark datasets...")
    with open(TABMEPP_FILE, encoding="utf-8") as f:
        tabme_data = json.load(f)
    with open(DOCSPLIT_FILE, encoding="utf-8") as f:
        docsplit_data = json.load(f)

    suites = {
        "tabmepp": {
            "train": tabme_data["train_packets"],
            "test": tabme_data["test_packets"][: args.max_llm_packets] if not args.skip_llms else tabme_data["test_packets"],
        },
        "docsplit_poly_seq": {
            "train": docsplit_data["train_packets"],
            "test": docsplit_data["poly_seq_test"][: args.max_llm_packets] if not args.skip_llms else docsplit_data["poly_seq_test"],
        },
        "docsplit_mono_seq": {
            "train": docsplit_data["train_packets"],
            "test": docsplit_data["mono_seq_test"][: args.max_llm_packets] if not args.skip_llms else docsplit_data["mono_seq_test"],
        },
    }

    all_results: Dict[str, Any] = {}
    gbdt_probs_cache: Dict[str, List[List[float]]] = {}

    # 1. Run Algorithms 1, 2, 3 across all suites
    for s_key, s_val in suites.items():
        test_pkts = s_val["test"]
        train_pkts = s_val["train"]
        total_pages = sum(p["total_pages"] for p in test_pkts)
        total_docs = sum(p["num_documents"] for p in test_pkts)
        print(f"\n=== Evaluating Classical & Supervised Algorithms on {s_key} ({len(test_pkts)} packets, {total_docs} docs, {total_pages} pages) ===")

        suite_algos = []

        # Algo 1: Rules & Heuristic
        t0 = time.perf_counter()
        p1 = run_algorithm_1_rules_heuristic(test_pkts)
        el1 = time.perf_counter() - t0
        m1 = evaluate_predictions(test_pkts, p1)
        print(f"  [Algo 1] Rules-Heuristic         -> Boundary F1: {m1['boundary_f1']:6.2f}% | Doc Exact F1: {m1['doc_exact_f1']:6.2f}% | STP: {m1['packet_stp_pct']:5.1f}% | Pk: {m1['pk_error']:5.2f}%")
        suite_algos.append({
            "id": "algo1_rules",
            "name": "Rules + Pagination/Header Heuristic",
            "paradigm": "Era 1: Heuristic / Regex",
            "metrics": m1,
            "elapsed_sec": round(el1, 2),
        })

        # Algo 2: Unsupervised TF-IDF Continuity
        t0 = time.perf_counter()
        p2 = run_algorithm_2_tfidf_continuity(test_pkts)
        el2 = time.perf_counter() - t0
        m2 = evaluate_predictions(test_pkts, p2)
        print(f"  [Algo 2] TFIDF-Continuity        -> Boundary F1: {m2['boundary_f1']:6.2f}% | Doc Exact F1: {m2['doc_exact_f1']:6.2f}% | STP: {m2['packet_stp_pct']:5.1f}% | Pk: {m2['pk_error']:5.2f}%")
        suite_algos.append({
            "id": "algo2_tfidf",
            "name": "Unsupervised TF-IDF + Layout Shift",
            "paradigm": "Era 2: Unsupervised IR",
            "metrics": m2,
            "elapsed_sec": round(el2, 2),
        })

        # Algo 3: Supervised 30-Feature Multimodal GBDT + LR
        t0 = time.perf_counter()
        p3, probs3 = train_and_run_algorithm_3_supervised_gbdt(train_pkts, test_pkts)
        el3 = time.perf_counter() - t0
        m3 = evaluate_predictions(test_pkts, p3)
        gbdt_probs_cache[s_key] = probs3
        print(f"  [Algo 3] Supervised-GBDT-30F     -> Boundary F1: {m3['boundary_f1']:6.2f}% | Doc Exact F1: {m3['doc_exact_f1']:6.2f}% | STP: {m3['packet_stp_pct']:5.1f}% | Pk: {m3['pk_error']:5.2f}%")
        suite_algos.append({
            "id": "algo3_gbdt",
            "name": "Supervised 30-Feature Spatial+Continuity GBDT",
            "paradigm": "Era 3: Supervised Pairwise ML",
            "metrics": m3,
            "elapsed_sec": round(el3, 2),
        })

        all_results[s_key] = {
            "metadata": {
                "num_packets": len(test_pkts),
                "total_docs": total_docs,
                "total_pages": total_pages,
            },
            "algorithms": suite_algos,
        }

    if not args.skip_llms:
        for model_cfg in LOCAL_MODELS:
            model_file = MODELS_DIR / model_cfg["file"]
            if not model_file.exists():
                print(f"Skipping missing model {model_file}")
                continue
            print(f"\n=======================================================")
            print(f"Starting llama-server for {model_cfg['name']}...")
            print(f"=======================================================")
            server = LlamaServerManager(model_cfg)
            try:
                server.start()
                for s_key, s_val in suites.items():
                    test_pkts = s_val["test"]
                    print(f"\n--- Running LLM Algorithms with {model_cfg['name']} on {s_key} ---")

                    # Algo 4: Holistic Windowed Zero-Shot LLM (DocSplit Baseline)
                    t0 = time.perf_counter()
                    p4 = run_algorithm_4_holistic_window_llm(server, test_pkts, window_size=8)
                    el4 = time.perf_counter() - t0
                    m4 = evaluate_predictions(test_pkts, p4)
                    print(f"  [Algo 4] Holistic-Window ({model_cfg['id']}) -> Boundary F1: {m4['boundary_f1']:6.2f}% | Doc Exact F1: {m4['doc_exact_f1']:6.2f}% | STP: {m4['packet_stp_pct']:5.1f}% | ({el4:.1f}s)")
                    all_results[s_key]["algorithms"].append({
                        "id": f"algo4_holistic_{model_cfg['id']}",
                        "name": f"Holistic Windowed LLM ({model_cfg['name']})",
                        "paradigm": "Era 4: Holistic Zero-Shot LLM",
                        "metrics": m4,
                        "elapsed_sec": round(el4, 2),
                    })

                    # Algo 5A & 5B: Two-Stage Pairwise Signature LLM & Calibrated Hybrid Ensemble
                    t0 = time.perf_counter()
                    p5_llm, p5_hybrid = run_algorithm_5_twostage_pairwise_llm(
                        server, test_pkts, gbdt_probs_cache[s_key]
                    )
                    el5 = time.perf_counter() - t0
                    m5_llm = evaluate_predictions(test_pkts, p5_llm)
                    m5_hyb = evaluate_predictions(test_pkts, p5_hybrid)
                    print(f"  [Algo 5A] TwoStage-Pairwise ({model_cfg['id']}) -> Boundary F1: {m5_llm['boundary_f1']:6.2f}% | Doc Exact F1: {m5_llm['doc_exact_f1']:6.2f}% | STP: {m5_llm['packet_stp_pct']:5.1f}% | ({el5:.1f}s)")
                    print(f"  [Algo 5B] TwoStage-Hybrid   ({model_cfg['id']}) -> Boundary F1: {m5_hyb['boundary_f1']:6.2f}% | Doc Exact F1: {m5_hyb['doc_exact_f1']:6.2f}% | STP: {m5_hyb['packet_stp_pct']:5.1f}%")
                    all_results[s_key]["algorithms"].append({
                        "id": f"algo5_pairwise_{model_cfg['id']}",
                        "name": f"Two-Stage Pairwise Signature LLM ({model_cfg['name']})",
                        "paradigm": "Era 4+: 2-Stage Pairwise LLM",
                        "metrics": m5_llm,
                        "elapsed_sec": round(el5, 2),
                    })
                    all_results[s_key]["algorithms"].append({
                        "id": f"algo5_hybrid_{model_cfg['id']}",
                        "name": f"Two-Stage Hybrid (Signature LLM + 30F GBDT, {model_cfg['name']})",
                        "paradigm": "Era 4+: 2-Stage Hybrid Ensemble",
                        "metrics": m5_hyb,
                        "elapsed_sec": round(el5, 2),
                    })
            finally:
                server.stop()

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    md = generate_markdown_leaderboard(all_results)
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\nSaved results to {RESULTS_FILE} and leaderboard to {LEADERBOARD_FILE}")


if __name__ == "__main__":
    main()

