"""Production Page Stream Segmentation (PSS) & Document Classification Pipeline.

A 4-stage hierarchical multimodal architecture:
  Stage 1: Front-Door Ingestion & Layout Parsing (Text, Whitespace, 2D Bounding Boxes)
  Stage 2: Deterministic Fast-Path Filter (<1ms | 0 Cost)
  Stage 3: Spatial 30-Feature GBDT Classifier (<2ms)
  Stage 4: Configurable Edge Model Joint Adjudicator & Document Classifier

Includes:
  - Dataset Registry: Load datasets by label/identifier (`tabmepp`, `docsplit_poly_seq`, `docsplit_mono_seq`, or PDF).
  - Telemetry & Tracing: 100% local Arize Phoenix integration + JSONL diagnostic error logging.
  - Configurable Stage 4 model: Presets (`gemma-4-e4b`, `gemma-4-e2b`, `lfm3-1.7b`, `qwen2.5-7b`), custom GGUF, or endpoint.
  - Zero distributed dependencies: 100% single-laptop execution in 8GB VRAM / 16GB RAM.
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request

import numpy as np
from pypdf import PdfReader, PdfWriter

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PKL = MODELS_DIR / "pss_classifier.pkl"

# Local model registry for single-laptop execution (RTX 4060 8GB VRAM)
LOCAL_MODELS_REGISTRY = {
    "gemma-4-e4b": {
        "name": "Gemma-4-E4B-IT (Q5_K_M)",
        "file": "gemma-4-E4B-it-Q5_K_M.gguf",
        "ctx": 4096,
        "gpu_layers": 99,
        "vram_mb": 3350,
    },
    "gemma-4-e2b": {
        "name": "Gemma-4-E2B-IT (Q5_K_M)",
        "file": "gemma-4-E2B-it-Q5_K_M.gguf",
        "ctx": 4096,
        "gpu_layers": 99,
        "vram_mb": 1850,
    },
    "lfm3-1.7b": {
        "name": "LFM3-1.7B-Instruct (Q8_0)",
        "file": "LFM3-1.7B-Q8_0.gguf",
        "ctx": 4096,
        "gpu_layers": 99,
        "vram_mb": 1900,
    },
    "qwen2.5-7b": {
        "name": "Qwen2.5-7B-Instruct (Q6_K)",
        "file": "Qwen2.5-7B-Instruct-Q6_K.gguf",
        "ctx": 4096,
        "gpu_layers": 99,
        "vram_mb": 6200,
    },
}

DEFAULT_CLAIM_CATEGORIES = [
    "ACORD_FNOL_LOSS_NOTICE",
    "POLICE_ACCIDENT_REPORT",
    "MEDICAL_TREATMENT_RECORDS",
    "MEDICAL_BILLING_CMS1500_UB04",
    "VEHICLE_REPAIR_ESTIMATE_APPRAISAL",
    "DAMAGE_PHOTO_ATTACHMENTS",
    "WAGE_LOSS_EMPLOYMENT_VERIFICATION",
    "SWORN_STATEMENT_PROOF_OF_LOSS",
    "CLAIMANT_CORRESPONDENCE",
    "POLICY_DECLARATIONS_ENDORSEMENTS",
    "BUSINESS_DOCUMENT",
]


# ============================================================================
# 1. DATASET REGISTRY
# ============================================================================

class DatasetRegistry:
    """Loads standardized PSS benchmark datasets by label/identifier."""

    @staticmethod
    def get_supported_identifiers() -> List[str]:
        return ["tabmepp", "docsplit_poly_seq", "docsplit_mono_seq"]

    @staticmethod
    def load_dataset(
        identifier: str,
        split: str = "test",
        max_packets: Optional[int] = None,
        packet_index: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Loads benchmark packets by identifier."""
        identifier_lower = identifier.lower().strip()

        if identifier_lower == "tabmepp":
            path = DATA_DIR / "tabmepp_benchmark.json"
            if not path.exists():
                raise FileNotFoundError(f"TABME++ benchmark file not found at {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = "test_packets" if split == "test" else "train_packets"
            packets = data.get(key, [])
            desc = f"Official TABME++ Benchmark (UCSF Archival Folders, {split} split)"

        elif identifier_lower == "docsplit_poly_seq":
            path = DATA_DIR / "docsplit_seq_benchmark.json"
            if not path.exists():
                raise FileNotFoundError(f"DocSplit benchmark file not found at {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            packets = data.get("poly_seq_test", []) if split == "test" else data.get("train_packets", [])
            desc = f"DocSplit-Poly-Seq Benchmark (Multi-Category Contiguous Packets, {split} split)"

        elif identifier_lower == "docsplit_mono_seq":
            path = DATA_DIR / "docsplit_seq_benchmark.json"
            if not path.exists():
                raise FileNotFoundError(f"DocSplit benchmark file not found at {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            packets = data.get("mono_seq_test", []) if split == "test" else data.get("train_packets", [])
            desc = f"DocSplit-Mono-Seq Benchmark (Same-Category Contiguous Packets, {split} split)"

        else:
            supported = ", ".join(DatasetRegistry.get_supported_identifiers())
            raise ValueError(f"Unknown dataset identifier '{identifier}'. Supported: {supported}")

        if packet_index is not None:
            if 0 <= packet_index < len(packets):
                packets = [packets[packet_index]]
            else:
                raise IndexError(f"packet_index {packet_index} out of range (0..{len(packets)-1})")
        elif max_packets is not None and max_packets > 0:
            packets = packets[:max_packets]

        return packets, desc


# ============================================================================
# 2. ARIZE PHOENIX TELEMETRY & DIAGNOSTIC TRACER
# ============================================================================

class PhoenixTracer:
    """Manages Arize Phoenix OpenTelemetry tracing and offline error JSONL logging."""

    def __init__(
        self,
        enabled: bool = False,
        port: int = 6006,
        error_traces_path: Optional[Path] = None,
    ):
        self.enabled = enabled
        self.port = port
        self.error_traces_path = error_traces_path or (ROOT_DIR / "results" / "error_traces.jsonl")
        self.error_traces_path.parent.mkdir(parents=True, exist_ok=True)
        self.server_proc = None
        self.otel_tracer = None

        if self.enabled:
            self._init_phoenix()

    def _is_server_alive(self) -> bool:
        """Checks if local Phoenix server is responding to health checks."""
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/healthz", headers={"User-Agent": "PSS-Tracer"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _init_phoenix(self):
        """Initializes local Arize Phoenix session and OpenTelemetry tracer."""
        try:
            from phoenix.otel import register
            from opentelemetry import trace

            # Check if Phoenix server is already running
            if not self._is_server_alive():
                print(f"[Phoenix] Launching local Arize Phoenix server on http://127.0.0.1:{self.port}...")
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PHOENIX_HOST"] = "127.0.0.1"
                env["PHOENIX_PORT"] = str(self.port)
                env["PHOENIX_GRPC_PORT"] = "4317"
                
                self.server_proc = subprocess.Popen(
                    [sys.executable, "-m", "phoenix.server.main", "serve"],
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

                # Wait up to 12s for server startup
                deadline = time.time() + 12.0
                started = False
                while time.time() < deadline:
                    if self._is_server_alive():
                        started = True
                        break
                    time.sleep(0.5)

                if not started:
                    print(f"[Phoenix] Warning: Server did not respond within 12s; falling back to offline JSONL logging.")
                    self.enabled = False
                    return

            # Register OpenTelemetry tracer provider with Phoenix endpoint
            register(
                project_name="pss-insurance-claims",
                endpoint=f"http://127.0.0.1:{self.port}/v1/traces",
                set_global_tracer_provider=True,
            )
            self.otel_tracer = trace.get_tracer("pss_segmentation_tracer")
            print(f"[Phoenix] Telemetry active! View live traces & error spans at: http://127.0.0.1:{self.port}")
        except ImportError:
            print("[Phoenix] Note: 'arize-phoenix' not yet installed; falling back to offline JSONL logging.")
            self.enabled = False
        except Exception as e:
            print(f"[Phoenix] Warning: Could not initialize Phoenix UI ({e}); continuing with JSONL logging.")
            self.enabled = False

    def log_transition_trace(self, trace_record: Dict[str, Any]):
        """Logs transition telemetry to JSONL file and Arize Phoenix span."""
        # 1. Append to local JSONL for offline triage
        try:
            with open(self.error_traces_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record) + "\n")
        except Exception as e:
            print(f"[Telemetry] Warning writing JSONL trace: {e}")

        # 2. Log to Phoenix OpenTelemetry if active
        if self.enabled and self.otel_tracer:
            try:
                p_prev = trace_record.get("page_prev_num", 0)
                p_curr = trace_record.get("page_curr_num", 0)
                span_name = f"transition_p{p_prev}_to_p{p_curr}"
                with self.otel_tracer.start_as_current_span(span_name) as span:
                    pred = trace_record.get("prediction", {})
                    stage_attr = trace_record.get("stage_attribution", {})
                    outcome = pred.get("outcome_class", "UNANNOTATED")
                    diag = trace_record.get("diagnostic_snapshots", {})

                    is_split = bool(pred.get("is_split", False))
                    stage = str(pred.get("decision_stage", ""))
                    doc_type = pred.get("doc_type_id") or "UNKNOWN"
                    prob = float(pred.get("split_probability", 0.0))

                    # 1. Map to OpenInference Semantic Conventions for native Phoenix UI columns
                    is_llm = stage == "STAGE_4_LLM_ADJUDICATOR"
                    span_kind = "LLM" if is_llm else "DECISION"
                    span.set_attribute("openinference.span.kind", span_kind)

                    in_summary = f"P{p_prev} End: {diag.get('prev_page_tail', '')[:90]!r} -> P{p_curr} Start: {diag.get('curr_page_head', '')[:90]!r}"
                    span.set_attribute("input.value", in_summary)

                    out_summary = f"{'SPLIT' if is_split else 'CONTINUE'} | Stage: {stage} | P={prob:.2f}"
                    if is_split and doc_type != "UNKNOWN":
                        out_summary += f" | Doc: {doc_type}"
                    span.set_attribute("output.value", out_summary)

                    # 2. Native OpenTelemetry status (lights up green OK / red ERROR in Phoenix)
                    from opentelemetry.trace import StatusCode
                    is_error = outcome in ("FALSE_POSITIVE", "FALSE_NEGATIVE")
                    if is_error:
                        span.set_status(StatusCode.ERROR, f"Error: {outcome}")
                    else:
                        span.set_status(StatusCode.OK)

                    # 3. Custom attributes for filtering and deep-dive triage
                    span.set_attribute("packet_id", trace_record.get("packet_id", ""))
                    span.set_attribute("page_prev_num", p_prev)
                    span.set_attribute("page_curr_num", p_curr)
                    span.set_attribute("is_split", is_split)
                    span.set_attribute("split_probability", prob)
                    span.set_attribute("decision_stage", stage)
                    span.set_attribute("outcome_class", outcome)
                    span.set_attribute("is_error", is_error)

                    if pred.get("doc_type_id"):
                        span.set_attribute("doc_type_id", str(pred["doc_type_id"]))
                    if "gbdt_probability" in stage_attr and stage_attr["gbdt_probability"] is not None:
                        span.set_attribute("gbdt_prob", float(stage_attr["gbdt_probability"]))
                    if stage_attr.get("llm_reasoning"):
                        span.set_attribute("llm_reasoning", str(stage_attr["llm_reasoning"]))
                    if stage_attr.get("llm_confidence") is not None:
                        span.set_attribute("llm_confidence", float(stage_attr["llm_confidence"]))
                    if stage_attr.get("llm_latency_sec") is not None:
                        span.set_attribute("llm_latency_sec", float(stage_attr["llm_latency_sec"]))
                    if is_error:
                        span.set_attribute("error_type", outcome)
            except Exception:
                pass


# ============================================================================
# 3. STAGE 1 & 2: FEATURE EXTRACTION & FAST-PATH RULES
# ============================================================================

def extract_page_signature(page: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts layout, typographical, and continuity features from a page."""
    text = (page.get("full_text", "") or "").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    top_text = " ".join(lines[:4]) if lines else ""
    bottom_text = " ".join(lines[-4:]) if lines else ""

    # Page number detection
    exp_num = None
    has_page_one = False
    has_cont = False

    for c in page.get("corner_numbers", []):
        raw_c = str(c.get("text", "")).strip()
        m_of = re.search(r"(\d+)\s*(?:of|/)\s*(\d+)", raw_c, re.I)
        if m_of:
            p_cur, p_tot = int(m_of.group(1)), int(m_of.group(2))
            exp_num = p_cur
            if p_cur == 1:
                has_page_one = True
            elif p_cur > 1:
                has_cont = True
            break
        m_p = re.search(r"\b(?:page|pg\.?|p\.?)\s*(\d+)\b", raw_c, re.I)
        if m_p:
            exp_num = int(m_p.group(1))
            if exp_num == 1:
                has_page_one = True
            elif exp_num > 1:
                has_cont = True
            break

    # Title trigger / Standard form banner detection
    has_title = False
    upper_count = 0
    title_regex = re.compile(
        r"\b(INVOICE|STATEMENT|AGREEMENT|CONTRACT|RESUME|MEMORANDUM|REPORT|POLICY|CERTIFICATE|"
        r"ACORD|CMS-1500|UB-04|CRASH REPORT|CLAIM FORM|POLICE DEPARTMENT|NOTICE OF LOSS)\b",
        re.I,
    )
    for ln in lines[:3]:
        if title_regex.search(ln):
            has_title = True
            break
        if len(ln) >= 5 and ln.isupper():
            upper_count += 1

    if upper_count >= 1:
        has_title = True

    # Closing / Signature triggers
    has_closing = False
    closing_regex = re.compile(
        r"\b(SIGNATURE|DATED|TOTAL AMOUNT|BALANCE DUE|SINCERELY|RESPECTFULLY|NOTARY PUBLIC|"
        r"SWORN TO|AUTHORIZED SIGNATURE|GRAND TOTAL|PAGE TOTAL)\b",
        re.I,
    )
    for ln in lines[-4:]:
        if closing_regex.search(ln):
            has_closing = True
            break

    # Sentence boundary continuity
    ends_mid = False
    if lines:
        last_ln = lines[-1]
        if last_ln and last_ln[-1] not in ".!?:;\"'":
            ends_mid = True

    starts_mid = False
    if lines:
        first_ln = lines[0]
        if first_ln and first_ln[0].islower():
            starts_mid = True

    # Entity / Document IDs
    id_pattern = re.compile(r"\b([A-Z]{1,4}[-#]?\d{3,10}|\d{3}-\d{2}-\d{4}|INV-\d+|POL-\d+|CLM-\d+)\b")
    found_ids = set(id_pattern.findall(text))

    return {
        "num_lines": len(lines),
        "bottom_whitespace": float(page.get("bottom_whitespace", 0.0)),
        "top_whitespace": float(page.get("top_whitespace", 0.0)),
        "first_line_height": float(page.get("first_line_height", 0.0)),
        "max_top_line_height": float(page.get("max_top_line_height", 0.0)),
        "centered_top_lines": int(page.get("centered_top_lines", 0)),
        "exp_page_num": exp_num,
        "has_page_one_marker": has_page_one,
        "has_cont_marker": has_cont,
        "has_title_trigger": has_title,
        "has_closing_trigger": has_closing,
        "ends_mid_sentence": ends_mid,
        "starts_mid_sentence": starts_mid,
        "top_text": top_text,
        "bottom_text": bottom_text,
        "ids": found_ids,
        "full_text": text,
    }


def compute_packet_tfidf_similarities(pages: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    """Computes lexical and header cosine similarity transitions across adjacent pages."""
    n = len(pages)
    if n <= 1:
        return [0.0] * n, [0.0] * n

    full_texts = [(p.get("full_text", "") or "").lower() for p in pages]
    hdr_texts = [" ".join((p.get("full_text", "") or "").lower().splitlines()[:3]) for p in pages]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec_full = TfidfVectorizer(max_features=400, stop_words="english")
    try:
        tfidf_full = vec_full.fit_transform(full_texts)
        full_sims = [1.0]
        for i in range(1, n):
            sim = float(cosine_similarity(tfidf_full[i - 1], tfidf_full[i])[0, 0])
            full_sims.append(round(sim, 4))
    except Exception:
        full_sims = [0.0] * n

    vec_hdr = TfidfVectorizer(max_features=200, stop_words="english")
    try:
        tfidf_hdr = vec_hdr.fit_transform(hdr_texts)
        hdr_sims = [1.0]
        for i in range(1, n):
            sim = float(cosine_similarity(tfidf_hdr[i - 1], tfidf_hdr[i])[0, 0])
            hdr_sims.append(round(sim, 4))
    except Exception:
        hdr_sims = [0.0] * n

    return full_sims, hdr_sims


def extract_30f_pairwise_vector(
    sp: Dict[str, Any], sc: Dict[str, Any], full_sim: float, hdr_sim: float, run_len: int
) -> List[float]:
    """Extracts 30-feature spatial & continuity vector for Stage-3 GBDT."""
    shared_ids = len(sp["ids"].intersection(sc["ids"]))
    prev_only = len(sp["ids"] - sc["ids"])
    curr_only = len(sc["ids"] - sp["ids"])

    return [
        full_sim,
        hdr_sim,
        float(run_len),
        float(sp["bottom_whitespace"]),
        float(sc["top_whitespace"]),
        float(sc["first_line_height"]),
        float(sc["max_top_line_height"]),
        float(sc["centered_top_lines"]),
        1.0 if sp["has_closing_trigger"] else 0.0,
        1.0 if sc["has_title_trigger"] else 0.0,
        1.0 if sc["has_page_one_marker"] else 0.0,
        1.0 if sc["has_cont_marker"] else 0.0,
        1.0 if sp["has_cont_marker"] else 0.0,
        1.0 if sp["ends_mid_sentence"] and sc["starts_mid_sentence"] else 0.0,
        float(abs(sp["num_lines"] - sc["num_lines"])),
        float(sc["num_lines"]),
        float(sp["num_lines"]),
        float(shared_ids),
        float(prev_only),
        float(curr_only),
        1.0 if shared_ids > 0 and curr_only == 0 else 0.0,
        1.0 if curr_only > 0 and shared_ids == 0 else 0.0,
        float(sp["bottom_whitespace"] * (1.0 - full_sim)),
        float(sc["first_line_height"] * (1.0 - hdr_sim)),
        1.0 if (sp["exp_page_num"] and sc["exp_page_num"] and sc["exp_page_num"] == sp["exp_page_num"] + 1) else 0.0,
        1.0 if (sc["exp_page_num"] and sc["exp_page_num"] == 1) else 0.0,
        1.0 if (sc["exp_page_num"] and sc["exp_page_num"] > 1) else 0.0,
        1.0 if run_len > 10 else 0.0,
        1.0 if run_len == 1 else 0.0,
        float(min(run_len, 20) / 20.0),
    ]


# ============================================================================
# 4. STAGE 4: LOCAL ADJUDICATOR & DOCUMENT CLASSIFIER (Zero Distributed RPC)
# ============================================================================

class LocalLlamaServerManager:
    """Manages local llama-server on a single laptop GPU (RTX 4060, 8GB VRAM)."""

    def __init__(self, model_config: Dict[str, Any], port: int = 8080):
        self.config = model_config
        self.port = port
        self.proc: Optional[subprocess.Popen] = None

    def start(self):
        """Starts local llama-server process."""
        model_file = self.config["file"]
        model_path = MODELS_DIR / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        llama_server_bin = ROOT_DIR / "bin" / "llama-cuda" / "llama-server.exe"
        if not llama_server_bin.exists():
            llama_server_bin = ROOT_DIR / "bin" / "llama-server.exe"
        if not llama_server_bin.exists():
            raise FileNotFoundError(f"llama-server binary not found at {llama_server_bin}")

        cmd = [
            str(llama_server_bin),
            "-m", str(model_path),
            "-c", str(self.config.get("ctx", 4096)),
            "-ngl", str(self.config.get("gpu_layers", 99)),
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--reasoning", "off",
            "--reasoning-format", "none",
        ]

        print(f"[LocalServer] Starting {self.config['name']} on 127.0.0.1:{self.port} (Single Laptop RTX 4060)...")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for readiness
        import requests
        url = f"http://127.0.0.1:{self.port}/health"
        t0 = time.time()
        while time.time() - t0 < 60:
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    print(f"[LocalServer] {self.config['name']} is READY!")
                    return
            except Exception:
                pass
            time.sleep(1)

        raise TimeoutError(f"Local server failed to start within 60s on port {self.port}")

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 150) -> str:
        """Sends chat completion to local server."""
        import requests
        url = f"http://127.0.0.1:{self.port}/v1/chat/completions"
        combined = f"{system_prompt}\n\n{user_prompt}"
        payload = {
            "messages": [
                {"role": "user", "content": combined},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(url, json=payload, timeout=45)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
        except Exception as e:
            return f"ERROR: {e}"

    def stop(self):
        """Stops local llama-server."""
        if self.proc:
            print("[LocalServer] Stopping local llama-server...")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None


# ============================================================================
# 5. CORE 4-STAGE PIPELINE SEGMENTER
# ============================================================================

class PssPipelineSegmenter:
    """Production 4-Stage Multimodal PSS Pipeline with Phoenix telemetry."""

    def __init__(
        self,
        model_pkl: Path = MODEL_PKL,
        model_key: str = "gemma-4-e4b",
        threshold_low: float = 0.20,
        threshold_high: float = 0.80,
        tracer: Optional[PhoenixTracer] = None,
    ):
        self.model_pkl = model_pkl
        self.model_key = model_key
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.tracer = tracer or PhoenixTracer(enabled=False)

        self.classifier_artifact = None
        if model_pkl.exists():
            with open(model_pkl, "rb") as f:
                self.classifier_artifact = pickle.load(f)

    def process_packet(
        self,
        pages_data: List[Dict[str, Any]],
        packet_id: str = "unnamed_packet",
        mode: str = "hybrid",
        always_llm: bool = False,
        server: Optional[LocalLlamaServerManager] = None,
        ground_truth_starts: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Processes a packet through the 4-stage pipeline."""
        n = len(pages_data)
        if n == 0:
            return []

        sigs = [extract_page_signature(p) for p in pages_data]
        full_sims, hdr_sims = compute_packet_tfidf_similarities(pages_data)

        gbdt = self.classifier_artifact.get("gbdt") if self.classifier_artifact else None
        lr = self.classifier_artifact.get("lr") if self.classifier_artifact else None
        scaler = self.classifier_artifact.get("scaler") if self.classifier_artifact else None

        transitions = []
        run_len = 1

        for i in range(1, n):
            sp, sc = sigs[i - 1], sigs[i]
            n_prev, n_curr = sp["exp_page_num"], sc["exp_page_num"]
            from_pg = i
            to_pg = i + 1
            gt_split = (to_pg in ground_truth_starts) if ground_truth_starts else None

            # -------------------------------------------------------------
            # STAGE 2: DETERMINISTIC FAST-PATH FILTER (<1ms)
            # (Bypassed if always_llm is active to measure LLM accuracy ceiling)
            # -------------------------------------------------------------
            if not always_llm:
                if n_prev is not None and n_curr is not None and n_curr == n_prev + 1:
                    t_record = {
                        "from_page": from_pg,
                        "to_page": to_pg,
                        "split": False,
                        "prob": 0.01,
                        "reason": f"Sequential page number ({n_prev} -> {n_curr})",
                        "stage": "STAGE_2_FAST_PATH",
                        "rule": "sequential_pagination",
                        "doc_type_id": None,
                    }
                    transitions.append(t_record)
                    self._record_telemetry(packet_id, from_pg, to_pg, t_record, sp, sc, gt_split)
                    run_len += 1
                    continue

                # Only trigger continuation if it has an explicit 'Page X of Y' or 'Page X' marker that matches sequence
                if sc["has_cont_marker"] and not sc["has_page_one_marker"] and (n_prev is None or n_curr == n_prev + 1 or (n_curr <= 5 and n_prev is None)):
                    t_record = {
                        "from_page": from_pg,
                        "to_page": to_pg,
                        "split": False,
                        "prob": 0.05,
                        "reason": f"Explicit continuation page marker ({n_curr})",
                        "stage": "STAGE_2_FAST_PATH",
                        "rule": "continuation_page_marker",
                        "doc_type_id": None,
                    }
                    transitions.append(t_record)
                    self._record_telemetry(packet_id, from_pg, to_pg, t_record, sp, sc, gt_split)
                    run_len += 1
                    continue

                if sc["has_page_one_marker"] and not sc["has_cont_marker"]:
                    t_record = {
                        "from_page": from_pg,
                        "to_page": to_pg,
                        "split": True,
                        "prob": 0.99,
                        "reason": "Explicit Page 1 / Title header marker detected",
                        "stage": "STAGE_2_FAST_PATH",
                        "rule": "page_one_marker",
                        "doc_type_id": None,
                    }
                    transitions.append(t_record)
                    self._record_telemetry(packet_id, from_pg, to_pg, t_record, sp, sc, gt_split)
                    run_len = 1
                    continue

            # -------------------------------------------------------------
            # STAGE 3: SPATIAL 30-FEATURE GBDT CLASSIFIER (<2ms)
            # -------------------------------------------------------------
            p_gbdt = 0.50
            if gbdt is not None and lr is not None and scaler is not None:
                feat = np.array(
                    [extract_30f_pairwise_vector(sp, sc, full_sims[i], hdr_sims[i], run_len)],
                    dtype=np.float32,
                )
                p_g = float(gbdt.predict_proba(feat)[0, 1])
                p_l = float(lr.predict_proba(scaler.transform(feat))[0, 1])
                p_gbdt = 0.75 * p_g + 0.25 * p_l
            else:
                p_gbdt = 0.85 if (full_sims[i] < 0.18 or sc["has_title_trigger"]) else 0.15

            # High confidence cut or fast mode bypass (bypassed if always_llm is active)
            if not always_llm:
                if p_gbdt > self.threshold_high or mode == "fast" or server is None:
                    is_split = p_gbdt >= 0.50
                    t_record = {
                        "from_page": from_pg,
                        "to_page": to_pg,
                        "split": is_split,
                        "prob": round(p_gbdt, 4),
                        "reason": f"GBDT spatial/continuity classifier (P={p_gbdt:.3f})",
                        "stage": "STAGE_3_SPATIAL_GBDT",
                        "rule": "gbdt_high_confidence" if (p_gbdt > self.threshold_high or p_gbdt < self.threshold_low) else "gbdt_standard",
                        "doc_type_id": None,
                    }
                    transitions.append(t_record)
                    self._record_telemetry(packet_id, from_pg, to_pg, t_record, sp, sc, gt_split)
                    run_len = 1 if is_split else run_len + 1
                    continue

                # High confidence continuation from GBDT
                if p_gbdt < self.threshold_low:
                    t_record = {
                        "from_page": from_pg,
                        "to_page": to_pg,
                        "split": False,
                        "prob": round(p_gbdt, 4),
                        "reason": f"GBDT strong layout continuation (P={p_gbdt:.3f})",
                        "stage": "STAGE_3_SPATIAL_GBDT",
                        "rule": "gbdt_strong_continue",
                        "doc_type_id": None,
                    }
                    transitions.append(t_record)
                    self._record_telemetry(packet_id, from_pg, to_pg, t_record, sp, sc, gt_split)
                    run_len += 1
                    continue

            # Fallback if server is not available
            if server is None:
                is_split = p_gbdt >= 0.50
                t_record = {
                    "from_page": from_pg,
                    "to_page": to_pg,
                    "split": is_split,
                    "prob": round(p_gbdt, 4),
                    "reason": f"GBDT fallback (server None, P={p_gbdt:.3f})",
                    "stage": "STAGE_3_SPATIAL_GBDT",
                    "rule": "gbdt_fallback",
                    "doc_type_id": None,
                }
                transitions.append(t_record)
                self._record_telemetry(packet_id, from_pg, to_pg, t_record, sp, sc, gt_split)
                run_len = 1 if is_split else run_len + 1
                continue

            # -------------------------------------------------------------
            # STAGE 4: LOCAL ADJUDICATOR & DOCUMENT CLASSIFIER
            # -------------------------------------------------------------
            sys_prompt = (
                "You are an expert Document Packet Segmentation & Classification adjudicator evaluating a boundary between Page t-1 and Page t.\n"
                "You must decide if Page t begins a NEW independent document (SPLIT) or continues the current document (CONTINUE).\n"
                "Sequential page numbering (e.g. 1 -> 2) or sentences running across pages indicate CONTINUE.\n"
                "New document titles, form headers, new author/inmate/patient details, or Page 1 markers indicate SPLIT.\n"
                "If SPLIT, also classify the new document category from: " + ", ".join(DEFAULT_CLAIM_CATEGORIES) + ".\n"
                "Reply ONLY with a JSON object: {\"decision\": \"SPLIT\" or \"CONTINUE\", \"confidence\": 0.5 to 1.0, \"doc_type_id\": \"...\", \"reason\": \"short explanation\"}"
            )
            shared_ids = sorted(list(sp["ids"].intersection(sc["ids"])))[:4]
            prev_only = sorted(list(sp["ids"] - sc["ids"]))[:4]
            curr_only = sorted(list(sc["ids"] - sp["ids"]))[:4]

            exp_info = []
            if n_prev is not None and n_curr is not None:
                exp_info.append(f"SeqPages: {n_prev}->{n_curr}")
            elif n_curr is not None:
                exp_info.append(f"ExtractedCurrPage={n_curr}")
            if sc.get("has_page_one_marker"):
                exp_info.append("Page1Marker=True")
            if sc.get("has_cont_marker"):
                exp_info.append("ContMarker=True")
            exp_meta = f" | {', '.join(exp_info)}" if exp_info else ""

            user_prompt = (
                f"Transition Page {from_pg} -> Page {to_pg} (Prior GBDT Split Prob = {p_gbdt:.2f}{exp_meta}):\n"
                f"Page {from_pg} (t-1): Lines={sp['num_lines']}, BotWhitespace={sp['bottom_whitespace']:.2f}, ClosingTrigger={sp['has_closing_trigger']}, EndsMidSent={sp['ends_mid_sentence']}\n"
                f"  Top: {sp['top_text'][:160]!r}\n  Bottom: {sp['bottom_text'][:160]!r}\n"
                f"Page {to_pg} (t): Lines={sc['num_lines']}, TopWhitespace={sc['top_whitespace']:.2f}, TitleTrigger={sc['has_title_trigger']}, StartsMidSent={sc['starts_mid_sentence']}\n"
                f"  Top: {sc['top_text'][:200]!r}\n"
                f"Continuity: LexicalSim={full_sims[i]:.3f}, HeaderSim={hdr_sims[i]:.3f}, SharedIDs={shared_ids}, PrevOnly={prev_only}, CurrOnly={curr_only}\n"
                f"Output JSON:"
            )

            t0_llm = time.perf_counter()
            resp = server.chat(sys_prompt, user_prompt, max_tokens=110)
            llm_latency = time.perf_counter() - t0_llm

            resp_u = resp.upper()
            llm_split = bool("\"SPLIT\"" in resp_u or (resp_u.find("SPLIT") != -1 and resp_u.find("CONTINUE") == -1))

            m_conf = re.search(r"\"confidence\"\s*:\s*([0-9.]+)", resp)
            conf = float(m_conf.group(1)) if m_conf else 0.75
            conf = max(0.5, min(0.98, conf))
            llm_prob = conf if llm_split else (1.0 - conf)

            m_doc = re.search(r"\"doc_type_id\"\s*:\s*\"([^\"]+)\"", resp)
            doc_type = m_doc.group(1) if m_doc else ("NEW_DOCUMENT" if llm_split else None)

            m_reason = re.search(r"\"reason\"\s*:\s*\"([^\"]+)\"", resp)
            llm_reason = m_reason.group(1) if m_reason else ("LLM: SPLIT" if llm_split else "LLM: CONTINUE")

            is_split = llm_split
            fused_prob = llm_prob

            t_record = {
                "from_page": from_pg,
                "to_page": to_pg,
                "split": is_split,
                "prob": round(fused_prob, 4),
                "reason": f"Stage 4 Adjudication: {llm_reason}",
                "stage": "STAGE_4_LLM_ADJUDICATOR",
                "rule": "always_llm_adjudication" if always_llm else "llm_joint_adjudication",
                "doc_type_id": doc_type if is_split else None,
                "llm_confidence": conf,
                "llm_latency_sec": round(llm_latency, 3),
            }
            transitions.append(t_record)
            self._record_telemetry(packet_id, from_pg, to_pg, t_record, sp, sc, gt_split, llm_reason=llm_reason)
            run_len = 1 if is_split else run_len + 1

        return transitions

    def _record_telemetry(
        self,
        packet_id: str,
        from_pg: int,
        to_pg: int,
        t_record: Dict[str, Any],
        sp: Dict[str, Any],
        sc: Dict[str, Any],
        gt_split: Optional[bool],
        llm_reason: Optional[str] = None,
    ):
        """Constructs and emits a diagnostic transition telemetry record."""
        pred_split = t_record["split"]
        outcome = "UNANNOTATED"
        err_tag = None

        if gt_split is not None:
            if pred_split and gt_split:
                outcome = "TRUE_POSITIVE"
            elif not pred_split and not gt_split:
                outcome = "TRUE_NEGATIVE"
            elif pred_split and not gt_split:
                outcome = "FALSE_POSITIVE"
                err_tag = "FP_OVERSPLIT"
            elif not pred_split and gt_split:
                outcome = "FALSE_NEGATIVE"
                err_tag = "FN_MISSED_SPLIT"

        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "packet_id": packet_id,
            "page_prev_num": from_pg,
            "page_curr_num": to_pg,
            "ground_truth": {"is_split": gt_split},
            "prediction": {
                "is_split": pred_split,
                "doc_type_id": t_record.get("doc_type_id"),
                "decision_stage": t_record["stage"],
                "outcome_class": outcome,
                "confidence": t_record.get("prob"),
            },
            "stage_attribution": {
                "rule_name": t_record.get("rule"),
                "gbdt_probability": t_record.get("prob"),
                "llm_reasoning": llm_reason,
            },
            "diagnostic_snapshots": {
                "prev_page_head": sp["top_text"][:120],
                "prev_page_tail": sp["bottom_text"][:100],
                "curr_page_head": sc["top_text"][:120],
                "extracted_pagination": {"prev": sp["exp_page_num"], "curr": sc["exp_page_num"]},
            },
            "error_taxonomy_tag": err_tag,
        }
        self.tracer.log_transition_trace(record)


# ============================================================================
# 6. EVALUATION METRICS ENGINE
# ============================================================================

def evaluate_packet_metrics(
    packets: List[Dict[str, Any]], all_transitions: List[List[Dict[str, Any]]]
) -> Dict[str, float]:
    """Computes standard PSS evaluation metrics (Boundary F1, Doc Exact Match F1, STP, P_k)."""
    tp = fp = fn = tn = 0
    doc_exact_tp = doc_exact_fp = doc_exact_fn = 0
    perfect_packets = 0
    total_packets = len(packets)
    pk_errors = []
    windowdiff_errors = []

    for pkt, transitions in zip(packets, all_transitions):
        n = len(pkt["pages"])
        pred_flags = [1] + [1 if t["split"] else 0 for t in transitions]
        gt_flags = [p.get("is_first_page", 1 if i == 0 else 0) for i, p in enumerate(pkt["pages"])]

        # Boundary precision/recall (indices 1..n-1)
        for i in range(1, n):
            p_val, g_val = pred_flags[i], gt_flags[i]
            if p_val == 1 and g_val == 1:
                tp += 1
            elif p_val == 1 and g_val == 0:
                fp += 1
            elif p_val == 0 and g_val == 1:
                fn += 1
            else:
                tn += 1

        # Document exact span matching
        pred_spans = []
        cur_start = 1
        for i in range(1, n):
            if pred_flags[i] == 1:
                pred_spans.append((cur_start, i))
                cur_start = i + 1
        pred_spans.append((cur_start, n))

        gt_spans = []
        if "subdocuments" in pkt and pkt["subdocuments"]:
            for d in pkt["subdocuments"]:
                gt_spans.append((d["start_page"], d["end_page"]))
        else:
            cur_start = 1
            for i in range(1, n):
                if gt_flags[i] == 1:
                    gt_spans.append((cur_start, i))
                    cur_start = i + 1
            gt_spans.append((cur_start, n))

        gt_set = set(gt_spans)
        pred_set = set(pred_spans)

        exact_matches = len(gt_set.intersection(pred_set))
        doc_exact_tp += exact_matches
        doc_exact_fp += len(pred_set - gt_set)
        doc_exact_fn += len(gt_set - pred_set)

        if pred_flags == gt_flags:
            perfect_packets += 1

        # P_k and WindowDiff
        k = max(2, int(round((n / max(1, len(gt_spans))) / 2.0)))
        pk_err = 0
        wd_err = 0
        for i in range(n - k):
            gt_diff = gt_flags[i + 1 : i + k + 1].count(1)
            pred_diff = pred_flags[i + 1 : i + k + 1].count(1)
            if (gt_diff > 0) != (pred_diff > 0):
                pk_err += 1
            if gt_diff != pred_diff:
                wd_err += 1
        total_probes = max(1, n - k)
        pk_errors.append(pk_err / total_probes)
        windowdiff_errors.append(wd_err / total_probes)

    b_prec = tp / max(1, tp + fp) * 100.0
    b_rec = tp / max(1, tp + fn) * 100.0
    b_f1 = (2 * b_prec * b_rec) / max(1e-6, b_prec + b_rec)

    d_prec = doc_exact_tp / max(1, doc_exact_tp + doc_exact_fp) * 100.0
    d_rec = doc_exact_tp / max(1, doc_exact_tp + doc_exact_fn) * 100.0
    d_f1 = (2 * d_prec * d_rec) / max(1e-6, d_prec + d_rec)

    stp = (perfect_packets / max(1, total_packets)) * 100.0
    avg_pk = float(np.mean(pk_errors) * 100.0) if pk_errors else 0.0
    avg_wd = float(np.mean(windowdiff_errors) * 100.0) if windowdiff_errors else 0.0

    return {
        "boundary_precision": round(b_prec, 2),
        "boundary_recall": round(b_rec, 2),
        "boundary_f1": round(b_f1, 2),
        "doc_exact_precision": round(d_prec, 2),
        "doc_exact_recall": round(d_rec, 2),
        "doc_exact_f1": round(d_f1, 2),
        "packet_stp_pct": round(stp, 1),
        "pk_error": round(avg_pk, 2),
        "windowdiff_error": round(avg_wd, 2),
        "total_packets": total_packets,
        "perfect_packets": perfect_packets,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ============================================================================
# 7. CLI COMMAND-LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Production Page Stream Segmentation & Document Classification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Target Input: either a dataset identifier OR a raw PDF file
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=DatasetRegistry.get_supported_identifiers(),
        help="Load benchmark dataset by label (tabmepp, docsplit_poly_seq, docsplit_mono_seq)",
    )
    group.add_argument(
        "--pdf",
        type=Path,
        help="Path to a raw multi-page PDF file to segment",
    )

    # Pipeline & Model Settings
    parser.add_argument(
        "--mode",
        choices=["fast", "hybrid", "llm"],
        default="hybrid",
        help="Pipeline mode: 'fast' (Rules + GBDT only), 'hybrid' (Rules + GBDT + Stage 4 LLM), or 'llm' (Always call Stage 4 LLM)",
    )
    parser.add_argument(
        "--always-llm",
        action="store_true",
        help="Force Stage 4 LLM adjudication on every page transition (bypasses Stage 2 and Stage 3 short-circuits to evaluate accuracy ceiling)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-4-e4b",
        help="Stage 4 LLM model key ('gemma-4-e4b', 'gemma-4-e2b', 'lfm3-1.7b', 'qwen2.5-7b') or custom GGUF path",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.20,
        help="Lower GBDT boundary threshold (P < low => CONTINUE)",
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=0.80,
        help="Upper GBDT boundary threshold (P > high => SPLIT)",
    )

    # Testing & Slicing Options
    parser.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Limit evaluation to first N packets (useful for smoke tests)",
    )
    parser.add_argument(
        "--packet-index",
        type=int,
        default=None,
        help="Evaluate only a specific packet index (0-indexed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print segmentation metrics without writing files to disk",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for physically sliced subdocument PDFs",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Export structured evaluation results to JSON file",
    )

    # Telemetry & Arize Phoenix
    parser.add_argument(
        "--enable-phoenix",
        action="store_true",
        help="Launch and log live OpenTelemetry traces to local Arize Phoenix UI",
    )
    parser.add_argument(
        "--phoenix-port",
        type=int,
        default=6006,
        help="Local web UI port for Arize Phoenix",
    )
    parser.add_argument(
        "--error-trace",
        type=Path,
        default=ROOT_DIR / "results" / "error_traces.jsonl",
        help="Filepath to append granular transition telemetry for error analysis",
    )

    args = parser.parse_args()

    is_always_llm = bool(args.always_llm or (args.mode == "llm"))
    mode_str = "LLM-CEILING (ALWAYS-LLM)" if is_always_llm else args.mode.upper()

    print("=" * 70)
    print("PRODUCTION PAGE STREAM SEGMENTATION & CLASSIFICATION PIPELINE")
    print(f"Mode: {mode_str} | Model: {args.model} | Device: Local Laptop (RTX 4060 8GB)")
    print("=" * 70)

    # Initialize Phoenix telemetry
    tracer = PhoenixTracer(
        enabled=args.enable_phoenix,
        port=args.phoenix_port,
        error_traces_path=args.error_trace,
    )

    # Initialize Pipeline Segmenter
    segmenter = PssPipelineSegmenter(
        model_key=args.model,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        tracer=tracer,
    )

    # Resolve Stage 4 Server if in Hybrid or Always-LLM Mode
    server = None
    if args.mode in ("hybrid", "llm") or is_always_llm:
        model_cfg = LOCAL_MODELS_REGISTRY.get(args.model)
        if model_cfg is None:
            # Custom GGUF path fallback
            p = Path(args.model)
            if p.exists():
                model_cfg = {"name": p.stem, "file": p.name, "ctx": 4096, "gpu_layers": 99}
            else:
                model_cfg = LOCAL_MODELS_REGISTRY["gemma-4-e4b"]
        server = LocalLlamaServerManager(model_cfg, port=8080)

    # Load Data
    if args.dataset:
        packets, desc = DatasetRegistry.load_dataset(
            args.dataset, split="test", max_packets=args.max_packets, packet_index=args.packet_index
        )
        print(f"\n[Dataset] Loaded {len(packets)} packets from: {desc}")
    else:
        # Load single PDF
        if not args.pdf.exists():
            print(f"Error: PDF file {args.pdf} not found.", file=sys.stderr)
            sys.exit(1)
        reader = PdfReader(str(args.pdf))
        pages_data = []
        for idx, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            pages_data.append({
                "packet_page_num": idx + 1,
                "full_text": txt,
                "bottom_whitespace": 0.15,
                "top_whitespace": 0.05,
                "first_line_height": 0.02,
                "max_top_line_height": 0.02,
                "centered_top_lines": 0,
                "corner_numbers": [],
            })
        packets = [{
            "packet_id": args.pdf.stem,
            "pages": pages_data,
            "subdocuments": [],
        }]
        print(f"\n[PDF] Loaded {len(pages_data)} pages from {args.pdf.name}")

    # Execute Pipeline
    all_transitions = []
    t0_total = time.perf_counter()

    try:
        if server is not None:
            server.start()

        for idx, pkt in enumerate(packets):
            pkt_id = pkt.get("packet_id", f"pkt_{idx+1}")
            pages = pkt["pages"]
            print(f"\n[{idx+1}/{len(packets)}] Processing '{pkt_id}' ({len(pages)} pages)...", flush=True)

            # Ground truth start pages if available
            gt_starts = set()
            if "subdocuments" in pkt and pkt["subdocuments"]:
                for d in pkt["subdocuments"]:
                    gt_starts.add(d["start_page"])
            else:
                for p_idx, p in enumerate(pages):
                    if p.get("is_first_page", 1 if p_idx == 0 else 0) == 1:
                        gt_starts.add(p_idx + 1)

            t0_pkt = time.perf_counter()
            transitions = segmenter.process_packet(
                pages,
                packet_id=pkt_id,
                mode=args.mode,
                always_llm=is_always_llm,
                server=server,
                ground_truth_starts=gt_starts,
            )
            pkt_time = time.perf_counter() - t0_pkt

            split_count = sum(1 for t in transitions if t["split"]) + 1
            print(f"  -> Segmented into {split_count} documents in {pkt_time:.2f}s ({len(pages)/max(0.01, pkt_time):.1f} pages/sec)")
            all_transitions.append(transitions)

    finally:
        if server is not None:
            server.stop()

    total_time = time.perf_counter() - t0_total

    # Compute Metrics if Ground Truth is Available
    has_gt = any("subdocuments" in p and p["subdocuments"] for p in packets) or any(
        "is_first_page" in p["pages"][0] for p in packets if p["pages"]
    )
    if has_gt:
        metrics = evaluate_packet_metrics(packets, all_transitions)
        print("\n" + "=" * 70)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 70)
        print(f"  Boundary F1:            {metrics['boundary_f1']:.2f}% (Precision: {metrics['boundary_precision']:.2f}%, Recall: {metrics['boundary_recall']:.2f}%)")
        print(f"  Doc Exact Match F1:     {metrics['doc_exact_f1']:.2f}% (Precision: {metrics['doc_exact_precision']:.2f}%, Recall: {metrics['doc_exact_recall']:.2f}%)")
        print(f"  Packet STP (Perfect):   {metrics['packet_stp_pct']:.1f}% ({metrics['perfect_packets']}/{metrics['total_packets']} packets)")
        print(f"  $P_k$ Error Rate:       {metrics['pk_error']:.2f}%")
        print(f"  WindowDiff Error Rate:  {metrics['windowdiff_error']:.2f}%")
        print(f"  True Positives: {metrics['tp']} | False Positives: {metrics['fp']} | False Negatives: {metrics['fn']}")
        print(f"  Total Wall Time:        {total_time:.2f}s (Average: {total_time/len(packets):.2f}s per packet)")
        print("=" * 70)

        if args.json:
            out_json = {
                "benchmark_name": args.dataset or args.pdf.stem,
                "mode": "llm" if is_always_llm else args.mode,
                "always_llm": is_always_llm,
                "model": args.model,
                "thresholds": {"low": args.threshold_low, "high": args.threshold_high},
                "total_time_seconds": round(total_time, 2),
                "metrics": metrics,
            }
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(out_json, f, indent=2)
            print(f"[Export] Saved results JSON to: {args.json}")

    print(f"\n[Telemetry] Granular transition traces logged to: {args.error_trace}")
    if args.enable_phoenix and tracer.enabled:
        print(f"[Phoenix] OpenTelemetry dashboard active at: http://127.0.0.1:{tracer.port}")


if __name__ == "__main__":
    main()
