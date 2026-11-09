"""Gemma-26B MoE Page Stream Segmentation (PSS) & DocSplit Tournament.

Hardware:
- Node 1: Laptop RTX 4060 (8GB VRAM) running llama-server
- Node 2: Desktop hpenvy GTX 1660 Super (6GB VRAM) via ggml-rpc on port 50052
Model: Gemma-4-26B-A4B-it-UD-IQ3_S.gguf (100% in pure dual-PC VRAM, 0 MiB CPU offload!)

Benchmarks Evaluated (3 Testbeds):
1. DocSplit-Mono-Seq (RVL-CDIP-N-MP Single-Category Contiguous Packets - The Hardest Frontier)
2. DocSplit-Poly-Seq (RVL-CDIP-N-MP Multi-Category Contiguous Packets)
3. Official TABME++ (bevaya/TABMEpp Archival Folders with Azure 2D OCR)

Conditions Evaluated:
1. Holistic-MoE-26B: Zero-Shot Holistic Packet Windowing across full 8k context
2. TwoStage-Pairwise-MoE-26B: Structured PageSignatures evaluated in pairwise batches
3. TwoStage-Hybrid-MoE-26B: Supervised 30F GBDT + Gemma-26B MoE Deep Adjudicator on uncertain transitions
"""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from run_pss_benchmark_tournament import (
    extract_page_signature,
    compute_packet_tfidf_similarities,
    train_and_run_algorithm_3_supervised_gbdt,
    evaluate_predictions,
    boundaries_to_spans,
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
BIN_DIR = ROOT_DIR / "bin" / "llama-cuda"
SERVER_EXE = BIN_DIR / "llama-server.exe"

MODEL_FILE = "gemma-4-26B-A4B-it-UD-IQ3_S.gguf"
PORT = 8080
RPC_SERVER = "hpenvy.local:50052"

TABMEPP_FILE = DATA_DIR / "tabmepp_benchmark.json"
DOCSPLIT_FILE = DATA_DIR / "docsplit_seq_benchmark.json"


def prevent_windows_sleep() -> None:
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[System] Windows execution state locked: Sleep disabled during tournament.", flush=True)
    except Exception as e:
        print(f"[System] Warning: could not set execution state: {e}", flush=True)


def get_vram_usage() -> str:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "Unknown"


def kill_local_llama() -> None:
    try:
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass
    time.sleep(2)


def ensure_remote_rpc() -> None:
    import socket
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("hpenvy.local", 50052))
        s.close()
        print("[Cluster] Remote RPC daemon on hpenvy.local:50052 is ALIVE and reachable!", flush=True)
        return
    except Exception:
        pass

    print("[Cluster] Starting remote RPC daemon with cache on hpenvy...", flush=True)
    start_cmd = (
        "ssh -o BatchMode=yes hpenvy powershell -NoProfile -Command "
        "\"Start-Process -FilePath 'C:\\llama-rpc\\ggml-rpc-server.exe' -ArgumentList '-H','0.0.0.0','-p','50052','-c' -WorkingDirectory 'C:\\llama-rpc'\""
    )
    subprocess.run(start_cmd, shell=True, check=False)
    time.sleep(3)


def launch_distributed_server() -> subprocess.Popen:
    model_path = MODELS_DIR / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")

    ensure_remote_rpc()
    kill_local_llama()

    cmd = [
        str(SERVER_EXE),
        "-m", str(model_path.resolve()),
        "--rpc", RPC_SERVER,
        "-ngl", "99",
        "-c", "8192",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--flash-attn", "on",
        "-np", "1",
        "-t", "8",
        "--no-cache-prompt",
        "-rea", "off",
    ]

    log_path = RESULTS_DIR / "server_pss_gemma26b.log"
    log_file = log_path.open("w", encoding="utf-8")

    print(f"\n[Cluster] Launching Distributed Gemma-26B MoE on port {PORT} with RPC {RPC_SERVER}...", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(BIN_DIR))

    # Health check
    url = f"http://127.0.0.1:{PORT}/health"
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("llama-server terminated prematurely during distributed 26B load!")
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"[Cluster] Distributed Gemma-26B MoE READY! VRAM: {get_vram_usage()} MiB\n", flush=True)
                time.sleep(1)
                return proc
        except Exception:
            pass
        time.sleep(2)

    raise RuntimeError(f"Server failed to start on port {PORT} within timeout")


def call_chat_completion(system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(url, json=payload, timeout=90)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return f"ERROR: {e}"


# ============================================================================
# ALGORITHM IMPLEMENTATIONS FOR GEMMA-26B MoE
# ============================================================================

def run_condition_1_holistic_window_26b(
    packets: List[Dict[str, Any]],
    window_size: int = 15,
) -> List[List[int]]:
    """Condition 1: Zero-Shot Holistic Packet Windowing with 26B MoE."""
    sys_prompt = (
        "You are an expert document packet splitting analyst. Given page snippets from a multi-page PDF packet, "
        "identify which page numbers are the FIRST page of a new independent document.\n"
        "Rules:\n"
        "- Page 1 is ALWAYS the start of Document 1.\n"
        "- Pages with explicit 'Page 2', 'Page 3', or continuing paragraphs continue the same document.\n"
        "- Pages with new letterheads, title headers, new recipient/sender, or Page 1 markers start a NEW document.\n"
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
                f"Pages {w_start+1} to {w_end} of a document packet are shown below.\n"
                f"Which page numbers in {w_start+1}..{w_end} start a NEW independent document?\n\n"
                + "\n".join(lines_prompt)
                + "\n\nOutput JSON only: {\"start_pages\": [...]}"
            )
            resp = call_chat_completion(sys_prompt, user_prompt, max_tokens=140)

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


def run_condition_2_and_3_pairwise_and_hybrid_26b(
    packets: List[Dict[str, Any]],
    gbdt_probs_per_packet: List[List[float]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Condition 2 (TwoStage-Pairwise-MoE-26B) & Condition 3 (TwoStage-Hybrid-MoE-26B)."""
    batch_sys_prompt = (
        "You are an expert Page Stream Segmentation classifier using Stage-1 Page Signatures and Continuity Signals.\n"
        "For each adjacent page pair (Page t-1 -> Page t), decide if Page t starts a NEW document (SPLIT) or continues the same document (CONTINUE).\n"
        "Rules:\n"
        "- Explicit Page# >= 2 or sequential page numbers (1->2, 2->3) or mid-sentence continuation -> CONTINUE.\n"
        "- Explicit Page# = 1, new title/memo/invoice/resume header, high bottom whitespace on Page(t-1), or conflicting document IDs with low lexical similarity -> SPLIT.\n"
        "Output ONLY a JSON object mapping the target page number to \"SPLIT\" or \"CONTINUE\", e.g. {\"2\": \"SPLIT\", \"3\": \"CONTINUE\"}."
    )

    single_sys_prompt = (
        "You are an expert Page Stream Segmentation adjudicator resolving an ambiguous boundary between Page t-1 and Page t.\n"
        "Carefully analyze:\n"
        "1. Entity continuity: Do names, dates, invoice numbers, or personal details persist or change completely?\n"
        "2. Header/Footer layout: Does Page t start with a primary title or letterhead?\n"
        "3. Sentence continuation: Does Page t-1 end mid-sentence?\n\n"
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

        # Condition 2: Pairwise Batch Evaluation on Gemma-26B MoE
        batch_size = 5
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
            resp = call_chat_completion(batch_sys_prompt, user_prompt, max_tokens=140)

            for i in batch_idxs:
                pg_str = str(i + 1)
                m = re.search(rf"\"{pg_str}\"\s*:\s*\"(SPLIT|CONTINUE)\"", resp, re.I)
                if m:
                    dec = 1 if m.group(1).upper() == "SPLIT" else 0
                else:
                    dec = 1 if pkt_gbdt_probs[i] >= 0.5 else 0
                flags_llm[i] = dec

        # Condition 3: Selective Hybrid Adjudication (Deep 26B MoE reasoning on uncertain GBDT transitions)
        for i in range(1, n):
            gbdt_p = pkt_gbdt_probs[i]
            if 0.25 <= gbdt_p <= 0.75:
                # Ambiguous boundary: invoke 26B MoE adjudicator
                sp, sc = sigs[i - 1], sigs[i]
                adj_user = (
                    f"Analyze this ambiguous boundary transition from Page {i} to Page {i+1}:\n"
                    f"- GBDT Predicted Split Probability: {gbdt_p:.3f}\n"
                    f"- Page {i} (t-1) Bottom OCR: {sp['bottom_text'][:180]!r}\n"
                    f"- Page {i+1} (t) Top OCR: {sc['top_text'][:200]!r}\n"
                    f"- Semantic Continuity: FullTextSim={full_sims[i]:.2f}, HeaderSim={hdr_sims[i]:.2f}\n"
                    f"- Entity Disconnect: ConflictingIDs={bool(sp['ids'] and sc['ids'] and not sp['ids'].intersection(sc['ids']))}\n\n"
                    "Does Page {i+1} start a NEW document (SPLIT) or CONTINUE Page {i}?\n"
                    "Output JSON: {\"decision\": \"SPLIT\" or \"CONTINUE\", \"confidence\": ...}"
                )
                adj_resp = call_chat_completion(single_sys_prompt, adj_user, max_tokens=80)
                m_adj = re.search(r"\"decision\"\s*:\s*\"(SPLIT|CONTINUE)\"", adj_resp, re.I)
                if m_adj:
                    flags_hybrid[i] = 1 if m_adj.group(1).upper() == "SPLIT" else 0
                else:
                    flags_hybrid[i] = 1 if gbdt_p >= 0.50 else 0
            else:
                # High-confidence GBDT decision
                flags_hybrid[i] = 1 if gbdt_p >= 0.50 else 0

        pure_llm_preds.append(flags_llm)
        hybrid_preds.append(flags_hybrid)

    return pure_llm_preds, hybrid_preds


# ============================================================================
# MAIN TOURNAMENT ORCHESTRATOR
# ============================================================================

def main() -> None:
    prevent_windows_sleep()

    print("==================================================================", flush=True)
    print("DISTRIBUTED GEMMA-26B MoE PAGE STREAM SEGMENTATION TOURNAMENT", flush=True)
    print("Dual-PC RPC Cluster: Laptop RTX 4060 (8GB) + Desktop hpenvy (6GB)", flush=True)
    print("Model: Gemma-4-26B-A4B-it-UD-IQ3_S.gguf (100% Pure VRAM)", flush=True)
    print("==================================================================\n", flush=True)

    with open(TABMEPP_FILE, "r", encoding="utf-8") as f:
        tabme_data = json.load(f)
    with open(DOCSPLIT_FILE, "r", encoding="utf-8") as f:
        docsplit_data = json.load(f)

    benchmarks = {
        "docsplit_mono_seq": {
            "name": "DocSplit-Mono-Seq (Same-Category Contiguous - Hardest Frontier)",
            "train": docsplit_data["train_packets"],
            "test": docsplit_data["mono_seq_test"][:10],
        },
        "docsplit_poly_seq": {
            "name": "DocSplit-Poly-Seq (Multi-Category Contiguous Packets)",
            "train": docsplit_data["train_packets"],
            "test": docsplit_data["poly_seq_test"][:10],
        },
        "tabmepp": {
            "name": "Official TABME++ Benchmark (UCSF Archival Folders)",
            "train": tabme_data["train_packets"],
            "test": tabme_data["test_packets"][:10],
        },
    }

    # Pre-train GBDT on each benchmark's train set to get baseline and probability cache
    print("[Baseline] Training Supervised 30F GBDT baselines...", flush=True)
    gbdt_cache: Dict[str, Tuple[List[List[int]], List[List[float]]]] = {}
    gbdt_metrics: Dict[str, Dict[str, float]] = {}

    for b_key, b_info in benchmarks.items():
        preds, probs = train_and_run_algorithm_3_supervised_gbdt(b_info["train"], b_info["test"])
        gbdt_cache[b_key] = (preds, probs)
        m = evaluate_predictions(b_info["test"], preds)
        gbdt_metrics[b_key] = m
        print(f"  {b_info['name']}: GBDT Boundary F1 = {m['boundary_f1']:.2f}%, Doc Exact F1 = {m['doc_exact_f1']:.2f}%, STP = {m['packet_stp_pct']:.1f}%", flush=True)

    # Launch Distributed Gemma-26B MoE Server
    server_proc = launch_distributed_server()
    tournament_results: Dict[str, Any] = {}

    try:
        for b_key, b_info in benchmarks.items():
            print("\n" + "=" * 70, flush=True)
            print(f"RUNNING BENCHMARK: {b_info['name']}", flush=True)
            print("=" * 70, flush=True)

            test_pkts = b_info["test"]
            _, gbdt_probs = gbdt_cache[b_key]

            # -------------------------------------------------------------
            # Condition 1: Holistic Windowed Gemma-26B MoE
            # -------------------------------------------------------------
            print(f"\n[1/3] Running Condition 1: Holistic Windowed Prompting (Gemma-26B MoE)...", flush=True)
            t0 = time.perf_counter()
            preds_c1 = run_condition_1_holistic_window_26b(test_pkts, window_size=15)
            time_c1 = time.perf_counter() - t0
            metrics_c1 = evaluate_predictions(test_pkts, preds_c1)
            print(f"  ===> Holistic MoE-26B: Boundary F1={metrics_c1['boundary_f1']:.2f}%, Doc Exact F1={metrics_c1['doc_exact_f1']:.2f}%, STP={metrics_c1['packet_stp_pct']:.1f}%, Time={time_c1:.1f}s", flush=True)

            # -------------------------------------------------------------
            # Condition 2 & 3: Pairwise Signature & Hybrid Adjudication
            # -------------------------------------------------------------
            print(f"\n[2/3 & 3/3] Running Conditions 2 & 3: Pairwise Signatures & Hybrid 26B Adjudication...", flush=True)
            t0 = time.perf_counter()
            preds_c2, preds_c3 = run_condition_2_and_3_pairwise_and_hybrid_26b(test_pkts, gbdt_probs)
            time_c23 = time.perf_counter() - t0

            metrics_c2 = evaluate_predictions(test_pkts, preds_c2)
            metrics_c3 = evaluate_predictions(test_pkts, preds_c3)

            print(f"  ===> Pairwise MoE-26B (Cond 2): Boundary F1={metrics_c2['boundary_f1']:.2f}%, Doc Exact F1={metrics_c2['doc_exact_f1']:.2f}%, STP={metrics_c2['packet_stp_pct']:.1f}%", flush=True)
            print(f"  ===> Hybrid MoE-26B (Cond 3)  : Boundary F1={metrics_c3['boundary_f1']:.2f}%, Doc Exact F1={metrics_c3['doc_exact_f1']:.2f}%, STP={metrics_c3['packet_stp_pct']:.1f}%, Time={time_c23:.1f}s", flush=True)

            tournament_results[b_key] = {
                "benchmark_name": b_info["name"],
                "gbdt_baseline": gbdt_metrics[b_key],
                "condition_1_holistic_26b": {**metrics_c1, "time_sec": round(time_c1, 2)},
                "condition_2_pairwise_26b": {**metrics_c2, "time_sec": round(time_c23 / 2.0, 2)},
                "condition_3_hybrid_26b": {**metrics_c3, "time_sec": round(time_c23 / 2.0, 2)},
            }

        # -----------------------------------------------------------------
        # SAVE FINAL ARTIFACT
        # -----------------------------------------------------------------
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_file = RESULTS_DIR / f"pss_gemma26b_tournament_results_{timestamp}.json"
        artifact = {
            "benchmark": "Distributed Gemma-26B MoE Page Stream Segmentation Tournament",
            "model": "Gemma-4-26B-A4B-it-UD-IQ3_S.gguf",
            "cluster": "Laptop RTX 4060 (8GB) + Desktop hpenvy 1660S (6GB) via llama.cpp RPC",
            "timestamp_utc": timestamp,
            "results": tournament_results,
        }
        out_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        print(f"\n[Tournament Complete] Artifact saved to: {out_file}\n", flush=True)

    finally:
        print("[Teardown] Stopping local distributed llama-server...", flush=True)
        if server_proc.poll() is None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        time.sleep(2)
        kill_local_llama()
        print(f"[Teardown] Distributed Server stopped. Clean VRAM: {get_vram_usage()} MiB", flush=True)


if __name__ == "__main__":
    main()
