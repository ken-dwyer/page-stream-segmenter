"""Test NuExtract on Mono-Seq False Negatives and True Negatives.
Evaluates whether NuExtract-tiny (and later NuExtract-v1.5) can rescue the 11 GBDT False Negatives
without causing False Positives on True Continuations.
"""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

MODELS_DIR = ROOT_DIR / "models"
MODEL_FILE = sys.argv[1] if len(sys.argv) > 1 else "nuextract-tiny-q8.gguf"
PORT = 8080
SERVER_BIN = ROOT_DIR / "bin" / "llama-cuda" / "llama-server.exe"

def start_server(model_name: str) -> subprocess.Popen:
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(1)

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    cmd = [
        str(SERVER_BIN),
        "-m", str(model_path),
        "-c", "4096",
        "-ngl", "99",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--reasoning", "off",
        "--reasoning-format", "none",
    ]
    log_file = open(ROOT_DIR / "results" / f"server_test_{Path(model_name).stem}.log", "w", encoding="utf-8")
    print(f"[Server] Starting {model_name} on port {PORT}...", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("Server terminated prematurely!")
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            if r.status_code == 200:
                print(f"[Server] {model_name} is READY in VRAM!", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError("Server failed to start")


def query_nuextract(text: str, schema: Dict[str, Any], max_tokens: int = 150) -> Tuple[Dict[str, Any], float]:
    schema_str = json.dumps(schema, indent=4)
    # Official NuExtract-v1.5 prompt format:
    prompt = f"<|input|>\n### Template:\n{schema_str}\n### Text:\n{text}\n\n<|output|>\n"
    url = f"http://127.0.0.1:{PORT}/completion"
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": max_tokens,
        "stop": ["<|input|>", "<|endoftext|>", "### Text:"],
        "stream": False,
    }
    t0 = time.time()
    res = requests.post(url, json=payload, timeout=20)
    lat = time.time() - t0
    res.raise_for_status()
    raw = res.json().get("content", "").strip()

    # Parse JSON
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(raw)
    except Exception:
        parsed = {"document_title": "", "document_id": ""}
    return parsed, lat


def main():
    proc = start_server(MODEL_FILE)

    from pss_pipeline import DatasetRegistry
    packets, _ = DatasetRegistry.load_dataset("docsplit_mono_seq", split="test")
    pkts = {p["packet_id"]: p for p in packets}

    # 11 False Negatives where GBDT gave P < 0.02
    test_transitions = [
        # (packet_id, from_page, to_page, is_true_split, description)
        ("docsplit_mono_seq_budget_01", 12, 13, True, "Budget: Revenues -> Sources & Uses (P=0.0129)"),
        ("docsplit_mono_seq_handwritten_04", 4, 5, True, "Handwritten: Case Page 4/4 -> Pre-Cert (P=0.0154)"),
        ("docsplit_mono_seq_invoice_05", 12, 13, True, "Invoice: Amount Due -> Estimate 2963 (P=0.0134)"),
        ("docsplit_mono_seq_invoice_05", 15, 16, True, "Invoice: Authorized By -> Broward County (P=0.0105)"),
        ("docsplit_mono_seq_letter_06", 12, 13, True, "Letter: Date Prepared -> Reeves County Attorney (P=0.0013)"),
        ("docsplit_mono_seq_memo_07", 2, 3, True, "Memo: CDC info -> Clearinghouse Memo (P=0.0071)"),
        ("docsplit_mono_seq_memo_07", 18, 19, True, "Memo: -6- -> Senate of Maryland (P=0.0010)"),
        ("docsplit_mono_seq_news article_08", 2, 3, True, "News: Copyright notice -> NYT 1853 (P=0.0085)"),
        ("docsplit_mono_seq_questionnaire_09", 11, 12, True, "Questionnaire: ACE Score -> CBC Questionnaire (P=0.0015)"),
        ("docsplit_mono_seq_specification_12", 10, 11, True, "Spec: Joint Fill Tech Data -> CertainTeed (P=0.0007)"),
        ("docsplit_mono_seq_email_14", 4, 5, True, "Email: SF Water signoff -> Melendez email (P=0.0013)"),
        # True Negatives (Continuations within same document)
        ("docsplit_mono_seq_budget_01", 1, 2, False, "Budget P1->P2 (True Continue)"),
        ("docsplit_mono_seq_budget_01", 2, 3, False, "Budget P2->P3 (True Continue)"),
        ("docsplit_mono_seq_invoice_05", 1, 2, False, "Invoice P1->P2 (True Continue)"),
        ("docsplit_mono_seq_letter_06", 1, 2, False, "Letter P1->P2 (True Continue)"),
        ("docsplit_mono_seq_memo_07", 3, 4, False, "Memo P3->P4 (True Continue)"),
        ("docsplit_mono_seq_specification_12", 1, 2, False, "Spec P1->P2 (True Continue)"),
    ]

    schema = {
        "document_title": "",
        "document_id": "",
    }

    print("\n" + "=" * 80)
    print(f"RUNNING NUEXTRACT EVALUATION WITH {MODEL_FILE}")
    print("=" * 80, flush=True)

    rescued = 0
    total_fn = 0
    fp_count = 0
    total_tn = 0

    results_table = []

    for pkt_id, p_from, p_to, gt_split, desc in test_transitions:
        pkt = pkts.get(pkt_id)
        if not pkt:
            continue
        p1 = pkt["pages"][p_from - 1]
        p2 = pkt["pages"][p_to - 1]

        # NuExtract on first 1200 chars of each page
        t1 = p1.get("full_text", "")[:1200]
        t2 = p2.get("full_text", "")[:1200]

        ext1, lat1 = query_nuextract(t1, schema)
        ext2, lat2 = query_nuextract(t2, schema)

        title1 = (ext1.get("document_title") or "").strip().lower()
        title2 = (ext2.get("document_title") or "").strip().lower()
        id1 = (ext1.get("document_id") or "").strip().lower()
        id2 = (ext2.get("document_id") or "").strip().lower()

        # Discontinuity logic:
        # A split is flagged if page 2 has a title that is non-empty and differs from page 1,
        # OR page 2 has a new document_id that differs from page 1.
        has_new_title = bool(title2 and title2 != title1)
        has_new_id = bool(id2 and id2 != id1 and len(id2) >= 3)
        discontinuity = has_new_title or has_new_id

        if gt_split:
            total_fn += 1
            if discontinuity:
                rescued += 1
                status = "RESCUED (TP)"
            else:
                status = "MISSED (FN)"
        else:
            total_tn += 1
            if discontinuity:
                fp_count += 1
                status = "FALSE ALARM (FP)"
            else:
                status = "CORRECT CONTINUE (TN)"

        results_table.append({
            "desc": desc,
            "gt": "SPLIT" if gt_split else "CONT",
            "pred": "SPLIT" if discontinuity else "CONT",
            "status": status,
            "title1": title1[:30],
            "title2": title2[:30],
            "id1": id1[:20],
            "id2": id2[:20],
            "lat": round(lat1 + lat2, 2),
        })

        print(f"[{status:^18}] {desc}")
        print(f"   P{p_from} Title: {title1!r:30} | ID: {id1!r}")
        print(f"   P{p_to} Title: {title2!r:30} | ID: {id2!r}")
        print(f"   Discontinuity: {discontinuity} (latency: {lat1+lat2:.2f}s)\n", flush=True)

    print("=" * 80)
    print(f"SUMMARY FOR {MODEL_FILE}:")
    print(f"  False Negatives Rescued: {rescued}/{total_fn} ({rescued/total_fn*100:.1f}%)")
    print(f"  True Continuations Kept: {total_tn - fp_count}/{total_tn} (FP Rate: {fp_count}/{total_tn})")
    print("=" * 80)

    try:
        proc.terminate()
    except Exception:
        pass

if __name__ == "__main__":
    main()
