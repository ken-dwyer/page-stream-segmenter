"""NuExtract Local Structured Extraction & PSS Boundary Discontinuity Benchmark
Evaluates NuExtract-tiny (0.5B Q8 GGUF) on RTX 4060:
1. Downstream KIE: Schema-guided zero-shot structured information extraction on discrete documents.
2. Upstream PSS Stage 1.5: Header entity discontinuity tracking to resolve same-category Mono-Seq splits.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR = ROOT_DIR / "models"
MODEL_FILE = "nuextract-tiny-q8.gguf"
RESULTS_DIR = ROOT_DIR / "results"
SERVER_BIN = ROOT_DIR / "bin" / "llama-cuda" / "llama-server.exe"
PORT = 8080

def ensure_server() -> subprocess.Popen:
    """Spins up local llama-server with NuExtract-tiny."""
    # Check if already running and healthy
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
        if r.status_code == 200:
            print("[Server] Local server already alive and responsive.", flush=True)
            return None
    except Exception:
        pass

    # Kill any stale instance
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    time.sleep(1)

    model_path = MODELS_DIR / MODEL_FILE
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    cmd = [
        str(SERVER_BIN),
        "-m", str(model_path),
        "-c", "8192",
        "-ngl", "99",
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--reasoning", "off",
        "--reasoning-format", "none",
    ]

    log_path = RESULTS_DIR / "server_nuextract.log"
    log_file = log_path.open("w", encoding="utf-8")
    print(f"[Server] Launching NuExtract-tiny on port {PORT}...", flush=True)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server terminated prematurely! Check {log_path}")
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            if r.status_code == 200:
                print("[Server] NuExtract-tiny is READY in VRAM!", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(1)

    raise TimeoutError("NuExtract server failed to start within 60s")


def query_nuextract(document_text: str, schema_dict: Dict[str, Any], max_tokens: int = 500) -> Tuple[Dict[str, Any], float, float]:
    """Sends document and JSON schema to NuExtract and returns parsed JSON, latency, and tokens/sec."""
    schema_str = json.dumps(schema_dict, indent=2)
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nWe need to extract information from the following document into this JSON format:\n"
        f"{schema_str}\n\nDocument:\n{document_text}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    url = f"http://127.0.0.1:{PORT}/completion"
    payload = {
        "prompt": prompt,
        "temperature": 0.0,
        "n_predict": max_tokens,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "stream": False,
    }

    t0 = time.time()
    res = requests.post(url, json=payload, timeout=30)
    latency = time.time() - t0
    res.raise_for_status()
    data = res.json()

    content = data.get("content", "").strip()
    timings = data.get("timings", {})
    tps = timings.get("predicted_per_second", 0.0)

    # Clean markdown json blocks if present
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(content)
    except Exception:
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {"raw_output": content, "error": "json_parse_error"}

    return parsed, latency, tps


def get_vram_mb() -> float:
    """Gets current GPU memory usage via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, text=True, check=True
        )
        return float(res.stdout.strip())
    except Exception:
        return 0.0


def run_experiment():
    print("=" * 70)
    print("NUEXTRACT-TINY LOCAL DOCUMENT EXTRACTION & PSS BENCHMARK")
    print("=" * 70, flush=True)

    server_proc = ensure_server()
    time.sleep(2)
    vram_used = get_vram_mb()
    print(f"[VRAM] GPU Memory Active: {vram_used:.1f} MiB\n", flush=True)

    # Load dataset packets for testing
    from pss_pipeline import DatasetRegistry
    mono_packets, _ = DatasetRegistry.load_dataset("docsplit_mono_seq", split="test")
    poly_packets, _ = DatasetRegistry.load_dataset("docsplit_poly_seq", split="test")

    packet_dict = {p["packet_id"]: p for p in (mono_packets + poly_packets)}

    results = {
        "model": "NuExtract-tiny (Qwen2-0.5B, Q8_0 GGUF)",
        "vram_mb": vram_used,
        "task1_downstream_kie": [],
        "task2_upstream_pss_discontinuity": [],
    }

    # =========================================================================
    # TASK 1: DOWNSTREAM KEY-VALUE INFORMATION EXTRACTION (KIE)
    # =========================================================================
    print("--- [TASK 1] DOWNSTREAM SCHEMA-GUIDED EXTRACTION (KIE) ---", flush=True)

    kie_test_cases = [
        {
            "name": "Invoice Key-Value Extraction",
            "packet_id": "docsplit_mono_seq_invoice_05",
            "page_idx": 0,  # Page 1
            "schema": {
                "invoice_number": "",
                "vendor_or_company": "",
                "billing_date": "",
                "due_date": "",
                "total_amount": "",
                "recipient_or_client": "",
            },
        },
        {
            "name": "Inmate Medication & Medical Treatment Form",
            "packet_id": "docsplit_mono_seq_form_15",
            "page_idx": 4,  # Page 5 (Start of Form 2)
            "schema": {
                "form_title": "",
                "institution_or_facility": "",
                "inmate_or_patient_name": "",
                "id_number": "",
                "medical_complaint_or_diagnosis": "",
                "date": "",
            },
        },
        {
            "name": "Curriculum Vitae / Resume Extraction",
            "packet_id": "docsplit_mono_seq_resume_10",
            "page_idx": 0,  # Page 1
            "schema": {
                "candidate_name": "",
                "current_job_title": "",
                "education_institution": "",
                "degree": "",
                "skills_or_expertise": [],
            },
        },
        {
            "name": "Corporate Memo / Executive Correspondence",
            "packet_id": "docsplit_mono_seq_memo_07",
            "page_idx": 0,  # Page 1
            "schema": {
                "sender": "",
                "recipient": "",
                "date": "",
                "subject": "",
                "action_items_or_summary": "",
            },
        },
        {
            "name": "Scientific / Research Article",
            "packet_id": "docsplit_mono_seq_scientific publication_11",
            "page_idx": 0,  # Page 1
            "schema": {
                "paper_title": "",
                "authors": [],
                "institution_or_journal": "",
                "publication_date": "",
                "abstract_summary": "",
            },
        },
    ]

    for tc in kie_test_cases:
        pkt = packet_dict.get(tc["packet_id"])
        if not pkt or tc["page_idx"] >= len(pkt["pages"]):
            print(f"Skipping {tc['name']}: packet {tc['packet_id']} not found", flush=True)
            continue

        doc_text = pkt["pages"][tc["page_idx"]]["full_text"][:2500]  # First 2500 chars
        extracted, latency, tps = query_nuextract(doc_text, tc["schema"])

        # Check verbatim substring grounding
        verbatim_count = 0
        total_fields = 0
        doc_lower = doc_text.lower()
        for k, v in extracted.items():
            if k in tc["schema"] and v:
                total_fields += 1
                val_str = str(v).lower()
                # Check if first 10 chars of value exist verbatim in text
                if val_str[:15] in doc_lower or any(part in doc_lower for part in val_str.split() if len(part) > 3):
                    verbatim_count += 1

        grounding_pct = (verbatim_count / total_fields * 100.0) if total_fields > 0 else 0.0
        is_valid_json = "error" not in extracted and len(extracted) >= len(tc["schema"])

        rec = {
            "test_case": tc["name"],
            "packet_id": tc["packet_id"],
            "page_num": tc["page_idx"] + 1,
            "latency_sec": round(latency, 3),
            "tokens_per_sec": round(tps, 1),
            "is_valid_json": is_valid_json,
            "grounding_pct": round(grounding_pct, 1),
            "schema": tc["schema"],
            "extracted": extracted,
        }
        results["task1_downstream_kie"].append(rec)
        print(f"  -> {tc['name']}: {latency:.2f}s | {tps:.1f} tok/s | Valid: {is_valid_json} | Grounding: {grounding_pct:.1f}%")
        print(f"     Extracted: {json.dumps(extracted, ensure_ascii=False)[:120]}...\n", flush=True)

    # =========================================================================
    # TASK 2: UPSTREAM STAGE 1.5 ENTITY DISCONTINUITY FOR PSS
    # =========================================================================
    print("--- [TASK 2] STAGE 1.5 ENTITY DISCONTINUITY (PSS BOUNDARY RESCUE) ---", flush=True)

    header_schema = {
        "document_title": "",
        "document_id_or_number": "",
        "organization_or_author": "",
        "date": "",
    }

    pss_test_transitions = [
        {
            "name": "Inmate Medication Form (False Negative Rescue)",
            "packet_id": "docsplit_mono_seq_form_15",
            "prev_page_idx": 3,  # Page 4 (End of Form 1)
            "curr_page_idx": 4,  # Page 5 (Start of Form 2 - True Split!)
            "ground_truth_split": True,
            "gbdt_prob": 0.0203,  # GBDT failed here by assigning < 0.02!
        },
        {
            "name": "Invoice Packet Boundary (Different Invoice Split)",
            "packet_id": "docsplit_mono_seq_invoice_05",
            "prev_page_idx": 2,  # Page 3 (End of Invoice 1)
            "curr_page_idx": 3,  # Page 4 (Start of Invoice 2 - True Split!)
            "ground_truth_split": True,
            "gbdt_prob": 0.1245,
        },
        {
            "name": "Same Multi-Page Document Continuation (True Negative)",
            "packet_id": "docsplit_mono_seq_budget_01",
            "prev_page_idx": 0,  # Page 1 (Budget Doc 1)
            "curr_page_idx": 1,  # Page 2 (Budget Doc 1 Continuation - False Split Risk)
            "ground_truth_split": False,
            "gbdt_prob": 0.0150,
        },
        {
            "name": "Resume to Resume Transition (Mono-Seq Split)",
            "packet_id": "docsplit_mono_seq_resume_10",
            "prev_page_idx": 1,  # Page 2 (End of Resume 1)
            "curr_page_idx": 2,  # Page 3 (Start of Resume 2 - True Split!)
            "ground_truth_split": True,
            "gbdt_prob": 0.0812,
        },
    ]

    for pt in pss_test_transitions:
        pkt = packet_dict.get(pt["packet_id"])
        if not pkt:
            continue

        prev_text = pkt["pages"][pt["prev_page_idx"]]["full_text"][:1200]
        curr_text = pkt["pages"][pt["curr_page_idx"]]["full_text"][:1200]

        prev_meta, lat1, _ = query_nuextract(prev_text, header_schema, max_tokens=150)
        curr_meta, lat2, _ = query_nuextract(curr_text, header_schema, max_tokens=150)

        # Entity Discontinuity Rule:
        # A split is flagged if:
        # 1. document_id_or_number differs and both non-empty, OR
        # 2. document_title is detected on curr page and is new/different, OR
        # 3. organization_or_author is detected and different
        id_differs = bool(prev_meta.get("document_id_or_number") and curr_meta.get("document_id_or_number") and 
                          prev_meta["document_id_or_number"].strip().lower() != curr_meta["document_id_or_number"].strip().lower())
        title_appears = bool(curr_meta.get("document_title") and 
                             curr_meta["document_title"].strip().lower() != prev_meta.get("document_title", "").strip().lower())
        author_differs = bool(prev_meta.get("organization_or_author") and curr_meta.get("organization_or_author") and 
                              prev_meta["organization_or_author"].strip().lower() != curr_meta["organization_or_author"].strip().lower())

        nuextract_detected_split = id_differs or title_appears or author_differs
        is_correct = (nuextract_detected_split == pt["ground_truth_split"])

        rec = {
            "transition_name": pt["name"],
            "packet_id": pt["packet_id"],
            "pages": f"P{pt['prev_page_idx']+1} -> P{pt['curr_page_idx']+1}",
            "ground_truth_split": pt["ground_truth_split"],
            "gbdt_prob": pt["gbdt_prob"],
            "prev_entities": prev_meta,
            "curr_entities": curr_meta,
            "nuextract_detected_split": nuextract_detected_split,
            "is_correct": is_correct,
            "latency_total_sec": round(lat1 + lat2, 3),
        }
        results["task2_upstream_pss_discontinuity"].append(rec)
        print(f"  -> {pt['name']} ({rec['pages']}):")
        print(f"     Ground Truth: {'SPLIT' if pt['ground_truth_split'] else 'CONTINUE'} | GBDT P: {pt['gbdt_prob']:.4f}")
        print(f"     Prev: {prev_meta}")
        print(f"     Curr: {curr_meta}")
        print(f"     NuExtract Verdict: {'SPLIT' if nuextract_detected_split else 'CONTINUE'} | Correct: {'YES' if is_correct else 'NO'}\n", flush=True)

    # Save structured results
    out_json = RESULTS_DIR / "nuextract_experiment_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Results] Saved raw benchmark metrics to {out_json}", flush=True)

    # Clean server shutdown
    if server_proc:
        print("[Server] Stopping NuExtract-tiny server...", flush=True)
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except Exception:
            server_proc.kill()

    return results

if __name__ == "__main__":
    run_experiment()
