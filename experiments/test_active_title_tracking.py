import io
import json
import subprocess
import sys
import time
from pathlib import Path
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pss_pipeline import DatasetRegistry, parse_nuextract_json

pkts, _ = DatasetRegistry.load_dataset("docsplit_mono_seq", split="test")
p = pkts[0]

server_bin = ROOT_DIR / "bin" / "llama-cuda" / "llama-server.exe"
model_path = ROOT_DIR / "models" / "NuExtract-v1.5-Q4_K_M.gguf"

cmd = [
    str(server_bin),
    "-m", str(model_path),
    "-c", "4096",
    "-ngl", "99",
    "--port", "8081",
    "--host", "127.0.0.1",
    "--reasoning", "off",
    "--reasoning-format", "none",
]

proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

schema = {"document_title": "", "document_id": ""}
schema_str = json.dumps(schema, indent=4)
stop_tokens = ["<|input|>", "<|endoftext|>", "### Text:"]

# Ground truth starts for budget_01: [1, 5, 7, 11, 13]
gt_starts = {d["start_page"] for d in p["subdocuments"]}
print(f"Ground Truth Document Starts: {sorted(list(gt_starts))}")

extractions = []
for idx in range(len(p["pages"])):
    txt = p["pages"][idx]["full_text"][:1200].replace("\xa0", " ").replace("\u2010", "-")
    prompt = f"<|input|>\n### Template:\n{schema_str}\n### Text:\n{txt}\n\n<|output|>\n"
    r = requests.post("http://127.0.0.1:8081/completion", json={"prompt": prompt, "n_predict": 100, "temperature": 0.0})
    parsed = parse_nuextract_json(r.json().get("content", ""))
    t = (parsed.get("document_title") or "").strip().lower()
    i = (parsed.get("document_id") or "").strip().lower()
    extractions.append((t, i))
    print(f"Page {idx+1:2d}: Title={t!r:45} | ID={i!r}")

proc.terminate()

# Now simulate active title tracking
print("\n--- SIMULATING ACTIVE TITLE TRACKING ---")
active_title = extractions[0][0]
active_id = extractions[0][1]

pred_splits = {1}
for page_num in range(2, len(p["pages"]) + 1):
    t_curr, id_curr = extractions[page_num - 1]
    
    # Is it a new document?
    is_new_title = bool(t_curr and active_title and t_curr != active_title)
    is_new_id = bool(id_curr and active_id and id_curr != active_id and len(id_curr) >= 3 and not id_curr.isdigit())
    
    # If currently active title was empty, and curr page has a title
    first_title = bool(t_curr and not active_title)
    
    # Split condition
    if is_new_title or is_new_id:
        pred_splits.add(page_num)
        active_title = t_curr
        active_id = id_curr
        print(f"Page {page_num:2d}: SPLIT -> New Title: {t_curr!r} (was {active_title!r})")
    else:
        if t_curr and not active_title:
            active_title = t_curr
        if id_curr and not active_id:
            active_id = id_curr

print("\nResults:")
print(f"Ground Truth: {sorted(list(gt_starts))}")
print(f"Predicted:    {sorted(list(pred_splits))}")
tp = len(gt_starts.intersection(pred_splits) - {1})
fp = len(pred_splits - gt_starts - {1})
fn = len(gt_starts - pred_splits - {1})
print(f"TP: {tp} | FP: {fp} | FN: {fn}")
