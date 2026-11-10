import io
import json
import sys
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from pss_pipeline import DatasetRegistry, extract_page_signature

packets, _ = DatasetRegistry.load_dataset("docsplit_mono_seq", split="test")
pkts = {p["packet_id"]: p for p in packets}

targets = [
    ("docsplit_mono_seq_budget_01", 12, 13),
    ("docsplit_mono_seq_handwritten_04", 4, 5),
    ("docsplit_mono_seq_invoice_05", 12, 13),
    ("docsplit_mono_seq_invoice_05", 15, 16),
    ("docsplit_mono_seq_letter_06", 12, 13),
    ("docsplit_mono_seq_memo_07", 2, 3),
    ("docsplit_mono_seq_memo_07", 18, 19),
    ("docsplit_mono_seq_news article_08", 2, 3),
    ("docsplit_mono_seq_questionnaire_09", 11, 12),
    ("docsplit_mono_seq_specification_12", 10, 11),
    ("docsplit_mono_seq_email_14", 4, 5),
]

print("Evaluating 11 GBDT False Negatives:")
caught = 0
for pkt_id, p_prev, p_curr in targets:
    pkt = pkts[pkt_id]
    sp = extract_page_signature(pkt["pages"][p_prev - 1])
    sc = extract_page_signature(pkt["pages"][p_curr - 1])
    
    has_closing_prev = sp["has_closing_trigger"]
    has_title_curr = sc["has_title_trigger"]
    first_line = sc["full_text"].strip().splitlines()[0].strip() if sc["full_text"].strip().splitlines() else ""
    starts_one = first_line in ["1", "- 1 -", "1.", "Page 1", "PAGE 1"] or first_line.startswith("1 ")
    has_top_ws = sc["top_whitespace"] > 0.08
    has_centered = sc["centered_top_lines"] > 0
    has_letterhead = any(w in sc["top_text"].lower() for w in ["attorney", "county", "board of", "commissioners", "from:", "published:"])
    
    triggered = (has_closing_prev or has_title_curr or starts_one or has_centered or has_letterhead)
    if triggered:
        caught += 1
    print(f"[{'TRIGGERED' if triggered else 'MISSED':^10}] {pkt_id} P{p_prev}->P{p_curr}: closing={has_closing_prev}, title={has_title_curr}, one={starts_one}, centered={has_centered}, letterhead={has_letterhead}")

print(f"\nCaught {caught}/{len(targets)} False Negatives!")
