"""Prepare Page Stream Segmentation (PSS) & DocSplit-Seq Benchmarks.

Builds two complementary benchmark suites in `data/`:
1. `data/tabmepp_benchmark.json`:
   - Official TABME++ (`bevaya/TABMEpp`) multi-document folders with Microsoft Azure OCR
     and 2D normalized bounding box layout features (`lines_data`).
   - 25 official test folders (~530 pages, ~235 documents) + 120 disjoint training folders
     (zero document overlap with test folders).
2. `data/docsplit_seq_benchmark.json`:
   - Built from `jordyvl/rvl_cdip_n_mp` (RVL-CDIP-N Multi-Page PDFs across 16 document classes)
     following the AWS DocSplit (`arXiv:2602.15958`) contiguous packet protocols:
     * `DocSplit-Poly-Seq`: 15 multi-category contiguous document packets.
     * `DocSplit-Mono-Seq`: 15 single-category contiguous document packets (same-class documents back-to-back).
     * Disjoint training packets for supervised baseline models.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import random
import re
import tarfile
from typing import Any, Dict, List, Tuple

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from pypdf import PdfReader

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_tabmepp_ocr(ocr_json_str: str) -> Dict[str, Any]:
    """Extract reading-order text and spatial 2D layout statistics from TABME++ Azure OCR JSON."""
    if not ocr_json_str:
        return {
            "full_text": "",
            "top_lines": [],
            "bottom_lines": [],
            "num_lines": 0,
            "min_y1": 1.0,
            "max_y3": 0.0,
            "bottom_whitespace": 1.0,
            "top_whitespace": 1.0,
            "first_line_height": 0.0,
            "max_top_line_height": 0.0,
            "centered_top_lines": 0,
            "corner_numbers": [],
            "avg_confidence": 0.0,
        }
    try:
        obj = json.loads(ocr_json_str)
    except Exception:
        return {
            "full_text": "",
            "top_lines": [],
            "bottom_lines": [],
            "num_lines": 0,
            "min_y1": 1.0,
            "max_y3": 0.0,
            "bottom_whitespace": 1.0,
            "top_whitespace": 1.0,
            "first_line_height": 0.0,
            "max_top_line_height": 0.0,
            "centered_top_lines": 0,
            "corner_numbers": [],
            "avg_confidence": 0.0,
        }

    lines = obj.get("lines_data", [])
    if not lines:
        return {
            "full_text": "",
            "top_lines": [],
            "bottom_lines": [],
            "num_lines": 0,
            "min_y1": 1.0,
            "max_y3": 0.0,
            "bottom_whitespace": 1.0,
            "top_whitespace": 1.0,
            "first_line_height": 0.0,
            "max_top_line_height": 0.0,
            "centered_top_lines": 0,
            "corner_numbers": [],
            "avg_confidence": 0.0,
        }

    # Sort lines by vertical band (2% page height buckets) then horizontal X1
    sorted_lines = sorted(lines, key=lambda l: (round(float(l.get("Y1", 0.0)) * 50), float(l.get("X1", 0.0))))

    text_lines = []
    top_lines = []
    bottom_lines = []
    corner_numbers = []
    min_y1 = 1.0
    max_y3 = 0.0
    max_top_h = 0.0
    centered_top = 0
    confs = []

    for idx, l in enumerate(sorted_lines):
        w = str(l.get("Word", "")).strip()
        if not w:
            continue
        x1 = float(l.get("X1", 0.0))
        y1 = float(l.get("Y1", 0.0))
        x2 = float(l.get("X2", x1))
        y3 = float(l.get("Y3", y1))
        h = max(0.0, y3 - y1)
        conf = float(l.get("Confidence", 0.9))
        confs.append(conf)

        min_y1 = min(min_y1, y1)
        max_y3 = max(max_y3, y3)
        text_lines.append(w)

        if y1 < 0.22 or len(top_lines) < 6:
            top_lines.append({"text": w, "x1": round(x1, 3), "y1": round(y1, 3), "h": round(h, 4)})
            max_top_h = max(max_top_h, h)
            mid_x = 0.5 * (x1 + x2)
            if 0.28 <= mid_x <= 0.72 and (x2 - x1) < 0.75:
                centered_top += 1

        if y3 > 0.80:
            bottom_lines.append({"text": w, "x1": round(x1, 3), "y1": round(y1, 3), "h": round(h, 4)})

        # Check top/bottom margin for standalone page numbers or "Page X"
        if (y1 < 0.12 or y3 > 0.86) and len(w) <= 18:
            if re.search(r"\b(page|pg\.?|p\.?)\s*\d+|\b\d+\s*(of|/)\s*\d+|^-?\s*\d{1,3}\s*-?$", w, re.I):
                corner_numbers.append({"text": w, "y1": round(y1, 3), "x1": round(x1, 3)})

    if not bottom_lines and text_lines:
        for l in sorted_lines[-4:]:
            w = str(l.get("Word", "")).strip()
            if w:
                bottom_lines.append({
                    "text": w,
                    "x1": round(float(l.get("X1", 0.0)), 3),
                    "y1": round(float(l.get("Y1", 0.0)), 3),
                    "h": round(max(0.0, float(l.get("Y3", 0.0)) - float(l.get("Y1", 0.0))), 4),
                })

    first_h = top_lines[0]["h"] if top_lines else 0.0
    return {
        "full_text": "\n".join(text_lines),
        "top_lines": top_lines[:8],
        "bottom_lines": bottom_lines[-6:],
        "num_lines": len(text_lines),
        "min_y1": round(min_y1 if text_lines else 1.0, 4),
        "max_y3": round(max_y3 if text_lines else 0.0, 4),
        "bottom_whitespace": round(max(0.0, 1.0 - max_y3) if text_lines else 1.0, 4),
        "top_whitespace": round(min_y1 if text_lines else 1.0, 4),
        "first_line_height": round(first_h, 4),
        "max_top_line_height": round(max_top_h, 4),
        "centered_top_lines": centered_top,
        "corner_numbers": corner_numbers,
        "avg_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
    }


def build_tabmepp_benchmark() -> None:
    print("=== 1. Building Official TABME++ Benchmark ===")
    p0 = hf_hub_download(repo_id="bevaya/TABMEpp", filename="data/test-00000-of-00002.parquet", repo_type="dataset")
    p1 = hf_hub_download(repo_id="bevaya/TABMEpp", filename="data/test-00001-of-00002.parquet", repo_type="dataset")
    p_folders = hf_hub_download(repo_id="bevaya/TABMEpp", filename="streams/test_folders.txt", repo_type="dataset")

    t0 = pq.read_table(p0, columns=["doc_id", "pg_id", "ocr"])
    t1 = pq.read_table(p1, columns=["doc_id", "pg_id", "ocr"])

    raw_docs: Dict[str, Dict[int, str]] = {}
    for table in (t0, t1):
        for d, p, o in zip(
            table["doc_id"].to_pylist(),
            table["pg_id"].to_pylist(),
            table["ocr"].to_pylist(),
        ):
            raw_docs.setdefault(d, {})[int(p)] = o

    folders: Dict[int, List[str]] = {}
    with open(p_folders, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                doc_id, fid = parts[0], int(parts[1])
                folders.setdefault(fid, []).append(doc_id)

    # Select first 25 folders as official test evaluation suite
    test_fids = list(range(1, 26))
    test_doc_ids = set()
    for fid in test_fids:
        test_doc_ids.update(folders[fid])

    # Select up to 120 training folders that have ZERO document overlap with test_doc_ids
    train_fids = []
    for fid in sorted(folders.keys()):
        if fid in test_fids:
            continue
        dids = folders[fid]
        if any(d in test_doc_ids for d in dids):
            continue
        train_fids.append(fid)
        if len(train_fids) >= 120:
            break

    needed_docs = set(test_doc_ids)
    for fid in train_fids:
        needed_docs.update(folders[fid])

    print(f"Parsing OCR & spatial features for {len(needed_docs)} unique TABME++ documents...")
    parsed_docs: Dict[str, List[Dict[str, Any]]] = {}
    for doc_id in needed_docs:
        pg_map = raw_docs[doc_id]
        pages = []
        for pg_id in sorted(pg_map.keys()):
            feat = parse_tabmepp_ocr(pg_map[pg_id])
            feat["doc_id"] = doc_id
            feat["pg_id"] = pg_id
            pages.append(feat)
        parsed_docs[doc_id] = pages

    def assemble_folder(fid: int) -> Dict[str, Any]:
        dids = folders[fid]
        stream_pages = []
        gt_spans = []
        cursor = 1
        for doc_idx, did in enumerate(dids):
            doc_pgs = parsed_docs[did]
            start_pg = cursor
            for local_idx, pg in enumerate(doc_pgs):
                stream_pages.append({
                    "packet_page_num": cursor,
                    "doc_id": did,
                    "doc_type": "business_document",
                    "local_page_num": local_idx + 1,
                    "is_first_page": 1 if local_idx == 0 else 0,
                    **pg,
                })
                cursor += 1
            end_pg = cursor - 1
            gt_spans.append({
                "local_doc_id": f"{did}_{doc_idx+1}",
                "doc_id": did,
                "doc_type": "business_document",
                "start_page": start_pg,
                "end_page": end_pg,
                "page_ordinals": list(range(start_pg, end_pg + 1)),
            })
        return {
            "packet_id": f"tabmepp_folder_{fid:04d}",
            "folder_id": fid,
            "total_pages": len(stream_pages),
            "num_documents": len(gt_spans),
            "subdocuments": gt_spans,
            "pages": stream_pages,
        }

    test_packets = [assemble_folder(fid) for fid in test_fids]
    train_packets = [assemble_folder(fid) for fid in train_fids]

    total_test_pages = sum(p["total_pages"] for p in test_packets)
    total_test_docs = sum(p["num_documents"] for p in test_packets)
    print(f"TABME++ Test Suite: {len(test_packets)} folders, {total_test_docs} docs, {total_test_pages} pages")
    print(f"TABME++ Train Suite: {len(train_packets)} folders, {sum(p['total_pages'] for p in train_packets)} pages")

    out_path = DATA_DIR / "tabmepp_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"test_packets": test_packets, "train_packets": train_packets}, f)
    print(f"Saved {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


def parse_pdf_page_text(text: str) -> Dict[str, Any]:
    """Create structured layout & text summary from a native PDF page's extracted text."""
    raw_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    top_lines = [{"text": ln, "x1": 0.1, "y1": round(0.03 * (i + 1), 3), "h": 0.02} for i, ln in enumerate(raw_lines[:8])]
    bottom_lines = [
        {"text": ln, "x1": 0.1, "y1": round(0.85 + 0.02 * i, 3), "h": 0.02}
        for i, ln in enumerate(raw_lines[-6:])
    ]
    corner_numbers = []
    for i, ln in enumerate(raw_lines[:3] + raw_lines[-3:]):
        if len(ln) <= 22 and re.search(r"\b(page|pg\.?|p\.?)\s*\d+|\b\d+\s*(of|/)\s*\d+|^-?\s*\d{1,3}\s*-?$", ln, re.I):
            corner_numbers.append({"text": ln, "y1": 0.05 if i < 3 else 0.92, "x1": 0.5})

    # Estimate bottom whitespace based on line count (typical page has ~45 lines)
    est_fill = min(0.95, len(raw_lines) / 45.0)
    return {
        "full_text": "\n".join(raw_lines),
        "top_lines": top_lines,
        "bottom_lines": bottom_lines,
        "num_lines": len(raw_lines),
        "min_y1": 0.05 if raw_lines else 1.0,
        "max_y3": round(est_fill, 4),
        "bottom_whitespace": round(1.0 - est_fill, 4),
        "top_whitespace": 0.05 if raw_lines else 1.0,
        "first_line_height": 0.025 if raw_lines else 0.0,
        "max_top_line_height": 0.025 if raw_lines else 0.0,
        "centered_top_lines": 1 if raw_lines and len(raw_lines[0]) < 55 else 0,
        "corner_numbers": corner_numbers,
        "avg_confidence": 0.99,
    }


def build_docsplit_seq_benchmark() -> None:
    print("\n=== 2. Building DocSplit-Seq (Mono-Seq & Poly-Seq) from RVL-CDIP-N-MP ===")
    tar_path = hf_hub_download(repo_id="jordyvl/rvl_cdip_n_mp", filename="data.tar.gz", repo_type="dataset")
    print(f"Downloaded rvl_cdip_n_mp archive: {tar_path}")

    docs_by_cat: Dict[str, List[Dict[str, Any]]] = {}
    with tarfile.open(tar_path, mode="r:gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.lower().endswith(".pdf"):
                continue
            parts = Path(member.name).parts
            if len(parts) < 2:
                continue
            category = parts[-2].lower()
            doc_name = Path(parts[-1]).stem
            try:
                pdf_bytes = tf.extractfile(member).read()
                reader = PdfReader(io.BytesIO(pdf_bytes))
                num_pgs = len(reader.pages)
                if num_pgs < 1 or num_pgs > 15:
                    continue
                pages_data = []
                non_empty = 0
                for idx in range(num_pgs):
                    t = (reader.pages[idx].extract_text() or "").strip()
                    if len(t) >= 40:
                        non_empty += 1
                    pg_feat = parse_pdf_page_text(t)
                    pg_feat["doc_id"] = doc_name
                    pg_feat["pg_id"] = idx
                    pages_data.append(pg_feat)
                # Keep PDFs where all pages (or >=85% of pages and page 0) have digital text
                if non_empty == num_pgs or (num_pgs >= 2 and non_empty / num_pgs >= 0.85 and len(pages_data[0]["full_text"]) >= 50):
                    docs_by_cat.setdefault(category, []).append({
                        "doc_id": doc_name,
                        "doc_type": category,
                        "num_pages": num_pgs,
                        "pages": pages_data,
                    })
            except Exception:
                continue

    print("Extracted digital-text documents by category:")
    for cat, dlist in sorted(docs_by_cat.items()):
        mp_count = sum(1 for d in dlist if d["num_pages"] >= 2)
        print(f"  {cat:24s}: {len(dlist):3d} docs ({mp_count:3d} multi-page)")

    rng = random.Random(42)
    test_pool: Dict[str, List[Dict[str, Any]]] = {}
    train_pool: Dict[str, List[Dict[str, Any]]] = {}
    for cat, dlist in docs_by_cat.items():
        shuffled = list(dlist)
        rng.shuffle(shuffled)
        # Prefer multi-page docs in both pools
        mp = [d for d in shuffled if d["num_pages"] >= 2]
        sp = [d for d in shuffled if d["num_pages"] == 1]
        ordered = mp + sp
        split_idx = max(1, len(ordered) // 2)
        test_pool[cat] = ordered[:split_idx]
        train_pool[cat] = ordered[split_idx:] if len(ordered) > 1 else ordered

    def create_packet(doc_list: List[Dict[str, Any]], packet_id: str, mode: str) -> Dict[str, Any]:
        stream_pages = []
        gt_spans = []
        cursor = 1
        for doc_idx, d in enumerate(doc_list):
            start_pg = cursor
            for local_idx, pg in enumerate(d["pages"]):
                stream_pages.append({
                    "packet_page_num": cursor,
                    "doc_id": d["doc_id"],
                    "doc_type": d["doc_type"],
                    "local_page_num": local_idx + 1,
                    "is_first_page": 1 if local_idx == 0 else 0,
                    **pg,
                })
                cursor += 1
            end_pg = cursor - 1
            gt_spans.append({
                "local_doc_id": f"{d['doc_type']}-{doc_idx+1:02d}",
                "doc_id": d["doc_id"],
                "doc_type": d["doc_type"],
                "start_page": start_pg,
                "end_page": end_pg,
                "page_ordinals": list(range(start_pg, end_pg + 1)),
            })
        return {
            "packet_id": packet_id,
            "mode": mode,
            "total_pages": len(stream_pages),
            "num_documents": len(gt_spans),
            "subdocuments": gt_spans,
            "pages": stream_pages,
        }

    # Build 15 DocSplit-Poly-Seq test packets (5-7 docs per packet, diverse categories, mostly multi-page)
    all_test_docs = [d for cat_docs in test_pool.values() for d in cat_docs]
    mp_test_docs = [d for d in all_test_docs if d["num_pages"] >= 2]
    sp_test_docs = [d for d in all_test_docs if d["num_pages"] == 1]

    poly_seq_test = []
    for i in range(15):
        chosen = rng.sample(mp_test_docs, k=min(4, len(mp_test_docs))) + rng.sample(sp_test_docs, k=min(2, len(sp_test_docs)))
        rng.shuffle(chosen)
        poly_seq_test.append(create_packet(chosen, f"docsplit_poly_seq_test_{i+1:02d}", "poly_seq"))

    # Build 15 DocSplit-Mono-Seq test packets (single category per packet, 4-5 docs of the SAME category back-to-back!)
    mono_eligible_cats = [cat for cat, dlist in test_pool.items() if len(dlist) >= 4 and sum(1 for d in dlist if d["num_pages"] >= 2) >= 2]
    if not mono_eligible_cats:
        mono_eligible_cats = [cat for cat, dlist in docs_by_cat.items() if len(dlist) >= 4]
    print("Mono-Seq eligible categories:", mono_eligible_cats)

    mono_seq_test = []
    for i in range(15):
        cat = mono_eligible_cats[i % len(mono_eligible_cats)]
        pool = test_pool[cat] if len(test_pool[cat]) >= 4 else docs_by_cat[cat]
        k = min(5, len(pool))
        chosen = rng.sample(pool, k=k)
        mono_seq_test.append(create_packet(chosen, f"docsplit_mono_seq_{cat}_{i+1:02d}", "mono_seq"))

    # Build 60 training packets from train_pool (30 poly_seq + 30 mono_seq)
    all_train_docs = [d for cat_docs in train_pool.values() for d in cat_docs]
    train_packets = []
    for i in range(30):
        chosen = rng.sample(all_train_docs, k=min(5, len(all_train_docs)))
        train_packets.append(create_packet(chosen, f"docsplit_train_poly_{i+1:02d}", "poly_seq"))
    train_mono_cats = [cat for cat, dlist in train_pool.items() if len(dlist) >= 3]
    for i in range(30):
        cat = train_mono_cats[i % len(train_mono_cats)]
        chosen = rng.sample(train_pool[cat], k=min(4, len(train_pool[cat])))
        train_packets.append(create_packet(chosen, f"docsplit_train_mono_{cat}_{i+1:02d}", "mono_seq"))

    print(
        f"DocSplit-Poly-Seq Test: {len(poly_seq_test)} packets, "
        f"{sum(p['num_documents'] for p in poly_seq_test)} docs, "
        f"{sum(p['total_pages'] for p in poly_seq_test)} pages"
    )
    print(
        f"DocSplit-Mono-Seq Test: {len(mono_seq_test)} packets, "
        f"{sum(p['num_documents'] for p in mono_seq_test)} docs, "
        f"{sum(p['total_pages'] for p in mono_seq_test)} pages"
    )

    out_path = DATA_DIR / "docsplit_seq_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "poly_seq_test": poly_seq_test,
                "mono_seq_test": mono_seq_test,
                "train_packets": train_packets,
            },
            f,
        )
    print(f"Saved {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    build_tabmepp_benchmark()
    build_docsplit_seq_benchmark()
