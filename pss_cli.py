"""Page Stream Segmentation (PSS) & Document Packet Splitter CLI.

Split contiguous multi-document PDFs (invoices, tax packets, medical records,
legal files, loan packets) into individual logically bounded PDF files.

Examples:
  # 1. Fast dry-run inspection (0 GPU VRAM, <0.3s):
  python pss_cli.py packet.pdf --dry-run

  # 2. Split PDF into subdocuments in output folder:
  python pss_cli.py packet.pdf -o ./extracted_docs/

  # 3. Hybrid AI mode (GBDT + local LLM adjudication on ambiguous boundaries):
  python pss_cli.py packet.pdf --mode hybrid --model gemma-4-e4b

  # 4. Export DocSplit JSON metadata:
  python pss_cli.py packet.pdf --json packet_split.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pypdf import PdfReader, PdfWriter

# Ensure local imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from run_pss_benchmark_tournament import (
    extract_page_signature,
    compute_packet_tfidf_similarities,
    extract_pairwise_feature_vector,
    boundaries_to_spans,
    LOCAL_MODELS,
    LlamaServerManager,
)

MODEL_PKL = ROOT_DIR / "models" / "pss_classifier.pkl"


def extract_pages_from_pdf(pdf_path: Path) -> Tuple[List[Dict[str, Any]], PdfReader]:
    """Extract page text and build structured page representations from a native PDF."""
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError(f"PDF {pdf_path} has 0 pages.")

    pages_data = []
    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

        top_lines = [
            {"text": ln, "x1": 0.1, "y1": round(0.03 * (i + 1), 3), "h": 0.02}
            for i, ln in enumerate(lines[:8])
        ]
        bottom_lines = [
            {"text": ln, "x1": 0.1, "y1": round(0.85 + 0.02 * i, 3), "h": 0.02}
            for i, ln in enumerate(lines[-6:])
        ]
        corner_numbers = []
        for i, ln in enumerate(lines[:3] + lines[-3:]):
            if len(ln) <= 22 and re.search(
                r"\b(page|pg\.?|p\.?)\s*\d+|\b\d+\s*(of|/)\s*\d+|^-?\s*\d{1,3}\s*-?$",
                ln,
                re.I,
            ):
                corner_numbers.append({"text": ln, "y1": 0.05 if i < 3 else 0.92, "x1": 0.5})

        est_fill = min(0.95, len(lines) / 45.0) if lines else 0.0
        pages_data.append({
            "packet_page_num": idx + 1,
            "full_text": "\n".join(lines),
            "top_lines": top_lines,
            "bottom_lines": bottom_lines,
            "num_lines": len(lines),
            "min_y1": 0.05 if lines else 1.0,
            "max_y3": round(est_fill, 4),
            "bottom_whitespace": round(1.0 - est_fill, 4),
            "top_whitespace": 0.05 if lines else 1.0,
            "first_line_height": 0.025 if lines else 0.0,
            "max_top_line_height": 0.025 if lines else 0.0,
            "centered_top_lines": 1 if lines and len(lines[0]) < 55 else 0,
            "corner_numbers": corner_numbers,
            "avg_confidence": 0.99,
        })

    return pages_data, reader


class PssSegmenter:
    """Production Page Stream Segmentation Engine supporting Fast, Heuristic, and Hybrid AI modes."""

    def __init__(self, model_pkl: Path = MODEL_PKL):
        self.model_pkl = model_pkl
        self.classifier_artifact = None
        if model_pkl.exists():
            with open(model_pkl, "rb") as f:
                self.classifier_artifact = pickle.load(f)

    def segment_fast(
        self, pages_data: List[Dict[str, Any]], threshold: float = 0.50
    ) -> List[Dict[str, Any]]:
        """Fast mode: Deterministic rules + 30-feature spatial GBDT/LR classifier (<0.1s)."""
        n = len(pages_data)
        if n == 0:
            return []

        sigs = [extract_page_signature(p) for p in pages_data]
        full_sims, hdr_sims = compute_packet_tfidf_similarities(pages_data)

        gbdt = self.classifier_artifact["gbdt"] if self.classifier_artifact else None
        lr = self.classifier_artifact["lr"] if self.classifier_artifact else None
        scaler = self.classifier_artifact["scaler"] if self.classifier_artifact else None

        transitions = []
        run_len = 1

        for i in range(1, n):
            sp, sc = sigs[i - 1], sigs[i]
            n_prev, n_curr = sp["exp_page_num"], sc["exp_page_num"]

            # Deterministic fast-paths
            if n_prev is not None and n_curr is not None and n_curr == n_prev + 1:
                transitions.append({
                    "from_page": i,
                    "to_page": i + 1,
                    "split": False,
                    "prob": 0.01,
                    "reason": f"Sequential page number ({n_prev} -> {n_curr})",
                    "mode": "deterministic_rule",
                })
                run_len += 1
                continue

            if n_curr is not None and n_curr >= 2 and not sc["has_page_one_marker"]:
                transitions.append({
                    "from_page": i,
                    "to_page": i + 1,
                    "split": False,
                    "prob": 0.05,
                    "reason": f"Explicit continuation page number ({n_curr})",
                    "mode": "deterministic_rule",
                })
                run_len += 1
                continue

            if sc["has_page_one_marker"] and not sc["has_cont_marker"]:
                transitions.append({
                    "from_page": i,
                    "to_page": i + 1,
                    "split": True,
                    "prob": 0.99,
                    "reason": "Explicit Page 1 / Title marker detected",
                    "mode": "deterministic_rule",
                })
                run_len = 1
                continue

            # Statistical Model Inference
            if gbdt is not None and lr is not None and scaler is not None:
                feat = np.array(
                    [extract_pairwise_feature_vector(sp, sc, full_sims[i], hdr_sims[i], run_len)],
                    dtype=np.float32,
                )
                p_gbdt = float(gbdt.predict_proba(feat)[0, 1])
                p_lr = float(lr.predict_proba(scaler.transform(feat))[0, 1])
                prob = 0.75 * p_gbdt + 0.25 * p_lr
                is_split = prob >= threshold

                reason_tags = []
                if prob >= 0.70:
                    reason_tags.append("High spatial/layout discontinuity")
                elif prob <= 0.30:
                    reason_tags.append("Strong semantic/visual continuation")
                else:
                    reason_tags.append("Subtle layout boundary")

                if sp["ids"] and sc["ids"] and not sp["ids"].intersection(sc["ids"]):
                    reason_tags.append("Conflicting entity IDs")
                if sp["bottom_whitespace"] > 0.35:
                    reason_tags.append(f"Preceding page whitespace ({sp['bottom_whitespace']*100:.0f}%)")

                transitions.append({
                    "from_page": i,
                    "to_page": i + 1,
                    "split": is_split,
                    "prob": round(prob, 4),
                    "reason": " | ".join(reason_tags) if reason_tags else "Model scoring",
                    "mode": "30f_gbdt_model",
                })
                run_len = 1 if is_split else run_len + 1
            else:
                # Heuristic fallback if model pkl is missing
                is_split = full_sims[i] < 0.18 or sc["has_title_trigger"]
                transitions.append({
                    "from_page": i,
                    "to_page": i + 1,
                    "split": is_split,
                    "prob": 0.85 if is_split else 0.15,
                    "reason": "Heuristic fallback (classifier not loaded)",
                    "mode": "heuristic",
                })
                run_len = 1 if is_split else run_len + 1

        return transitions

    def segment_hybrid(
        self,
        pages_data: List[Dict[str, Any]],
        llm_model_key: str = "gemma-4-e4b",
        threshold: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """Hybrid mode: Uses GBDT for clear cases; routes ambiguous transitions (0.25 <= p <= 0.75) to local LLM."""
        fast_transitions = self.segment_fast(pages_data, threshold=threshold)
        ambiguous = [
            t for t in fast_transitions if 0.25 <= t["prob"] <= 0.75 and t["mode"] != "deterministic_rule"
        ]

        if not ambiguous:
            return fast_transitions

        model_cfg = next((m for m in LOCAL_MODELS if m["id"] == llm_model_key), LOCAL_MODELS[2])
        print(f"\n[Hybrid Mode] Resolving {len(ambiguous)} ambiguous boundary transitions via {model_cfg['name']}...")

        server = LlamaServerManager(model_cfg)
        server.start()
        try:
            sigs = [extract_page_signature(p) for p in pages_data]
            full_sims, hdr_sims = compute_packet_tfidf_similarities(pages_data)

            for t in ambiguous:
                idx = t["to_page"] - 1  # 0-indexed current page
                sp, sc = sigs[idx - 1], sigs[idx]

                shared_ids = sorted(list(sp["ids"].intersection(sc["ids"])))[:5]
                prev_only = sorted(list(sp["ids"] - sc["ids"]))[:5]
                curr_only = sorted(list(sc["ids"] - sp["ids"]))[:5]

                sys_prompt = (
                    "You are an expert Page Stream Segmentation adjudicator resolving an ambiguous document boundary (Page t-1 -> Page t).\n"
                    "Compare Page(t-1)'s footer/closing and entity IDs with Page(t)'s header/title, pagination, and lexical similarity.\n"
                    "Reply ONLY with a JSON object: {\"decision\": \"SPLIT\" or \"CONTINUE\", \"confidence\": 0.5 to 1.0, \"reason\": \"short explanation\"}"
                )
                user_prompt = (
                    f"Transition Page {idx} -> Page {idx+1} (Prior Split Prob = {t['prob']:.2f}):\n"
                    f"Page {idx} (t-1): Lines={sp['num_lines']}, BotWhitespace={sp['bottom_whitespace']:.2f}, "
                    f"Closing={sp['has_closing_trigger']}, EndsMidSent={sp['ends_mid_sentence']}\n"
                    f"  Top: {sp['top_text'][:160]!r}\n  Bottom: {sp['bottom_text'][:160]!r}\n"
                    f"Page {idx+1} (t): Lines={sc['num_lines']}, TopWhitespace={sc['top_whitespace']:.2f}, "
                    f"TitleTrigger={sc['has_title_trigger']}, StartsMidSent={sc['starts_mid_sentence']}, UpperRatio={sc['upper_ratio']:.2f}\n"
                    f"  Top: {sc['top_text'][:200]!r}\n"
                    f"Continuity: LexicalSim={full_sims[idx]:.3f}, HeaderSim={hdr_sims[idx]:.3f}, "
                    f"SharedIDs={shared_ids}, PrevOnlyIDs={prev_only}, CurrOnlyIDs={curr_only}\n"
                    f"Output JSON:"
                )
                resp = server.chat(sys_prompt, user_prompt, max_tokens=90)
                resp_u = resp.upper()
                llm_split = bool(
                    "\"SPLIT\"" in resp_u
                    or (resp_u.find("SPLIT") != -1 and resp_u.find("CONTINUE") == -1)
                )

                m_conf = re.search(r"\"confidence\"\s*:\s*([0-9.]+)", resp)
                conf = float(m_conf.group(1)) if m_conf else 0.70
                conf = max(0.5, min(0.95, conf))
                llm_prob = conf if llm_split else (1.0 - conf)

                # Fuse GBDT prior (0.60) + LLM adjudicator (0.40)
                fused_prob = 0.60 * t["prob"] + 0.40 * llm_prob
                t["prob"] = round(fused_prob, 4)
                t["split"] = fused_prob >= threshold

                m_reason = re.search(r"\"reason\"\s*:\s*\"([^\"]+)\"", resp)
                llm_exp = m_reason.group(1) if m_reason else ("LLM: SPLIT" if llm_split else "LLM: CONTINUE")
                t["reason"] = f"Hybrid Adjudication ({model_cfg['name']}): {llm_exp}"
                t["mode"] = "hybrid_llm_adjudicated"
        finally:
            server.stop()

        return fast_transitions


def build_document_spans(
    pages_data: List[Dict[str, Any]], transitions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert page transitions into logical subdocument specifications."""
    n = len(pages_data)
    y_flags = [1] + [1 if t["split"] else 0 for t in transitions]
    raw_spans = boundaries_to_spans(y_flags)

    subdocs = []
    for doc_idx, (start, end) in enumerate(raw_spans, 1):
        doc_pages = pages_data[start - 1 : end]
        p1 = doc_pages[0]
        sig1 = extract_page_signature(p1)

        # Detect plausible document category / title
        lines = [ln.strip() for ln in p1["full_text"].splitlines() if ln.strip()]
        title_candidate = lines[0] if lines else "Document"
        for ln in lines[:4]:
            if len(ln) >= 5 and ln.isupper():
                title_candidate = ln
                break

        # Collect unique entity IDs
        all_ids = set()
        for p in doc_pages:
            sig = extract_page_signature(p)
            all_ids.update(sig["ids"])

        subdocs.append({
            "doc_index": doc_idx,
            "title_summary": title_candidate[:60],
            "start_page": start,
            "end_page": end,
            "page_count": end - start + 1,
            "page_ordinals": list(range(start, end + 1)),
            "detected_identifiers": sorted(list(all_ids))[:6],
        })

    return subdocs


def slice_pdf(
    reader: PdfReader,
    subdocs: List[Dict[str, Any]],
    output_dir: Path,
    base_stem: str,
) -> List[Path]:
    """Slice physical PDF into subdocument files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for doc in subdocs:
        writer = PdfWriter()
        for pg_num in doc["page_ordinals"]:
            writer.add_page(reader.pages[pg_num - 1])

        clean_title = re.sub(r"[^A-Za-z0-9_-]+", "_", doc["title_summary"]).strip("_")
        clean_title = clean_title[:30] if clean_title else "doc"
        filename = (
            f"{base_stem}_doc{doc['doc_index']:02d}_p{doc['start_page']}-p{doc['end_page']}_{clean_title}.pdf"
        )
        out_file = output_dir / filename
        with open(out_file, "wb") as f:
            writer.write(f)
        created_files.append(out_file)

    return created_files


def print_cli_summary(
    pdf_path: Path,
    subdocs: List[Dict[str, Any]],
    transitions: List[Dict[str, Any]],
    elapsed: float,
    output_files: Optional[List[Path]] = None,
) -> None:
    """Print an elegant, human-readable terminal summary of the segmentation run."""
    total_pages = len(transitions) + 1
    print("\n" + "=" * 78)
    print(f"📄 PAGE STREAM SEGMENTATION REPORT: {pdf_path.name}")
    print(f"   Total Pages: {total_pages} | Detected Documents: {len(subdocs)} | Processing Time: {elapsed:.2f}s")
    print("=" * 78)

    print("\n📑 DETECTED LOGICAL DOCUMENTS:")
    print(f" {'#':2s} | {'Pages':10s} | {'Len':4s} | {'Key Identifiers / Codes':28s} | {'Title / Header Snippet':24s}")
    print("-" * 78)
    for d in subdocs:
        pages_str = f"p. {d['start_page']} - {d['end_page']}"
        ids_str = ", ".join(d["detected_identifiers"])[:27] or "None detected"
        title_str = d["title_summary"][:24]
        print(f" {d['doc_index']:2d} | {pages_str:10s} | {d['page_count']:3d}p | {ids_str:28s} | {title_str}")

    print("\n🔍 BOUNDARY TRANSITIONS:")
    print(f" {'Boundary':14s} | {'Decision':10s} | {'Prob':6s} | {'Reason / Signal'}")
    print("-" * 78)
    for t in transitions:
        b_str = f"p.{t['from_page']} -> p.{t['to_page']}"
        dec_str = "✂️  SPLIT" if t["split"] else "   CONT"
        prob_str = f"{t['prob']*100:4.1f}%"
        print(f" {b_str:14s} | {dec_str:10s} | {prob_str:6s} | {t['reason']}")

    if output_files:
        print(f"\n💾 CREATED {len(output_files)} EXTRACTED PDF FILES IN:")
        print(f"   {output_files[0].parent}")
        for f in output_files:
            size_kb = f.stat().st_size / 1024
            print(f"   ✓ {f.name} ({size_kb:.1f} KB)")
    print("=" * 78 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Page Stream Segmentation (PSS) & Multi-Document PDF Splitter CLI."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to input PDF file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save split PDF files (default: <pdf_stem>_split/)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["fast", "hybrid"],
        default="fast",
        help="Inference mode: 'fast' (CPU GBDT, <0.2s) or 'hybrid' (GBDT + local LLM adjudication)",
    )
    parser.add_argument(
        "--model",
        choices=[m["id"] for m in LOCAL_MODELS],
        default="gemma-4-e4b",
        help="Local LLM model to use for ambiguous transitions in hybrid mode",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.50,
        help="Probability threshold for document split (default: 0.50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze and print segmentation report without creating PDF files",
    )
    parser.add_argument(
        "--json",
        type=Path,
        nargs="?",
        const=Path("-"),
        default=None,
        help="Export structured DocSplit JSON output to file (or stdout if no path)",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Do not write split PDF files to disk (analysis only)",
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"Error: PDF file {args.pdf_path} not found.", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    pages_data, reader = extract_pages_from_pdf(args.pdf_path)

    segmenter = PssSegmenter()
    if args.mode == "hybrid":
        transitions = segmenter.segment_hybrid(
            pages_data, llm_model_key=args.model, threshold=args.threshold
        )
    else:
        transitions = segmenter.segment_fast(pages_data, threshold=args.threshold)

    subdocs = build_document_spans(pages_data, transitions)
    elapsed = time.perf_counter() - t0

    output_files = None
    if not args.dry_run and not args.no_split:
        out_dir = args.output_dir or args.pdf_path.parent / f"{args.pdf_path.stem}_split"
        output_files = slice_pdf(reader, subdocs, out_dir, args.pdf_path.stem)

    print_cli_summary(args.pdf_path, subdocs, transitions, elapsed, output_files)

    if args.json:
        json_payload = {
            "source_pdf": str(args.pdf_path),
            "total_pages": len(pages_data),
            "detected_documents_count": len(subdocs),
            "mode": args.mode,
            "processing_seconds": round(elapsed, 3),
            "subdocuments": subdocs,
            "transitions": transitions,
        }
        if str(args.json) == "-":
            print(json.dumps(json_payload, indent=2))
        else:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2)
            print(f"Saved JSON metadata to {args.json}")


if __name__ == "__main__":
    main()
