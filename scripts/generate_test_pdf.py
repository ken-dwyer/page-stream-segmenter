"""Generate a realistic 12-page multi-document PDF packet for testing the CLI.

Stitches 3 distinct documents:
  Doc 1: 3-page Invoice Packet (Invoice #INV-2024-8841)
  Doc 2: 4-page Technical Memorandum (Confidential Memo)
  Doc 3: 5-page Curriculum Vitae / Resume
"""

import io
import tarfile
from pathlib import Path
from huggingface_hub import hf_hub_download
from pypdf import PdfReader, PdfWriter

ROOT_DIR = Path(__file__).resolve().parent
TEST_DIR = ROOT_DIR / "test_data"
TEST_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_PACKET_PDF = TEST_DIR / "sample_multi_doc_packet.pdf"


def main():
    tar_path = hf_hub_download(repo_id="jordyvl/rvl_cdip_n_mp", filename="data.tar.gz", repo_type="dataset")
    print(f"Reading sample PDFs from {tar_path}...")

    # Select 3 documents: 1 invoice, 1 memo, 1 resume
    selected = {}
    with tarfile.open(tar_path, mode="r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".pdf"):
                continue
            cat = Path(m.name).parts[-2].lower()
            if cat in ("invoice", "memo", "resume") and cat not in selected:
                b = tf.extractfile(m).read()
                r = PdfReader(io.BytesIO(b))
                if 2 <= len(r.pages) <= 6:
                    selected[cat] = (m.name, r)
                    print(f"  Found {cat:8s}: {Path(m.name).name} ({len(r.pages)} pages)")
            if len(selected) == 3:
                break

    writer = PdfWriter()
    meta = []
    cursor = 1
    for cat in ("invoice", "memo", "resume"):
        name, r = selected[cat]
        start = cursor
        for p in r.pages:
            writer.add_page(p)
            cursor += 1
        end = cursor - 1
        meta.append({"cat": cat, "start": start, "end": end, "pages": end - start + 1})

    with open(SAMPLE_PACKET_PDF, "wb") as f:
        writer.write(f)

    print(f"\nGenerated test packet: {SAMPLE_PACKET_PDF} ({len(writer.pages)} total pages)")
    for m in meta:
        print(f"  Ground Truth Doc: {m['cat']:8s} -> Pages {m['start']} to {m['end']} ({m['pages']} pages)")


if __name__ == "__main__":
    main()
