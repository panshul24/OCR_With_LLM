from __future__ import annotations

import os
import time
import json
from pathlib import Path
from typing import Optional

from app.ocr import extract_text_from_bytes
from app.llm import segment_and_categorize
from app.fuzzy import load_keywords, compute_keyword_scores

INBOX = Path(os.getenv("INBOX_DIR", "/data/inbox"))
OUTBOX = Path(os.getenv("OUTBOX_DIR", "/data/outbox"))
SUPPORTED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

SCAN_INTERVAL_SECONDS = float(os.getenv("WATCH_INTERVAL", "2.0"))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/data/config"))
KEYWORDS_FILE = CONFIG_DIR / "keywords.json"


def process_file(file_path: Path) -> Optional[Path]:
	try:
		data = file_path.read_bytes()
		text, meta = extract_text_from_bytes(data, filename=file_path.name)
		segments = segment_and_categorize(text)
		keywords = load_keywords(KEYWORDS_FILE)
		OUTBOX.mkdir(parents=True, exist_ok=True)
		if isinstance(segments, list):
			written = None
			for idx, seg in enumerate(segments, start=1):
				document_type = (seg or {}).get("document_type") or "uncategorized"
				folder = OUTBOX / document_type
				folder.mkdir(parents=True, exist_ok=True)
				# Prefer segment text for fuzzy scoring when available
				seg_text = (seg or {}).get("text_span") or text
				out_payload = {
					"filename": file_path.name,
					"segment_index": idx,
					"ocr_meta": meta,
					"segment": seg,
					"fuzzy_scores": compute_keyword_scores(seg_text, keywords),
				}
				out_path = folder / f"{file_path.stem}.segment{idx}.json"
				out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2))
				written = out_path
			return written
		else:
			out_path = OUTBOX / f"{file_path.stem}.json"
			out_payload = {
				"filename": file_path.name,
				"ocr_meta": meta,
				"text": text,
				"categories": segments,
				"fuzzy_scores": compute_keyword_scores(text, keywords),
			}
			out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2))
			return out_path
	except Exception as e:
		try:
			OUTBOX.mkdir(parents=True, exist_ok=True)
			(OUTBOX / f"{file_path.stem}.error.txt").write_text(str(e))
		except Exception:
			pass
		return None


def run_loop() -> None:
	INBOX.mkdir(parents=True, exist_ok=True)
	print(f"[watcher] Watching {INBOX} -> {OUTBOX}")
	processed: set[str] = set()
	while True:
		for p in INBOX.iterdir():
			if not p.is_file():
				continue
			if p.suffix.lower() not in SUPPORTED_EXT:
				continue
			key = f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}"
			if key in processed:
				continue
			out = process_file(p)
			if out is not None:
				processed.add(key)
				print(f"[watcher] Processed {p.name} -> {out.name}")
		time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
	run_loop()
