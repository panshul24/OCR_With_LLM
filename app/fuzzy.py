from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any
from rapidfuzz import fuzz, process

DEFAULT_KEYWORDS: Dict[str, List[str]] = {
	"license": ["license", "licence", "registration", "state medical board"],
	"form": ["form", "application", "questionnaire", "disclosure"],
	"invoice": ["invoice", "bill", "statement", "amount due"],
	"id": ["id", "identification", "passport", "driver"],
}


def load_keywords(config_path: Path) -> Dict[str, List[str]]:
	if config_path.exists():
		try:
			return json.loads(config_path.read_text())
		except Exception:
			return DEFAULT_KEYWORDS
	return DEFAULT_KEYWORDS


def compute_keyword_scores(text: str, keywords: Dict[str, List[str]]) -> Dict[str, float]:
	# Compute fuzzy partial ratio per category using the best matching keyword
	result: Dict[str, float] = {}
	for category, words in keywords.items():
		best = 0.0
		for w in words:
			# Use partial_ratio to handle substrings; scale 0..100
			score = fuzz.partial_ratio(text, w)
			if score > best:
				best = score
		result[category] = float(best)
	return result
