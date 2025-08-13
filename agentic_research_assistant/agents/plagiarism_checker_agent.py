from __future__ import annotations
from typing import Dict, List
import os
import hashlib

try:
	from copyleaks.copyleaks import Copyleaks
	from copyleaks.models.export import ExportResults
	COPYLEAKS_AVAILABLE = True
except Exception:
	COPYLEAKS_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlagiarismCheckerAgent:
	def __init__(self):
		self.use_copyleaks = COPYLEAKS_AVAILABLE and bool(os.getenv("COPYLEAKS_API_KEY"))

	def _semantic_score(self, text: str, sources: List[str]) -> float:
		if not text or not sources:
			return 1.0
		vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
		matrix = vectorizer.fit_transform([text] + sources)
		sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
		max_sim = float(sims.max()) if sims.size > 0 else 0.0
		return 1.0 - max_sim

	def check(self, summary: Dict, sources_by_task: Dict[str, List[Dict]]) -> Dict:
		full_text = summary.get("content", "")
		all_source_texts: List[str] = []
		for _, srcs in sources_by_task.items():
			for s in srcs:
				if s.get("content"):
					all_source_texts.append(s["content"][:10000])

		originality = self._semantic_score(full_text, all_source_texts)
		flags: List[Dict] = []
		if originality < 0.85:
			flags.append({"span": full_text[:1500], "reason": "High similarity to sources", "score": originality})

		return {
			"originality_score": originality,
			"flags": flags,
		}
