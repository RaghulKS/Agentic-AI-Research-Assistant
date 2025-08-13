from __future__ import annotations
import time
from typing import Dict, List

from tools.web_search_tool import WebSearchTool
from config.settings import settings


class RetrieverAgent:
	def __init__(self):
		self.search_tool = WebSearchTool()

	def retrieve(self, sub_question: str, max_results: int = 5) -> List[Dict]:
		try:
			results = self.search_tool.search(sub_question, max_results=max_results)
		except Exception as e:
			print(f"Search failed for '{sub_question}': {e}")
			return []
			
		enriched: List[Dict] = []
		for idx, r in enumerate(results, start=1):
			url = r.get("url")
			if not url:
				continue
				
			try:
				fetched = self.search_tool.fetch_and_extract(url)
				content = fetched.get("text", "")
				if len(content) > 50:
					item = {
						"id": f"S{idx}",
						"title": r.get("title"),
						"url": url,
						"snippet": r.get("snippet"),
						"content": content[:settings.max_content_length],
						"type": fetched.get("type", "unknown"),
						"fetched_at": time.time(),
					}
					enriched.append(item)
			except Exception as e:
				print(f"Failed to fetch {url}: {e}")
				continue
				
		return enriched
