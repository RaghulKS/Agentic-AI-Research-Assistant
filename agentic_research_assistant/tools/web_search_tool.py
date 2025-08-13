from __future__ import annotations
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from .pdf_reader_tool import PDFReaderTool
from config.settings import settings


class WebSearchTool:
	def __init__(self, timeout_seconds: int = None):
		self.timeout_seconds = timeout_seconds or settings.timeout_seconds
		self.pdf_reader = PDFReaderTool()
		self.headers = {"User-Agent": settings.user_agent}

	def search(self, query: str, max_results: int = 5) -> List[Dict]:
		results: List[Dict] = []
		with DDGS() as ddgs:
			for r in ddgs.text(query, max_results=max_results):
				results.append({
					"title": r.get("title"),
					"url": r.get("href") or r.get("url"),
					"snippet": r.get("body") or r.get("snippet"),
				})
		return results

	def fetch_and_extract(self, url: str) -> Dict:
		try:
			resp = requests.get(url, headers=self.headers, timeout=self.timeout_seconds)
			resp.raise_for_status()
			
			content_type = resp.headers.get("Content-Type", "").lower()
			if ("application/pdf" in content_type) or url.lower().endswith(".pdf"):
				text = self.pdf_reader.read_from_bytes(resp.content)
				return {"type": "pdf", "text": text}
			else:
				soup = BeautifulSoup(resp.text, "html.parser")
				
				for s in soup(["script", "style", "header", "footer", "nav", "form", "aside", "menu"]):
					s.extract()
				
				main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=["content", "main", "article"])
				if main_content:
					text = " ".join(main_content.get_text(" ").split())
				else:
					text = " ".join(soup.get_text(" ").split())
				
				text = text[:settings.max_content_length]
				return {"type": "html", "text": text}
				
		except Exception as e:
			return {"type": "error", "error": str(e), "text": ""}
