from __future__ import annotations
from typing import Dict
import os
from openai import OpenAI
from config.settings import settings


class RewriterAgent:
	def __init__(self, model: str | None = None):
		self.model = model or settings.model_name
		self.client = OpenAI(api_key=settings.openai_api_key)

	def rewrite(self, summary: Dict, plagiarism_report: Dict) -> Dict:
		text = summary.get("content", "")
		flags = plagiarism_report.get("flags", [])
		changed = 0
		
		if not flags:
			return {"content": text, "changed_segments": 0}
			
		for flag in flags:
			span = flag.get("span", "")
			if not span or len(span) < 50:
				continue
				
			try:
				prompt = (
					"Rewrite the following text to improve originality while preserving all factual content and citations. "
					"Restructure sentences, use different vocabulary, and vary sentence patterns. "
					"Maintain the same meaning, tone, and all citation numbers [1], [2], etc.\n\n"
					f"Original text:\n{span}\n\n"
					"Rewritten version:"
				)
				
				comp = self.client.chat.completions.create(
					model=self.model,
					messages=[
						{"role": "system", "content": "You are an expert academic editor specializing in rewriting for originality while maintaining accuracy."},
						{"role": "user", "content": prompt},
					],
					temperature=settings.temperature * 2  # Higher creativity for rewriting
				)
				rewrite_text = comp.choices[0].message.content.strip()
				
				if span in text and rewrite_text and len(rewrite_text) > 20:
					text = text.replace(span, rewrite_text)
					changed += 1
					
			except Exception as e:
				print(f"Failed to rewrite segment: {e}")
				continue
				
		return {"content": text, "changed_segments": changed}
