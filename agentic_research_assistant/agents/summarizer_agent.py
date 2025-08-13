from __future__ import annotations
from typing import Dict, List
import os
from openai import OpenAI
from config.settings import settings


class SummarizerAgent:
	def __init__(self, model: str | None = None):
		self.model = model or settings.model_name
		self.client = OpenAI(api_key=settings.openai_api_key)

	def summarize(self, root_query: str, tasks: List[Dict], sources_by_task: Dict[str, List[Dict]]) -> Dict:
		sections = []
		for task in tasks:
			q = task["question"]
			sources = sources_by_task.get(q, [])
			
			if not sources:
				sections.append({
					"task_id": task["id"],
					"question": q,
					"content": f"No sources found for: {q}",
					"citations": [],
				})
				continue
				
			citations = []
			context = []
			for i, s in enumerate(sources, start=1):
				title = s.get('title', 'Unknown Source')[:100]
				url = s.get('url', '')
				citations.append(f"[{i}] {title} - {url}")
				content_chunk = s.get("content", "")[:3000]
				if content_chunk:
					context.append(content_chunk)
					
			if not context:
				sections.append({
					"task_id": task["id"],
					"question": q,
					"content": f"Sources found but no readable content for: {q}",
					"citations": citations,
				})
				continue
				
			prompt = (
				f"Research Question: {q}\n\n"
				f"Instructions: {task.get('instructions', 'Summarize findings')}\n\n"
				"Based on the provided sources, write a comprehensive summary addressing this research question. "
				"Use in-text citations [1], [2], etc. that correspond to the source list. "
				"Focus on key facts, findings, and insights. Be objective and cite frequently."
			)
			
			try:
				messages = [
					{"role": "system", "content": "You are an expert research analyst. Synthesize information from multiple sources with proper citations."},
					{"role": "user", "content": prompt},
					{"role": "user", "content": "Source Material:\n" + "\n\n---\n\n".join(context)},
				]
				completion = self.client.chat.completions.create(
					model=self.model, 
					messages=messages,
					temperature=settings.temperature
				)
				content = completion.choices[0].message.content
			except Exception as e:
				content = f"Error generating summary: {e}\n\nKey sources: {', '.join([s.get('title', 'Unknown') for s in sources[:3]])}"
				
			sections.append({
				"task_id": task["id"],
				"question": q,
				"content": content,
				"citations": citations,
			})
			
		full_content = "\n\n".join(s["content"] for s in sections)
		return {"query": root_query, "sections": sections, "content": full_content}
