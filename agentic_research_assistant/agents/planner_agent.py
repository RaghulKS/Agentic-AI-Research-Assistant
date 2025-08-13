from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import os
from openai import OpenAI
from config.settings import settings


class PlannerAgent:
	def __init__(self, model: str | None = None):
		self.model = model or settings.model_name
		self.client = OpenAI(api_key=settings.openai_api_key)
		self.prompt_path = Path("prompts") / "planner_prompt.md"

	def plan(self, query: str) -> Dict:
		if self.prompt_path.exists():
			template = self.prompt_path.read_text(encoding="utf-8")
		else:
			template = (
				"You are a senior research strategist. Given the user query: {query} break it into sub-questions. "
				"Return JSON with a 'tasks' array of objects with id, question, instructions."
			)
		prompt = template.replace("{query}", query)

		try:
			completion = self.client.chat.completions.create(
				model=self.model,
				messages=[
					{"role": "system", "content": "You are an expert research planner. Break complex queries into focused, searchable sub-questions that build toward comprehensive understanding."},
					{"role": "user", "content": prompt},
				],
				response_format={"type": "json_object"},
			)
			content = completion.choices[0].message.content
			plan = json.loads(content)
			
			if not plan.get("tasks"):
				raise ValueError("No tasks generated")
				
		except Exception as e:
			keywords = query.split()[:3]
			plan = {"tasks": [
				{"id": "T1", "question": f"What is {query}?", "instructions": "Find definitions and basic information"},
				{"id": "T2", "question": f"Key aspects of {' '.join(keywords)}", "instructions": "Identify main components and characteristics"},
				{"id": "T3", "question": f"Latest developments in {' '.join(keywords)}", "instructions": "Find recent news and updates"}
			]}
			
		return plan
