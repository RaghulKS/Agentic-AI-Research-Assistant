from __future__ import annotations
from typing import Dict, List

from agents.plagiarism_checker_agent import PlagiarismCheckerAgent


class PlagiarismTool:
	def __init__(self):
		self.agent = PlagiarismCheckerAgent()

	def check(self, summary: Dict, sources_by_task: Dict[str, List[Dict]]) -> Dict:
		return self.agent.check(summary, sources_by_task)
