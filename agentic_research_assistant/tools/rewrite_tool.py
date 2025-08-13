from __future__ import annotations
from typing import Dict

from agents.rewriter_agent import RewriterAgent


class RewriteTool:
	def __init__(self):
		self.agent = RewriterAgent()

	def rewrite(self, summary: Dict, plagiarism_report: Dict) -> Dict:
		return self.agent.rewrite(summary, plagiarism_report)
