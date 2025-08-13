from __future__ import annotations
from typing import Dict, List
from pathlib import Path

from agents.reporter_agent import ReporterAgent


class ReportGeneratorTool:
	def __init__(self):
		self.agent = ReporterAgent()

	def generate(self, query: str, plan: Dict, sources_by_task: Dict, summary: Dict, plagiarism_report: Dict, output_dir: str | Path) -> Dict:
		return self.agent.generate_report(query, plan, sources_by_task, summary, plagiarism_report, Path(output_dir))
