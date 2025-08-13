from __future__ import annotations
from typing import Dict, List
from pathlib import Path
from datetime import datetime
import os

try:
	import pypandoc
	PANDOC_AVAILABLE = True
except Exception:
	PANDOC_AVAILABLE = False

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class ReporterAgent:
	def _markdown(self, query: str, plan: Dict, sources_by_task: Dict[str, List[Dict]], summary: Dict, plagiarism_report: Dict) -> str:
		lines: List[str] = []
		lines.append(f"# Research Report: {query}\n")
		lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
		lines.append(f"**Query**: {query}\n")
		
		lines.append("## Executive Summary\n")
		sections = summary.get("sections", [])
		if sections:
			first_paragraphs = []
			for sec in sections[:3]:
				content = sec.get("content", "")
				if content and len(content) > 100:
					first_para = content.split('\n')[0][:200] + "..."
					first_paragraphs.append(first_para)
			lines.append(" ".join(first_paragraphs))
		lines.append("")
		
		lines.append("## Table of Contents\n")
		for i, sec in enumerate(sections, 1):
			lines.append(f"{i}. [{sec['question']}](#{sec['task_id'].lower()})")
		lines.append("")
		
		for i, sec in enumerate(sections, 1):
			lines.append(f"## {i}. {sec['question']}")
			content = sec.get("content", "No content available")
			lines.append(content)
			
			cits = sec.get("citations", [])
			if cits:
				lines.append("\n### Sources")
				for c in cits:
					lines.append(f"- {c}")
			lines.append("")
		
		lines.append("## Research Quality Assessment")
		total_sources = sum(len(sources_by_task.get(sec['question'], [])) for sec in sections)
		lines.append(f"- **Total sources analyzed**: {total_sources}")
		lines.append(f"- **Research tasks completed**: {len(sections)}")
		lines.append(f"- **Originality score**: {plagiarism_report.get('originality_score', 0.0):.1%}")
		
		if plagiarism_report.get("flags"):
			lines.append(f"- **Content rewritten**: {len(plagiarism_report['flags'])} segments improved for originality")
		
		return "\n".join(lines)

	def generate_report(self, query: str, plan: Dict, sources_by_task: Dict[str, List[Dict]], summary: Dict, plagiarism_report: Dict, output_dir: Path) -> Dict:
		output_dir.mkdir(parents=True, exist_ok=True)
		md_text = self._markdown(query, plan, sources_by_task, summary, plagiarism_report)
		md_path = output_dir / "report.md"
		md_path.write_text(md_text, encoding="utf-8")

		pdf_path = output_dir / "report.pdf"
		if PANDOC_AVAILABLE:
			try:
				pypandoc.convert_text(md_text, "pdf", format="md", outputfile=str(pdf_path))
			except Exception:
				self._pdf_fallback(str(md_path), str(pdf_path))
		else:
			self._pdf_fallback(str(md_path), str(pdf_path))

		return {"markdown": str(md_path), "pdf": str(pdf_path)}

	def _pdf_fallback(self, md_path: str, pdf_path: str) -> None:
		# Minimal fallback to ensure a PDF exists without pandoc
		text = Path(md_path).read_text(encoding="utf-8")
		c = canvas.Canvas(pdf_path, pagesize=letter)
		width, height = letter
		y = height - 72
		for line in text.splitlines():
			if y < 72:
				c.showPage()
				y = height - 72
			c.drawString(72, y, line[:100])
			y -= 14
		c.save()
