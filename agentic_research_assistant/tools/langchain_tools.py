from __future__ import annotations
import json
import time
from typing import Dict, List, Optional, Type
from pathlib import Path

import requests
import fitz
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import settings


class WebSearchInput(BaseModel):
    query: str = Field(description="Search query to find relevant information")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information using DuckDuckGo and extract content from found pages"
    args_schema: Type[BaseModel] = WebSearchInput
    
    def __init__(self, timeout_seconds: Optional[int] = None):
        super().__init__()
        self.timeout_seconds = timeout_seconds or settings.timeout_seconds
        self.headers = {"User-Agent": settings.user_agent}

    def _run(
        self,
        query: str,
        max_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            enriched_results = []
            for i, result in enumerate(results, 1):
                url = result.get("href") or result.get("url")
                if not url:
                    continue
                
                content = self._fetch_content(url)
                if content and len(content) > 100:
                    enriched_results.append({
                        "id": i,
                        "title": result.get("title", ""),
                        "url": url,
                        "snippet": result.get("body", ""),
                        "content": content[:settings.max_content_length],
                        "timestamp": time.time()
                    })
            
            return json.dumps(enriched_results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return f"Search failed: {str(e)}"

    def _fetch_content(self, url: str) -> str:
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout_seconds)
            resp.raise_for_status()
            
            content_type = resp.headers.get("Content-Type", "").lower()
            if "application/pdf" in content_type or url.lower().endswith(".pdf"):
                return self._extract_pdf_text(resp.content)
            else:
                return self._extract_html_text(resp.text)
                
        except Exception:
            return ""

    def _extract_pdf_text(self, pdf_content: bytes) -> str:
        try:
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text[:8000]
        except Exception:
            return ""

    def _extract_html_text(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
                tag.extract()
            
            main_content = soup.find("main") or soup.find("article") or soup.find("div", class_=["content", "main"])
            if main_content:
                text = " ".join(main_content.get_text(" ").split())
            else:
                text = " ".join(soup.get_text(" ").split())
            
            return text[:8000]
        except Exception:
            return ""


class PlagiarismCheckInput(BaseModel):
    text: str = Field(description="Text content to check for plagiarism")
    sources: str = Field(description="JSON string of source materials to compare against")


class PlagiarismCheckTool(BaseTool):
    name: str = "plagiarism_check"
    description: str = "Check text content for potential plagiarism using semantic similarity analysis"
    args_schema: Type[BaseModel] = PlagiarismCheckInput

    def _run(
        self,
        text: str,
        sources: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            source_data = json.loads(sources)
            source_texts = []
            
            for source_list in source_data.values():
                for source in source_list:
                    content = source.get("content", "")
                    if content and len(content) > 100:
                        source_texts.append(content[:5000])
            
            if not source_texts or not text:
                return json.dumps({"originality_score": 1.0, "flags": []})
            
            vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
            all_texts = [text] + source_texts
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            max_similarity = similarities.max() if len(similarities) > 0 else 0.0
            originality_score = 1.0 - max_similarity
            
            flags = []
            if originality_score < settings.originality_threshold:
                text_segments = text.split('\n\n')
                for i, segment in enumerate(text_segments):
                    if len(segment) > 200:
                        flags.append({
                            "segment_id": i,
                            "text": segment[:500],
                            "similarity_score": max_similarity,
                            "reason": "High similarity to source material"
                        })
            
            return json.dumps({
                "originality_score": originality_score,
                "max_similarity": max_similarity,
                "flags": flags
            })
            
        except Exception as e:
            return json.dumps({"error": str(e), "originality_score": 1.0, "flags": []})


class ReportGenerationInput(BaseModel):
    title: str = Field(description="Title for the research report")
    content: str = Field(description="Main content of the report")
    metadata: str = Field(description="JSON metadata including sources, stats, etc.")


class ReportGenerationTool(BaseTool):
    name: str = "generate_report"
    description: str = "Generate a professional research report in Markdown and PDF formats"
    args_schema: Type[BaseModel] = ReportGenerationInput

    def _run(
        self,
        title: str,
        content: str,
        metadata: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            from datetime import datetime
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas
            
            reports_dir = Path(settings.reports_dir)
            reports_dir.mkdir(exist_ok=True)
            
            meta = json.loads(metadata) if metadata else {}
            
            md_content = f"""# {title}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

{content[:500]}...

## Detailed Analysis

{content}

## Research Metadata

- **Sources Analyzed**: {meta.get('total_sources', 'Unknown')}
- **Originality Score**: {meta.get('originality_score', 'N/A')}
- **Research Tasks**: {meta.get('num_tasks', 'Unknown')}

---
*Generated by Agentic Research Assistant*
"""
            
            md_path = reports_dir / "research_report.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            pdf_path = reports_dir / "research_report.pdf"
            try:
                import pypandoc
                pypandoc.convert_text(md_content, "pdf", format="md", outputfile=str(pdf_path))
            except Exception:
                self._create_simple_pdf(str(pdf_path), title, content)
            
            return json.dumps({
                "markdown_path": str(md_path),
                "pdf_path": str(pdf_path),
                "status": "success"
            })
            
        except Exception as e:
            return json.dumps({"error": str(e), "status": "failed"})

    def _create_simple_pdf(self, path: str, title: str, content: str):
        c = canvas.Canvas(path, pagesize=letter)
        width, height = letter
        y = height - 50
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, title)
        y -= 40
        
        c.setFont("Helvetica", 12)
        lines = content.split('\n')
        for line in lines:
            if y < 50:
                c.showPage()
                y = height - 50
            c.drawString(50, y, line[:80])
            y -= 15
        
        c.save()
