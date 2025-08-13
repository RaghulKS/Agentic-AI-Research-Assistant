from __future__ import annotations
from io import BytesIO
from typing import Optional

import fitz  # PyMuPDF


class PDFReaderTool:
	def read_from_bytes(self, content_bytes: bytes) -> str:
		with fitz.open(stream=BytesIO(content_bytes), filetype="pdf") as doc:
			texts = []
			for page in doc:
				texts.append(page.get_text())
			return "\n".join(texts)

	def read_local(self, path: str) -> str:
		with fitz.open(path) as doc:
			texts = []
			for page in doc:
				texts.append(page.get_text())
			return "\n".join(texts)
