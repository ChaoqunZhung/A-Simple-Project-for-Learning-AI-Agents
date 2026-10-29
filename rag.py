"""Automatically refreshed PDF chunks and bilingual character TF-IDF retrieval."""
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import threading

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    chunk_id: int
    text: str


class PaperIndex:
    def __init__(self, papers_dir: Path, chunk_size: int = 1000, overlap: int = 150):
        if not 0 <= overlap < chunk_size:
            raise ValueError("Require 0 <= overlap < chunk_size")
        self.papers_dir = papers_dir.resolve()
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._lock = threading.RLock()
        self._fingerprint = None
        self._chunks: list[Chunk] = []
        self._warnings: list[str] = []
        self._vectorizer = None
        self._matrix = None

    def _discover(self) -> tuple[list[Path], tuple]:
        paths, signature = [], []
        for path in sorted(self.papers_dir.rglob("*")):
            if path.suffix.lower() != ".pdf" or not path.is_file():
                continue
            if not path.resolve().is_relative_to(self.papers_dir):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            paths.append(path)
            signature.append((path.relative_to(self.papers_dir).as_posix(), stat.st_mtime_ns, stat.st_size))
        return paths, tuple(signature)

    def _refresh(self) -> None:
        paths, fingerprint = self._discover()
        if fingerprint == self._fingerprint:
            return
        chunks, warnings = [], []
        for path in paths:
            source = path.relative_to(self.papers_dir).as_posix()
            try:
                if path.stat().st_size > 50 * 1024 * 1024:
                    warnings.append(f"{source}: 超过 50 MiB，已跳过。")
                    continue
                extracted = False
                with path.open("rb") as stream:
                    reader = PdfReader(stream)
                    if reader.is_encrypted and not reader.decrypt(""):
                        warnings.append(f"{source}: PDF 已加密，无法解析。")
                        continue
                    for page_number, page in enumerate(reader.pages, 1):
                        content = re.sub(r"\s+", " ", page.extract_text() or "").strip()
                        if not content:
                            continue
                        extracted = True
                        for offset in range(0, len(content), self.chunk_size - self.overlap):
                            text = content[offset:offset + self.chunk_size]
                            chunks.append(Chunk(source, page_number, len(chunks), text))
                            if offset + self.chunk_size >= len(content):
                                break
                if not extracted:
                    warnings.append(f"{source}: 未提取到文字，扫描 PDF 需要先做 OCR。")
            except Exception as exc:
                logger.warning("PDF parse failed (%s)", type(exc).__name__)
                warnings.append(f"{source}: PDF 解析失败，已跳过无法解析的内容。")
        vectorizer, matrix = None, None
        if chunks:
            # Character n-grams support both Chinese and English without a tokenizer.
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3),
                                         sublinear_tf=True, max_features=100000)
            matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
        self._chunks, self._warnings = chunks, warnings
        self._vectorizer, self._matrix = vectorizer, matrix
        self._fingerprint = fingerprint
        logger.info("PDF index refreshed: %d files, %d chunks", len(paths), len(chunks))

    def search(self, query: str, top_k: int = 5) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("query must contain 1–2000 characters")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        with self._lock:
            self._refresh()
            if self._matrix is None:
                return {"results": [], "message": "没有可检索文本。请将可提取文字的 PDF 放入 data/papers/。",
                        "warnings": self._warnings[:30]}
            vector = self._vectorizer.transform([query])
            scores = (self._matrix @ vector.T).toarray().ravel()
            results = []
            for idx in np.argsort(-scores, kind="stable")[:top_k]:
                if scores[idx] <= 0:
                    continue
                chunk = self._chunks[int(idx)]
                results.append({"source": chunk.source, "page": chunk.page,
                                "chunk_id": chunk.chunk_id, "score": round(float(scores[idx]), 6),
                                "text": chunk.text})
            return {"query": query, "results": results, "warnings": self._warnings[:30],
                    "message": "" if results else "没有匹配片段，请尝试论文中的具体术语。"}
