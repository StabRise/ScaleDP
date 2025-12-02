import time
from types import MappingProxyType
from typing import Any

from pyspark import keyword_only
from semantic_text_splitter import TextSplitter as SemanticTextSplitter

from scaledp.schemas.Document import Document
from scaledp.schemas.TextChunks import TextChunks

from .BaseTextSplitter import BaseTextSplitter


class TextSplitter(BaseTextSplitter):
    """
    Text splitter implementation using semantic_text_splitter library.
    """

    defaultParams = MappingProxyType({**BaseTextSplitter.defaultParams})

    @keyword_only
    def __init__(self, **kwargs: Any):
        super(TextSplitter, self).__init__()
        self._setDefault(**self.defaultParams)
        self._set(**kwargs)
        # Note: splitter is created lazily in split() method because
        # chunk_size and chunk_overlap may be changed after initialization

    def _get_splitter(self):
        """Get or create the semantic text splitter with current parameters."""
        chunk_size = self.getOrDefault("chunk_size")
        return SemanticTextSplitter(chunk_size)

    def split(self, document: Document, page_number: int) -> TextChunks:
        start_time = time.time()
        try:
            splitter = self._get_splitter()
            chunks = splitter.chunks(document.text)
            exception = ""
        except Exception as e:
            chunks = []
            exception = str(e)
        processing_time = time.time() - start_time
        return TextChunks(
            path=document.path,
            chunks=chunks,
            page=page_number,
            exception=exception,
            processing_time=processing_time,
        )
