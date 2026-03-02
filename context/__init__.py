"""
Context package — smart context assembly for LLM prompts.

This package replaces the brute-force "dump all RAG chunks" approach with
a structured pipeline:

    1. ContextBuilder  — orchestrates context assembly
    2. SnippetSelector — picks the minimal set of code snippets
    3. TokenOptimizer  — enforces a token budget, truncates safely
"""

from .context_builder import ContextBuilder, ContextResult
from .snippet_selector import SnippetSelector, Snippet, SnippetRole
from .token_optimizer import TokenOptimizer

__all__ = [
    "ContextBuilder",
    "ContextResult",
    "SnippetSelector",
    "Snippet",
    "SnippetRole",
    "TokenOptimizer",
]
