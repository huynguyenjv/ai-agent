"""Language Plugin Interface — Section 6, Interface Contract.

Every language plugin implements exactly three required methods
and one optional method.
"""

from __future__ import annotations

import abc
from pathlib import Path

from mcp_server.models import CodeChunk, ExtractionMode


class LanguagePlugin(abc.ABC):
    """Abstract base class for language plugins."""

    @abc.abstractmethod
    def extensions(self) -> list[str]:
        """Return file extensions this plugin handles.

        Example: [".java"] for Java, [".tf", ".hcl"] for Terraform.
        """

    @abc.abstractmethod
    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        """Parse the file and extract CodeChunk objects.

        Args:
            path: Relative file path from repo root.
            source_bytes: Raw file content.
            mode: Controls extraction detail level:
                - full_body: signature + docstring + complete source body
                - signatures: signature + docstring only; body is empty
                - names_only: symbol name only; all other text fields empty
        """

    @abc.abstractmethod
    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """Resolve an import string to a local file path.

        Returns None if the import cannot be resolved to a local project file.
        """

    def extract_skeleton(
        self, path: str, source_bytes: bytes, include_methods: bool = True
    ) -> dict:
        """Return a compact structural overview of the file.

        Default implementation uses extract_chunks in names_only mode.
        Override when the language's AST structure requires custom logic.
        """
        chunks = self.extract_chunks(path, source_bytes, ExtractionMode.names_only)
        classes = []
        functions = []
        for chunk in chunks:
            entry = {
                "name": chunk.symbol_name,
                "type": chunk.chunk_type.value,
                "line": chunk.start_line,
            }
            if chunk.chunk_type.value == "grouping":
                classes.append(entry)
            elif chunk.chunk_type.value == "callable" and include_methods:
                functions.append(entry)

        return {
            "file": path,
            "classes": classes,
            "functions": functions,
        }
