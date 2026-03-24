"""JavaPlugin — Section 6, Per-Language Resolution Logic.

Tree-sitter based Java parser. Extracts classes, interfaces, methods.
Import resolution via package-to-path mapping under src/main/java.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser

from mcp_server.models import ChunkType, CodeChunk, ExtractionMode
from mcp_server.plugins.base import LanguagePlugin

JAVA_LANGUAGE = Language(tsjava.language())


class JavaPlugin(LanguagePlugin):
    def __init__(self) -> None:
        self._parser = Parser(JAVA_LANGUAGE)

    def extensions(self) -> list[str]:
        return [".java"]

    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        tree = self._parser.parse(source_bytes)
        root = tree.root_node
        source_text = source_bytes.decode("utf-8", errors="replace")
        file_hash = hashlib.md5(source_bytes).hexdigest()
        chunks: list[CodeChunk] = []
        raw_imports: list[str] = []

        # Extract imports
        for node in self._walk(root):
            if node.type == "import_declaration":
                import_text = node.text.decode("utf-8", errors="replace").strip()
                # Remove 'import ' prefix and trailing ';'
                import_str = import_text.replace("import ", "").replace("static ", "").rstrip(";").strip()
                raw_imports.append(import_str)

        # Extract classes, interfaces, enums
        for node in self._walk(root):
            if node.type in ("class_declaration", "interface_declaration", "enum_declaration"):
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = name_node.text.decode("utf-8")
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1

                embed_text, body = self._extract_by_mode(node, name, source_text, mode)

                chunks.append(CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, name),
                    chunk_type=ChunkType.grouping,
                    symbol_name=name,
                    embed_text=embed_text,
                    body=body,
                    file_path=path,
                    lang="java",
                    start_line=start_line,
                    end_line=end_line,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

                # Extract methods within classes
                if node.type in ("class_declaration", "enum_declaration"):
                    body_node = node.child_by_field_name("body")
                    if body_node:
                        for child in body_node.children:
                            if child.type in ("method_declaration", "constructor_declaration"):
                                m_name_node = child.child_by_field_name("name")
                                if m_name_node is None:
                                    continue
                                m_name = m_name_node.text.decode("utf-8")
                                full_name = f"{name}.{m_name}"
                                m_start = child.start_point[0] + 1
                                m_end = child.end_point[0] + 1

                                m_embed, m_body = self._extract_by_mode(
                                    child, full_name, source_text, mode
                                )

                                chunks.append(CodeChunk(
                                    chunk_id=CodeChunk.make_chunk_id("", path, full_name),
                                    chunk_type=ChunkType.callable,
                                    symbol_name=full_name,
                                    embed_text=m_embed,
                                    body=m_body,
                                    file_path=path,
                                    lang="java",
                                    start_line=m_start,
                                    end_line=m_end,
                                    file_hash=file_hash,
                                    raw_imports=raw_imports,
                                ))

        return chunks

    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """Map fully-qualified class name to file path.

        Search under src/main/java. A dep is project if its package
        prefix matches the project's group ID from pom.xml.
        """
        # Convert dots to path separators
        rel_path = import_str.replace(".", "/") + ".java"

        # Search under common Java source roots
        for src_root in ["src/main/java", "src", ""]:
            candidate = Path(repo_root) / src_root / rel_path
            if candidate.is_file():
                return candidate.relative_to(repo_root)

        return None

    @staticmethod
    def _extract_by_mode(
        node, name: str, source_text: str, mode: ExtractionMode
    ) -> tuple[str, str]:
        """Return (embed_text, body) based on extraction mode."""
        if mode == ExtractionMode.names_only:
            return name, ""

        # Extract signature (first line up to opening brace)
        node_text = node.text.decode("utf-8", errors="replace")
        lines = node_text.split("\n")
        sig_lines = []
        for line in lines:
            sig_lines.append(line)
            if "{" in line:
                break
        signature = "\n".join(sig_lines).strip()

        if mode == ExtractionMode.signatures:
            return signature, ""

        # full_body
        return signature, node_text

    @staticmethod
    def _walk(node):
        """Yield all descendant nodes."""
        cursor = node.walk()
        visited = False
        while True:
            if not visited:
                yield cursor.node
                if cursor.goto_first_child():
                    continue
            if cursor.goto_next_sibling():
                visited = False
                continue
            if not cursor.goto_parent():
                break
            visited = True
