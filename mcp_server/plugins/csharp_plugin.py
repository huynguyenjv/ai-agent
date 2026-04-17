"""CSharpPlugin — Section 6, Per-Language Resolution Logic.

Tree-sitter based C# parser. Extracts classes, interfaces, methods, properties.
Namespace-to-path resolution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser

from mcp_server.models import ChunkType, CodeChunk, ExtractionMode
from mcp_server.plugins.base import LanguagePlugin

CSHARP_LANGUAGE = Language(tscsharp.language())


class CSharpPlugin(LanguagePlugin):
    def __init__(self) -> None:
        self._parser = Parser(CSHARP_LANGUAGE)

    def extensions(self) -> list[str]:
        return [".cs"]

    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        tree = self._parser.parse(source_bytes)
        root = tree.root_node
        file_hash = hashlib.md5(source_bytes).hexdigest()
        chunks: list[CodeChunk] = []
        raw_imports: list[str] = []

        self._extract_recursive(root, path, file_hash, mode, chunks, raw_imports)
        return chunks

    def _extract_recursive(
        self,
        node,
        path: str,
        file_hash: str,
        mode: ExtractionMode,
        chunks: list[CodeChunk],
        raw_imports: list[str],
        parent_name: str = "",
    ) -> None:
        for child in node.children:
            # Using directives
            if child.type == "using_directive":
                raw_imports.append(child.text.decode("utf-8").strip())

            # Namespace
            elif child.type in ("namespace_declaration", "file_scoped_namespace_declaration"):
                name_node = child.child_by_field_name("name")
                ns_name = name_node.text.decode("utf-8") if name_node else ""
                body = child.child_by_field_name("body")
                if body:
                    self._extract_recursive(body, path, file_hash, mode, chunks, raw_imports, ns_name)
                else:
                    # File-scoped namespace
                    self._extract_recursive(child, path, file_hash, mode, chunks, raw_imports, ns_name)

            # Classes, interfaces, structs, enums, records
            elif child.type in (
                "class_declaration", "interface_declaration",
                "struct_declaration", "enum_declaration", "record_declaration",
            ):
                name_node = child.child_by_field_name("name")
                if name_node is None:
                    continue
                name = name_node.text.decode("utf-8")
                full_name = f"{parent_name}.{name}" if parent_name else name

                embed_text, body_text = self._extract_by_mode(child, full_name, mode)
                chunks.append(CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, full_name),
                    chunk_type=ChunkType.grouping,
                    symbol_name=full_name,
                    embed_text=embed_text,
                    body=body_text,
                    file_path=path,
                    lang="csharp",
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

                # Extract methods
                body_node = child.child_by_field_name("body")
                if body_node:
                    for member in body_node.children:
                        if member.type in ("method_declaration", "constructor_declaration"):
                            m_name_node = member.child_by_field_name("name")
                            if m_name_node is None:
                                continue
                            m_name = m_name_node.text.decode("utf-8")
                            m_full = f"{full_name}.{m_name}"
                            m_embed, m_body = self._extract_by_mode(member, m_full, mode)
                            chunks.append(CodeChunk(
                                chunk_id=CodeChunk.make_chunk_id("", path, m_full),
                                chunk_type=ChunkType.callable,
                                symbol_name=m_full,
                                embed_text=m_embed,
                                body=m_body,
                                file_path=path,
                                lang="csharp",
                                start_line=member.start_point[0] + 1,
                                end_line=member.end_point[0] + 1,
                                file_hash=file_hash,
                                raw_imports=raw_imports,
                            ))

    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """Map namespace to file path by converting dots to directory separators.

        Search under the project root.
        """
        # Clean 'using ' prefix and trailing ';'
        clean = import_str.replace("using ", "").rstrip(";").strip()
        # Remove 'static ' if present
        clean = clean.replace("static ", "")

        rel_path = clean.replace(".", "/") + ".cs"
        candidate = Path(repo_root) / rel_path
        if candidate.is_file():
            return candidate.relative_to(repo_root)

        # Try matching just the last part (class name)
        parts = clean.split(".")
        if parts:
            class_name = parts[-1]
            # Search for ClassName.cs recursively
            for f in Path(repo_root).rglob(f"{class_name}.cs"):
                return f.relative_to(repo_root)

        return None

    @staticmethod
    def _extract_by_mode(node, name: str, mode: ExtractionMode) -> tuple[str, str]:
        if mode == ExtractionMode.names_only:
            return name, ""

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
        return signature, node_text
