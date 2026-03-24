"""TypeScriptPlugin — Section 6, Per-Language Resolution Logic.

Tree-sitter based TypeScript/JavaScript parser.
Extracts classes, functions, arrow functions, interfaces.
Resolution by ./ ../ prefix detection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import tree_sitter_javascript as tsjs
from tree_sitter import Language, Parser

from mcp_server.models import ChunkType, CodeChunk, ExtractionMode
from mcp_server.plugins.base import LanguagePlugin

JS_LANGUAGE = Language(tsjs.language())


class TypeScriptPlugin(LanguagePlugin):
    """Handles .ts, .tsx, .js, .jsx files."""

    def __init__(self) -> None:
        self._parser = Parser(JS_LANGUAGE)

    def extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx"]

    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        tree = self._parser.parse(source_bytes)
        root = tree.root_node
        file_hash = hashlib.md5(source_bytes).hexdigest()
        chunks: list[CodeChunk] = []
        raw_imports: list[str] = []

        for node in root.children:
            # Import declarations
            if node.type == "import_statement":
                source_node = node.child_by_field_name("source")
                if source_node:
                    imp = source_node.text.decode("utf-8").strip("'\"")
                    raw_imports.append(imp)

            # Class declarations
            elif node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = name_node.text.decode("utf-8")
                embed_text, body = self._extract_by_mode(node, name, mode)

                chunks.append(CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, name),
                    chunk_type=ChunkType.grouping,
                    symbol_name=name,
                    embed_text=embed_text,
                    body=body,
                    file_path=path,
                    lang="typescript",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

                # Methods in class
                class_body = node.child_by_field_name("body")
                if class_body:
                    for child in class_body.children:
                        if child.type == "method_definition":
                            m_name_node = child.child_by_field_name("name")
                            if m_name_node is None:
                                continue
                            m_name = m_name_node.text.decode("utf-8")
                            full_name = f"{name}.{m_name}"
                            m_embed, m_body = self._extract_by_mode(child, full_name, mode)
                            chunks.append(CodeChunk(
                                chunk_id=CodeChunk.make_chunk_id("", path, full_name),
                                chunk_type=ChunkType.callable,
                                symbol_name=full_name,
                                embed_text=m_embed,
                                body=m_body,
                                file_path=path,
                                lang="typescript",
                                start_line=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                file_hash=file_hash,
                                raw_imports=raw_imports,
                            ))

            # Function declarations
            elif node.type == "function_declaration":
                name_node = node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = name_node.text.decode("utf-8")
                embed_text, body = self._extract_by_mode(node, name, mode)
                chunks.append(CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, name),
                    chunk_type=ChunkType.callable,
                    symbol_name=name,
                    embed_text=embed_text,
                    body=body,
                    file_path=path,
                    lang="typescript",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

            # Export statements may contain functions/classes
            elif node.type == "export_statement":
                for child in node.children:
                    if child.type == "class_declaration":
                        cls_name_node = child.child_by_field_name("name")
                        if cls_name_node is None:
                            continue
                        cls_name = cls_name_node.text.decode("utf-8")
                        embed_text, body = self._extract_by_mode(child, cls_name, mode)
                        chunks.append(CodeChunk(
                            chunk_id=CodeChunk.make_chunk_id("", path, cls_name),
                            chunk_type=ChunkType.grouping,
                            symbol_name=cls_name,
                            embed_text=embed_text,
                            body=body,
                            file_path=path,
                            lang="typescript",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            file_hash=file_hash,
                            raw_imports=raw_imports,
                        ))
                        # Methods in exported class
                        cls_body = child.child_by_field_name("body")
                        if cls_body:
                            for member in cls_body.children:
                                if member.type == "method_definition":
                                    m_name_node = member.child_by_field_name("name")
                                    if m_name_node is None:
                                        continue
                                    m_name = m_name_node.text.decode("utf-8")
                                    m_full = f"{cls_name}.{m_name}"
                                    m_embed, m_body = self._extract_by_mode(member, m_full, mode)
                                    chunks.append(CodeChunk(
                                        chunk_id=CodeChunk.make_chunk_id("", path, m_full),
                                        chunk_type=ChunkType.callable,
                                        symbol_name=m_full,
                                        embed_text=m_embed,
                                        body=m_body,
                                        file_path=path,
                                        lang="typescript",
                                        start_line=member.start_point[0] + 1,
                                        end_line=member.end_point[0] + 1,
                                        file_hash=file_hash,
                                        raw_imports=raw_imports,
                                    ))

                    elif child.type == "function_declaration":
                        fn_name_node = child.child_by_field_name("name")
                        if fn_name_node is None:
                            continue
                        fn_name = fn_name_node.text.decode("utf-8")
                        embed_text, body = self._extract_by_mode(child, fn_name, mode)
                        chunks.append(CodeChunk(
                            chunk_id=CodeChunk.make_chunk_id("", path, fn_name),
                            chunk_type=ChunkType.callable,
                            symbol_name=fn_name,
                            embed_text=embed_text,
                            body=body,
                            file_path=path,
                            lang="typescript",
                            start_line=child.start_point[0] + 1,
                            end_line=child.end_point[0] + 1,
                            file_hash=file_hash,
                            raw_imports=raw_imports,
                        ))

            # Arrow functions assigned to const/let/var
            elif node.type in ("lexical_declaration", "variable_declaration"):
                for decl in node.children:
                    if decl.type == "variable_declarator":
                        name_node = decl.child_by_field_name("name")
                        value_node = decl.child_by_field_name("value")
                        if name_node and value_node and value_node.type == "arrow_function":
                            name = name_node.text.decode("utf-8")
                            embed_text, body = self._extract_by_mode(decl, name, mode)
                            chunks.append(CodeChunk(
                                chunk_id=CodeChunk.make_chunk_id("", path, name),
                                chunk_type=ChunkType.callable,
                                symbol_name=name,
                                embed_text=embed_text,
                                body=body,
                                file_path=path,
                                lang="typescript",
                                start_line=decl.start_point[0] + 1,
                                end_line=decl.end_point[0] + 1,
                                file_hash=file_hash,
                                raw_imports=raw_imports,
                            ))

        return chunks

    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """Section 6: If import starts with ./ or ../, it is always project.

        Resolve relative to the importing file, trying .ts, .tsx, .js, .jsx,
        and index.ts extensions. If it does not start with ., it is third_party
        unless it starts with node:, which makes it stdlib.
        """
        if not import_str.startswith("."):
            return None  # third_party or stdlib — handled by DepClassifier

        from_dir = Path(repo_root) / Path(from_file).parent
        base = from_dir / import_str

        # Try exact path first
        if base.is_file():
            return base.relative_to(repo_root)

        # Try extensions
        for ext in [".ts", ".tsx", ".js", ".jsx"]:
            candidate = base.with_suffix(ext)
            if candidate.is_file():
                return candidate.relative_to(repo_root)

        # Try index files
        for index in ["index.ts", "index.tsx", "index.js", "index.jsx"]:
            candidate = base / index
            if candidate.is_file():
                return candidate.relative_to(repo_root)

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
