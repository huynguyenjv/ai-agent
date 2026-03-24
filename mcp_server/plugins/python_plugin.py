"""PythonPlugin — Section 6, Per-Language Resolution Logic.

Tree-sitter based Python parser. Extracts classes, functions, async functions.
Relative and absolute import resolution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from mcp_server.models import ChunkType, CodeChunk, ExtractionMode
from mcp_server.plugins.base import LanguagePlugin

PY_LANGUAGE = Language(tspython.language())


class PythonPlugin(LanguagePlugin):
    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def extensions(self) -> list[str]:
        return [".py"]

    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        tree = self._parser.parse(source_bytes)
        root = tree.root_node
        file_hash = hashlib.md5(source_bytes).hexdigest()
        chunks: list[CodeChunk] = []
        raw_imports: list[str] = []

        for node in root.children:
            # Imports
            if node.type == "import_statement":
                raw_imports.append(node.text.decode("utf-8").strip())
            elif node.type == "import_from_statement":
                raw_imports.append(node.text.decode("utf-8").strip())

            # Classes
            elif node.type == "class_definition":
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
                    lang="python",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

                # Extract methods within the class
                class_body = node.child_by_field_name("body")
                if class_body:
                    for child in class_body.children:
                        if child.type in ("function_definition", "decorated_definition"):
                            func_node = child
                            if child.type == "decorated_definition":
                                # Get the actual function inside decorated_definition
                                for c in child.children:
                                    if c.type == "function_definition":
                                        func_node = c
                                        break
                            m_name_node = func_node.child_by_field_name("name")
                            if m_name_node is None:
                                continue
                            m_name = m_name_node.text.decode("utf-8")
                            full_name = f"{name}.{m_name}"
                            m_embed, m_body = self._extract_by_mode(func_node, full_name, mode)

                            chunks.append(CodeChunk(
                                chunk_id=CodeChunk.make_chunk_id("", path, full_name),
                                chunk_type=ChunkType.callable,
                                symbol_name=full_name,
                                embed_text=m_embed,
                                body=m_body,
                                file_path=path,
                                lang="python",
                                start_line=func_node.start_point[0] + 1,
                                end_line=func_node.end_point[0] + 1,
                                file_hash=file_hash,
                                raw_imports=raw_imports,
                            ))

            # Top-level functions
            elif node.type in ("function_definition", "decorated_definition"):
                func_node = node
                if node.type == "decorated_definition":
                    for c in node.children:
                        if c.type == "function_definition":
                            func_node = c
                            break
                name_node = func_node.child_by_field_name("name")
                if name_node is None:
                    continue
                name = name_node.text.decode("utf-8")
                embed_text, body = self._extract_by_mode(func_node, name, mode)

                chunks.append(CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, name),
                    chunk_type=ChunkType.callable,
                    symbol_name=name,
                    embed_text=embed_text,
                    body=body,
                    file_path=path,
                    lang="python",
                    start_line=func_node.start_point[0] + 1,
                    end_line=func_node.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

        return chunks

    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """Handle relative imports (.module, ..module) and absolute imports.

        For absolute imports, search for a matching file or directory under src/ or repo root.
        """
        # Parse import string
        # Handle "from .foo import bar" or "import foo.bar"
        clean = import_str
        for prefix in ("from ", "import "):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        # Take just the module path (before ' import ')
        if " import " in clean:
            clean = clean.split(" import ")[0].strip()

        # Relative imports
        if clean.startswith("."):
            dots = 0
            for ch in clean:
                if ch == ".":
                    dots += 1
                else:
                    break
            module_part = clean[dots:]
            from_dir = Path(from_file).parent
            # Go up (dots) directories for relative imports
            # e.g. "from ..foo" has dots=2, go up 2 levels
            for _ in range(dots):
                from_dir = from_dir.parent

            rel = from_dir / module_part.replace(".", "/")
            for candidate in [
                Path(repo_root) / f"{rel}.py",
                Path(repo_root) / rel / "__init__.py",
            ]:
                if candidate.is_file():
                    return candidate.relative_to(repo_root)
            return None

        # Absolute imports
        module_path = clean.replace(".", "/")
        for src_root in ["src", ""]:
            base = Path(repo_root) / src_root if src_root else Path(repo_root)
            for candidate in [
                base / f"{module_path}.py",
                base / module_path / "__init__.py",
            ]:
                if candidate.is_file():
                    return candidate.relative_to(repo_root)

        return None

    @staticmethod
    def _extract_by_mode(node, name: str, mode: ExtractionMode) -> tuple[str, str]:
        if mode == ExtractionMode.names_only:
            return name, ""

        node_text = node.text.decode("utf-8", errors="replace")
        lines = node_text.split("\n")

        # Signature = def line + docstring
        sig_lines = [lines[0]]
        # Check for docstring
        if len(lines) > 1:
            stripped = lines[1].strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                sig_lines.append(lines[1])
                if not stripped[3:].rstrip().endswith(quote):
                    for line in lines[2:]:
                        sig_lines.append(line)
                        if quote in line:
                            break

        signature = "\n".join(sig_lines).strip()

        if mode == ExtractionMode.signatures:
            return signature, ""
        return signature, node_text
