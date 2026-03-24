"""GoPlugin — Section 6, Per-Language Resolution Logic.

Tree-sitter based Go parser. Extracts structs, interfaces, functions, methods.
Import resolution via go.mod module path.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import tree_sitter_go as tsgo
from tree_sitter import Language, Parser

from mcp_server.models import ChunkType, CodeChunk, ExtractionMode
from mcp_server.plugins.base import LanguagePlugin

GO_LANGUAGE = Language(tsgo.language())


class GoPlugin(LanguagePlugin):
    def __init__(self) -> None:
        self._parser = Parser(GO_LANGUAGE)

    def extensions(self) -> list[str]:
        return [".go"]

    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        tree = self._parser.parse(source_bytes)
        root = tree.root_node
        file_hash = hashlib.md5(source_bytes).hexdigest()
        chunks: list[CodeChunk] = []
        raw_imports: list[str] = []

        for node in self._walk(root):
            # Import specs
            if node.type == "import_spec":
                path_node = node.child_by_field_name("path")
                if path_node:
                    imp = path_node.text.decode("utf-8").strip('"')
                    raw_imports.append(imp)

            # Struct/interface type declarations
            elif node.type == "type_declaration":
                for child in node.children:
                    if child.type == "type_spec":
                        name_node = child.child_by_field_name("name")
                        type_node = child.child_by_field_name("type")
                        if name_node is None:
                            continue
                        name = name_node.text.decode("utf-8")
                        is_interface = type_node and type_node.type == "interface_type"
                        chunk_type = ChunkType.grouping

                        embed_text, body = self._extract_by_mode(child, name, mode)
                        chunks.append(CodeChunk(
                            chunk_id=CodeChunk.make_chunk_id("", path, name),
                            chunk_type=chunk_type,
                            symbol_name=name,
                            embed_text=embed_text,
                            body=body,
                            file_path=path,
                            lang="go",
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
                    lang="go",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

            # Method declarations (receiver)
            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                receiver_node = node.child_by_field_name("receiver")
                if name_node is None:
                    continue
                func_name = name_node.text.decode("utf-8")

                # Extract receiver type name
                recv_type = ""
                if receiver_node:
                    for child in self._walk(receiver_node):
                        if child.type == "type_identifier":
                            recv_type = child.text.decode("utf-8")
                            break

                full_name = f"{recv_type}.{func_name}" if recv_type else func_name
                embed_text, body = self._extract_by_mode(node, full_name, mode)
                chunks.append(CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, full_name),
                    chunk_type=ChunkType.callable,
                    symbol_name=full_name,
                    embed_text=embed_text,
                    body=body,
                    file_path=path,
                    lang="go",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

        return chunks

    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """Read go.mod to find the module path.

        If the import starts with the module path, the remainder
        is a relative path under the repo root.
        """
        go_mod_path = os.path.join(repo_root, "go.mod")
        if not os.path.isfile(go_mod_path):
            return None

        module_path = ""
        try:
            with open(go_mod_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("module "):
                        module_path = line.split(" ", 1)[1].strip()
                        break
        except OSError:
            return None

        if not module_path or not import_str.startswith(module_path):
            return None

        # Get relative package path
        rel_pkg = import_str[len(module_path):].lstrip("/")
        pkg_dir = Path(repo_root) / rel_pkg

        if pkg_dir.is_dir():
            # Return first .go file in the directory
            for f in sorted(pkg_dir.iterdir()):
                if f.suffix == ".go" and not f.name.endswith("_test.go"):
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

    @staticmethod
    def _walk(node):
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
