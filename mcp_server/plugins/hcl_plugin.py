"""HCLPlugin — Section 6, HCL-Specific chunk_type Mapping.

Tree-sitter based Terraform/HCL parser.
Extracts resource, data, variable, output, module, provider blocks.
Local module resolution via source attribute.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import tree_sitter_hcl as tshcl
from tree_sitter import Language, Parser

from mcp_server.models import ChunkType, CodeChunk, ExtractionMode
from mcp_server.plugins.base import LanguagePlugin

HCL_LANGUAGE = Language(tshcl.language())

# Section 6: HCL construct to chunk_type + symbol_name mapping
_BLOCK_TYPE_MAP = {
    "resource": (ChunkType.config_block, lambda labels: f"{labels[0]}.{labels[1]}" if len(labels) >= 2 else labels[0]),
    "data": (ChunkType.config_block, lambda labels: f"data.{labels[0]}.{labels[1]}" if len(labels) >= 2 else f"data.{labels[0]}"),
    "variable": (ChunkType.config_block, lambda labels: f"var.{labels[0]}" if labels else "var.unknown"),
    "output": (ChunkType.config_block, lambda labels: f"output.{labels[0]}" if labels else "output.unknown"),
    "module": (ChunkType.grouping, lambda labels: f"module.{labels[0]}" if labels else "module.unknown"),
    "provider": (ChunkType.config_block, lambda labels: f"provider.{labels[0]}" if labels else "provider.unknown"),
    "locals": (ChunkType.config_block, lambda labels: "locals"),
    "terraform": (ChunkType.config_block, lambda labels: "terraform"),
}


class HCLPlugin(LanguagePlugin):
    def __init__(self) -> None:
        self._parser = Parser(HCL_LANGUAGE)

    def extensions(self) -> list[str]:
        return [".tf", ".hcl"]

    def extract_chunks(
        self, path: str, source_bytes: bytes, mode: ExtractionMode
    ) -> list[CodeChunk]:
        tree = self._parser.parse(source_bytes)
        root = tree.root_node
        file_hash = hashlib.md5(source_bytes).hexdigest()
        chunks: list[CodeChunk] = []
        raw_imports: list[str] = []

        # HCL AST: config_file > body > block
        body_node = root
        for child in root.children:
            if child.type == "body":
                body_node = child
                break

        for node in body_node.children:
            if node.type == "block":
                block_type, labels = self._parse_block(node)
                if block_type is None:
                    continue

                mapping = _BLOCK_TYPE_MAP.get(block_type)
                if mapping is None:
                    continue

                chunk_type, name_fn = mapping
                symbol_name = name_fn(labels)

                embed_text, body = self._extract_by_mode(node, symbol_name, mode)

                chunks.append(CodeChunk(
                    chunk_id=CodeChunk.make_chunk_id("", path, symbol_name),
                    chunk_type=chunk_type,
                    symbol_name=symbol_name,
                    embed_text=embed_text,
                    body=body,
                    file_path=path,
                    lang="hcl",
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    file_hash=file_hash,
                    raw_imports=raw_imports,
                ))

                # Extract source attribute for module blocks (import resolution)
                if block_type == "module":
                    source_val = self._extract_source_attr(node)
                    if source_val:
                        raw_imports.append(source_val)

        return chunks

    def resolve_dep_path(
        self, import_str: str, from_file: str, repo_root: str
    ) -> Path | None:
        """Section 6: source attribute in module block.

        If starts with ./ or ../, it is a local module (project).
        Otherwise third_party.
        """
        if import_str.startswith("./") or import_str.startswith("../"):
            from_dir = Path(from_file).parent
            candidate = Path(repo_root) / from_dir / import_str
            if candidate.is_dir():
                # Return the first .tf file in that module directory
                for f in sorted(candidate.iterdir()):
                    if f.suffix == ".tf":
                        return f.relative_to(repo_root)
            return None

        return None  # Remote module or registry — third_party

    @staticmethod
    def _parse_block(node) -> tuple[str | None, list[str]]:
        """Parse a block node to extract block type and labels."""
        children = list(node.children)
        if not children:
            return None, []

        block_type = None
        labels: list[str] = []

        for child in children:
            if child.type == "identifier":
                if block_type is None:
                    block_type = child.text.decode("utf-8")
                else:
                    labels.append(child.text.decode("utf-8"))
            elif child.type == "string_lit":
                # Extract text from template_literal inside string_lit
                text = ""
                for sc in child.children:
                    if sc.type == "template_literal":
                        text = sc.text.decode("utf-8")
                        break
                if not text:
                    text = child.text.decode("utf-8").strip('"')
                labels.append(text)

        return block_type, labels

    @staticmethod
    def _extract_source_attr(node) -> str | None:
        """Extract the 'source' attribute value from a module block."""
        for child in node.children:
            if child.type == "body":
                for attr in child.children:
                    if attr.type == "attribute":
                        key = attr.children[0] if attr.children else None
                        if key and key.text.decode("utf-8") == "source":
                            for val_child in attr.children:
                                if val_child.type in ("string_lit", "template_literal"):
                                    return val_child.text.decode("utf-8").strip('"')
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
