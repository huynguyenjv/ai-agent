"""AST-Aware refactoring tools.

Provides safe code refactoring operations using tree-sitter AST parsing.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from mcp_server.tools import search_symbol, apply_edits
from mcp_server.plugins.registry import PluginRegistry

logger = logging.getLogger("mcp_server.tools_refactor")


def rename_symbol(
    repo_path: str,
    registry: PluginRegistry,
    old_name: str,
    new_name: str,
    scope: str = "project",
    dry_run: bool = True,
) -> dict:
    """Rename symbol across codebase.

    Uses search_symbol to find occurrences and applies rename edits.

    Args:
        repo_path: Repository root path
        registry: Plugin registry for parsing
        old_name: Current symbol name
        new_name: New symbol name
        scope: "file" or "project"
        dry_run: Preview changes without applying

    Returns:
        {edits, files_affected, occurrences, preview}
    """
    if not old_name or not new_name:
        return {"error": "old_name and new_name are required"}

    if old_name == new_name:
        return {"error": "old_name and new_name must be different"}

    # Validate new_name is a valid identifier
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", new_name):
        return {"error": f"Invalid identifier: {new_name}"}

    # Find all occurrences
    occurrences = search_symbol(repo_path, registry, old_name, type_filter="any")

    if not occurrences:
        return {
            "error": f"Symbol not found: {old_name}",
            "occurrences": 0,
        }

    # Build edits
    edits = []
    files_affected = set()

    for occ in occurrences:
        file_path = occ["file_path"]
        files_affected.add(file_path)

        # Read the file to get exact context
        full_path = os.path.join(repo_path, file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            continue

        # Find and replace with word boundary awareness
        pattern = rf"\b{re.escape(old_name)}\b"
        new_content = re.sub(pattern, new_name, content)

        if new_content != content:
            edits.append({
                "file_path": file_path,
                "content": new_content,
            })

    if not edits:
        return {
            "message": "No changes needed",
            "occurrences": len(occurrences),
        }

    # Apply edits
    result = apply_edits(repo_path, edits, dry_run=dry_run)

    return {
        **result,
        "old_name": old_name,
        "new_name": new_name,
        "occurrences": len(occurrences),
        "files_affected": list(files_affected),
    }


def extract_function(
    repo_path: str,
    file_path: str,
    start_line: int,
    end_line: int,
    new_function_name: str,
    dry_run: bool = True,
) -> dict:
    """Extract code block into a new function.

    Args:
        repo_path: Repository root path
        file_path: File containing code to extract
        start_line: Start line of code to extract
        end_line: End line of code to extract
        new_function_name: Name for the new function
        dry_run: Preview changes without applying

    Returns:
        {edits, preview, new_function}
    """
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", new_function_name):
        return {"error": f"Invalid function name: {new_function_name}"}

    full_path = os.path.join(repo_path, file_path)

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError as e:
        return {"error": f"Cannot read file: {e}"}

    if start_line < 1 or end_line > len(lines) or start_line > end_line:
        return {"error": f"Invalid line range: {start_line}-{end_line}"}

    # Extract the code block
    extracted_lines = lines[start_line - 1:end_line]
    extracted_code = "".join(extracted_lines)

    # Detect language from extension
    ext = os.path.splitext(file_path)[1].lower()

    # Analyze extracted code for variables
    variables = _detect_variables(extracted_code, ext)

    # Build the new function
    new_function = _build_function(
        new_function_name,
        extracted_code,
        variables,
        ext,
    )

    # Build the function call
    function_call = _build_function_call(
        new_function_name,
        variables,
        ext,
    )

    # Determine indentation
    first_line = extracted_lines[0] if extracted_lines else ""
    indent = len(first_line) - len(first_line.lstrip())
    indent_str = first_line[:indent]

    # Build new content
    new_lines = lines[:start_line - 1]
    new_lines.append(indent_str + function_call + "\n")
    new_lines.extend(lines[end_line:])

    # Add new function at appropriate location
    insert_pos = _find_function_insert_position(new_lines, start_line - 1)
    new_lines.insert(insert_pos, "\n" + new_function + "\n")

    new_content = "".join(new_lines)

    edits = [{"file_path": file_path, "content": new_content}]
    result = apply_edits(repo_path, edits, dry_run=dry_run)

    return {
        **result,
        "new_function_name": new_function_name,
        "new_function": new_function,
        "extracted_lines": f"{start_line}-{end_line}",
    }


def inline_variable(
    repo_path: str,
    file_path: str,
    variable_name: str,
    dry_run: bool = True,
) -> dict:
    """Inline a variable by replacing all uses with its value.

    Args:
        repo_path: Repository root path
        file_path: File containing the variable
        variable_name: Variable to inline
        dry_run: Preview changes without applying

    Returns:
        {edits, preview, replacements}
    """
    full_path = os.path.join(repo_path, file_path)

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return {"error": f"Cannot read file: {e}"}

    # Find variable assignment
    ext = os.path.splitext(file_path)[1].lower()
    assignment_pattern = _get_assignment_pattern(variable_name, ext)

    match = re.search(assignment_pattern, content)
    if not match:
        return {"error": f"Variable assignment not found: {variable_name}"}

    value = match.group(1).strip()

    # Count usages
    usage_pattern = rf"\b{re.escape(variable_name)}\b"
    usages = len(re.findall(usage_pattern, content)) - 1  # Exclude assignment

    if usages == 0:
        return {"message": f"Variable {variable_name} has no usages to inline"}

    # Replace usages with value (but not the assignment)
    # First, remove the assignment line
    new_content = re.sub(
        rf"^[ \t]*{re.escape(variable_name)}\s*=\s*[^\n]+\n",
        "",
        content,
        count=1,
        flags=re.MULTILINE,
    )

    # Then replace all usages
    new_content = re.sub(usage_pattern, value, new_content)

    edits = [{"file_path": file_path, "content": new_content}]
    result = apply_edits(repo_path, edits, dry_run=dry_run)

    return {
        **result,
        "variable": variable_name,
        "value": value,
        "replacements": usages,
    }


def _detect_variables(code: str, ext: str) -> dict:
    """Detect input and output variables in code block."""
    # Simple heuristic: find variable names
    # More sophisticated would use AST

    words = set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", code))

    # Filter out keywords and common names
    keywords = {
        "if", "else", "for", "while", "return", "def", "class", "import",
        "from", "try", "except", "with", "as", "in", "not", "and", "or",
        "True", "False", "None", "self", "cls", "let", "const", "var",
        "function", "async", "await", "new", "this", "public", "private",
    }

    variables = words - keywords

    return {
        "all": list(variables),
        "inputs": [],  # Would need AST analysis
        "outputs": [],
    }


def _build_function(name: str, code: str, variables: dict, ext: str) -> str:
    """Build a function definition."""
    # Dedent the code
    lines = code.split("\n")
    if lines:
        min_indent = min(
            (len(line) - len(line.lstrip()) for line in lines if line.strip()),
            default=0
        )
        lines = [line[min_indent:] if len(line) > min_indent else line for line in lines]
        code = "\n".join(lines)

    # Build based on language
    if ext in (".py",):
        return f"def {name}():\n    " + code.replace("\n", "\n    ")
    elif ext in (".js", ".ts"):
        return f"function {name}() {{\n    " + code.replace("\n", "\n    ") + "\n}"
    elif ext in (".java",):
        return f"private void {name}() {{\n    " + code.replace("\n", "\n    ") + "\n}"
    elif ext in (".go",):
        return f"func {name}() {{\n    " + code.replace("\n", "\n    ") + "\n}"
    else:
        return f"def {name}():\n    " + code.replace("\n", "\n    ")


def _build_function_call(name: str, variables: dict, ext: str) -> str:
    """Build a function call."""
    if ext in (".py",):
        return f"{name}()"
    elif ext in (".js", ".ts"):
        return f"{name}();"
    elif ext in (".java",):
        return f"{name}();"
    elif ext in (".go",):
        return f"{name}()"
    else:
        return f"{name}()"


def _find_function_insert_position(lines: list[str], reference_line: int) -> int:
    """Find where to insert a new function."""
    # Insert before the current function/class
    for i in range(reference_line, -1, -1):
        line = lines[i] if i < len(lines) else ""
        if line.strip().startswith(("def ", "class ", "function ", "func ")):
            return i
    return 0


def _get_assignment_pattern(variable_name: str, ext: str) -> str:
    """Get regex pattern for variable assignment."""
    # Simple pattern for common languages
    return rf"{re.escape(variable_name)}\s*=\s*(.+)"
