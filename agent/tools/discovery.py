"""
Discovery Tools — help the agent explore the codebase.
"""
import os
import re
from typing import List, Optional
import structlog

logger = structlog.get_logger()

def list_files(directory: str = ".", recursive: bool = True, max_depth: int = 3) -> List[str]:
    """List files in a directory to understand structure.
    
    Args:
        directory: The directory to list.
        recursive: Whether to list subdirectories.
        max_depth: Maximum depth for recursion.
    """
    files = []
    base_path = os.path.abspath(directory)
    
    for root, dirs, filenames in os.walk(base_path):
        rel_path = os.relpath(root, base_path)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        
        if depth > max_depth:
            dirs[:] = [] # Stop recursion
            continue
            
        for f in filenames:
            p = os.path.join(rel_path, f) if rel_path != "." else f
            files.append(p)
            
        if not recursive:
            break
            
    return files

def grep_search(query: str, directory: str = ".", file_pattern: str = "*.java") -> List[dict]:
    """Search for text patterns across the codebase.
    
    Args:
        query: The string or regex to search for.
        directory: Directory to search in.
        file_pattern: Glob pattern for files to include.
    """
    import fnmatch
    results = []
    base_path = os.path.abspath(directory)
    
    pattern = re.compile(query, re.IGNORECASE)
    
    for root, _, filenames in os.walk(base_path):
        for filename in fnmatch.filter(filenames, file_pattern):
            full_path = os.path.join(root, filename)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for i, line in enumerate(f, 1):
                        if pattern.search(line):
                            results.append({
                                "file": os.path.relpath(full_path, base_path),
                                "line": i,
                                "content": line.strip()
                            })
                            if len(results) > 50: # Cap results
                                return results
            except Exception:
                continue
    return results

def read_file(path: str) -> str:
    """Read full content of a file for detailed inspection.
    
    Args:
        path: Relative or absolute path to the file.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {path}: {str(e)}"
