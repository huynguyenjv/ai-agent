"""
Java Build Tools — automatic detection and execution for Maven and Gradle.
"""
import subprocess
import os
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()

def detect_build_system(root_path: str = ".") -> str:
    """Detect if project uses Maven or Gradle.
    
    Returns: 'maven', 'gradle', or 'unknown'
    """
    if os.path.exists(os.path.join(root_path, "pom.xml")):
        return "maven"
    if os.path.exists(os.path.join(root_path, "build.gradle")) or \
       os.path.exists(os.path.join(root_path, "build.gradle.kts")):
        return "gradle"
    return "unknown"

def get_build_command(base_cmd: str, root_path: str = ".") -> list[str]:
    """Get the appropriate command (wrapper or global) for the build system."""
    system = detect_build_system(root_path)
    
    if system == "maven":
        if os.name == 'nt': # Windows
            mvnw = os.path.join(root_path, "mvnw.cmd")
        else:
            mvnw = os.path.join(root_path, "mvnw")
            
        cmd_base = [mvnw] if os.path.exists(mvnw) else ["mvn"]
        
        if base_cmd == "compile":
            return cmd_base + ["compile", "-DskipTests"]
        elif base_cmd == "test":
            return cmd_base + ["test"]
            
    elif system == "gradle":
        if os.name == 'nt':
            gradlew = os.path.join(root_path, "gradlew.bat")
        else:
            gradlew = os.path.join(root_path, "gradlew")
            
        cmd_base = [gradlew] if os.path.exists(gradlew) else ["gradle"]
        
        if base_cmd == "compile":
            return cmd_base + ["classes", "testClasses"]
        elif base_cmd == "test":
            return cmd_base + ["test"]
            
    return []

async def run_compile() -> Dict[str, Any]:
    """Run compilation on the project."""
    # Try to get repo path from env if not in current dir
    repo_path = os.getenv("JAVA_REPO_PATH", ".")
    command = get_build_command("compile", repo_path)
    
    if not command:
        return {"passed": False, "error": "Could not detect build system (no pom.xml or build.gradle found)"}

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            cwd=repo_path
        )
        
        passed = process.returncode == 0
        return {
            "passed": passed,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "command": " ".join(command)
        }
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "command": " ".join(command)
        }

async def run_test(test_class: str) -> Dict[str, Any]:
    """Run a specific JUnit test class.
    
    Args:
        test_class: Simple name or FQN of the test class.
    """
    repo_path = os.getenv("JAVA_REPO_PATH", ".")
    system = detect_build_system(repo_path)
    command = get_build_command("test", repo_path)
    
    if not command:
        return {"passed": False, "error": "Could not detect build system"}

    if system == "maven":
        command.append(f"-Dtest={test_class}")
    elif system == "gradle":
        command.extend(["--tests", test_class])

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            cwd=repo_path
        )
        
        passed = process.returncode == 0
        return {
            "passed": passed,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "command": " ".join(command)
        }
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "command": " ".join(command)
        }
