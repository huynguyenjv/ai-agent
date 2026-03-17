import os
import shutil
from agent.tools.java_build import detect_build_system, get_build_command

def test_build_detection():
    test_dir = "tmp/test_projects"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    # 1. Test Maven
    maven_dir = os.path.join(test_dir, "maven_proj")
    os.makedirs(maven_dir)
    open(os.path.join(maven_dir, "pom.xml"), "w").close()
    
    assert detect_build_system(maven_dir) == "maven"
    cmd = get_build_command("test", maven_dir)
    print(f"Maven test command: {cmd}")
    assert "mvn" in cmd[0] or "mvnw" in cmd[0]

    # 2. Test Gradle (Groovy)
    gradle_dir = os.path.join(test_dir, "gradle_proj")
    os.makedirs(gradle_dir)
    open(os.path.join(gradle_dir, "build.gradle"), "w").close()
    
    assert detect_build_system(gradle_dir) == "gradle"
    cmd = get_build_command("test", gradle_dir)
    print(f"Gradle test command: {cmd}")
    assert "gradle" in cmd[0] or "gradlew" in cmd[0]

    # 3. Test Gradle (Kotlin)
    gradle_kts_dir = os.path.join(test_dir, "gradle_kts_proj")
    os.makedirs(gradle_kts_dir)
    open(os.path.join(gradle_kts_dir, "build.gradle.kts"), "w").close()
    
    assert detect_build_system(gradle_kts_dir) == "gradle"

    # 4. Test missing
    empty_dir = os.path.join(test_dir, "empty_proj")
    os.makedirs(empty_dir)
    assert detect_build_system(empty_dir) == "unknown"

    print("\nVerification successful! Build system detection functions as expected.")

if __name__ == "__main__":
    test_build_detection()
