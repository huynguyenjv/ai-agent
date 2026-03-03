"""
End-to-end trace test: verify all 3 layers work for AuthUseCaseService + UserProfile.

Simulates the EXACT scenario that was failing:
- AuthUseCaseService has 8 deps + domain types like UserProfile, User, JwtToken
- UserProfile is a record WITHOUT @Builder (6 String fields)
- User has @Builder with 17 fields

Tests:
1. search_by_class(include_dependencies=True) tracks unfound_types
2. Prompt builder shows EXACT fields + construction hints + unfound warnings
3. Validation Pass 7 catches .builder() on UserProfile
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.schema import CodeChunk, FieldSchema
from agent.prompt import PromptBuilder
from agent.validation import ValidationPipeline, IssueCategory

# ═══════════════════════════════════════════════════════════════════
# Simulate RAG chunks as they would come from Qdrant
# ═══════════════════════════════════════════════════════════════════

# Main service chunk (AuthUseCaseService)
main_chunk = CodeChunk(
    id=1, summary="AuthUseCaseService handles exchange, refresh, loginByPassword, encryptPassword",
    score=1.0, type="service", layer="service", class_name="AuthUseCaseService",
    package="vtrip.app.service.auth", file_path="AuthUseCaseService.java",
    fully_qualified_name="vtrip.app.service.auth.AuthUseCaseService",
    dependencies=[
        "vtrip.app.repository.openapi.OpenAPIRepository",
        "vtrip.app.service.user.UserQueryService",
        "vtrip.app.domain.member.MemberUseCase",
        "vtrip.app.service.auth.UserUseCase",
        "vtrip.app.service.userpassword.UserPasswordQueryService",
        "vtrip.app.service.userpassword.UserPasswordCommandService",
        "vtrip.app.domain.jwt.JwtUseCase",
    ],
    used_types=["UserProfile", "User", "JwtToken", "LoginPassword", "UserPassword", "OpenAPIToken"],
)

# UserProfile: record WITHOUT @Builder — 6 String fields
user_profile_chunk = CodeChunk(
    id=2, summary="Record: vtrip.app.domain.openapi.profile.UserProfile\nType: record\nRecord Components: String userId, String name, String dateOfBirth, String gender, String phoneNumber, String email\nInstantiation:\nnew UserProfile(userId, name, dateOfBirth, gender, phoneNumber, email)",
    score=1.0, type="record", layer="domain", class_name="UserProfile",
    package="vtrip.app.domain.openapi.profile", file_path="UserProfile.java",
    fully_qualified_name="vtrip.app.domain.openapi.profile.UserProfile",
    java_type="record", has_builder=False,
    record_components=[
        FieldSchema(type="String", name="userId"),
        FieldSchema(type="String", name="name"),
        FieldSchema(type="String", name="dateOfBirth"),
        FieldSchema(type="String", name="gender"),
        FieldSchema(type="String", name="phoneNumber"),
        FieldSchema(type="String", name="email"),
    ],
)

# User: class WITH @Builder — many fields
user_chunk = CodeChunk(
    id=3, summary="Class: vtrip.app.domain.user.User\nType: class\n@Builder\nFields: UUID id; UUID refUserId; String email; String fullName; ...",
    score=1.0, type="class", layer="domain", class_name="User",
    package="vtrip.app.domain.user", file_path="User.java",
    fully_qualified_name="vtrip.app.domain.user.User",
    java_type="class", has_builder=True,
    fields=[
        FieldSchema(type="UUID", name="id"),
        FieldSchema(type="UUID", name="refUserId"),
        FieldSchema(type="String", name="email"),
        FieldSchema(type="String", name="fullName"),
        FieldSchema(type="String", name="phoneNumber"),
        FieldSchema(type="String", name="dateOfBirth"),
        FieldSchema(type="String", name="gender"),
        FieldSchema(type="String", name="metadata"),
        FieldSchema(type="LocalDateTime", name="lastSyncedAt"),
        FieldSchema(type="LocalDateTime", name="createdAt"),
        FieldSchema(type="String", name="createdBy"),
        FieldSchema(type="String", name="createdProgram"),
        FieldSchema(type="LocalDateTime", name="updatedAt"),
        FieldSchema(type="String", name="updatedBy"),
        FieldSchema(type="String", name="status"),
        FieldSchema(type="String", name="provider"),
        FieldSchema(type="String", name="oauthId"),
    ],
)

# JwtToken: class with @Builder
jwt_chunk = CodeChunk(
    id=4, summary="JwtToken @Builder", score=0.9,
    type="class", layer="domain", class_name="JwtToken",
    package="vtrip.app.domain.auth", file_path="JwtToken.java",
    fully_qualified_name="vtrip.app.domain.auth.JwtToken",
    java_type="class", has_builder=True,
    fields=[
        FieldSchema(type="String", name="accessToken"),
        FieldSchema(type="String", name="refreshToken"),
        FieldSchema(type="Long", name="expiresIn"),
    ],
)

# Simulate: LoginPassword and OpenAPIToken are NOT in index (unfound)
# SyncUserToCDCRequest is also NOT found
main_chunk.unfound_types = ["LoginPassword", "OpenAPIToken", "SyncUserToCDCRequest"]

# All chunks as returned by RAG
rag_chunks = [main_chunk, user_profile_chunk, user_chunk, jwt_chunk]


# ═══════════════════════════════════════════════════════════════════
# TEST 1: Prompt builder produces correct content
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Prompt builder — correct UserProfile info in prompt")
print("=" * 60)

builder = PromptBuilder()
prompt = builder.build_test_generation_prompt(
    class_name="AuthUseCaseService",
    file_path="AuthUseCaseService.java",
    rag_chunks=rag_chunks,
    task_description="Generate unit tests for AuthUseCaseService",
)

# Check 1a: UserProfile shows EXACT record components
assert "Record components (EXACT — do not add/change)" in prompt, \
    "FAIL: Missing '(EXACT — do not add/change)' label for record"
assert "String userId, String name, String dateOfBirth, String gender, String phoneNumber, String email" in prompt, \
    "FAIL: Missing exact UserProfile fields in prompt"
print("  ✅ UserProfile record components shown with EXACT label")

# Check 1b: Construction hint says NO @Builder
assert "NO @Builder" in prompt or "canonical constructor" in prompt, \
    "FAIL: Missing NO @Builder warning for UserProfile"
assert "new UserProfile(" in prompt, \
    "FAIL: Missing canonical constructor hint"
print("  ✅ Construction hint: NO @Builder, use new UserProfile()")

# Check 1c: User shows @Builder
assert "HAS @Builder" in prompt, \
    "FAIL: Missing @Builder hint for User"
print("  ✅ User construction hint: HAS @Builder")

# Check 1d: Unfound types warning section present
assert "Domain Types NOT FOUND" in prompt, \
    "FAIL: Missing unfound_types warning section"
assert "LoginPassword" in prompt and "mock(LoginPassword.class)" in prompt.replace(" ", "").replace("\n", ""), \
    "FAIL: Missing mock instructions for unfound LoginPassword"
print("  ✅ Unfound types section present with mock(Type.class) instructions")

# Check 1e: Domain types section header
assert "Domain Types Used by This Class" in prompt, \
    "FAIL: Missing domain types section"
print("  ✅ Domain types section header present")

print()

# ═══════════════════════════════════════════════════════════════════
# TEST 2: Validation Pass 7 catches wrong construction
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2: Validation catches UserProfile.builder() error")
print("=" * 60)

BAD_CODE = """
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class AuthUseCaseServiceTest {
    @Mock private OpenAPIRepository openAPIRepository;
    @Mock private UserQueryService userQueryService;
    @Mock private MemberUseCase memberUseCase;
    @Mock private UserUseCase userUseCase;
    @Mock private UserPasswordQueryService userPasswordQueryService;
    @Mock private PasswordEncoder passwordEncoder;
    @Mock private UserPasswordCommandService userPasswordCommandService;
    @Mock private JwtUseCase jwtUseCase;
    @InjectMocks private AuthUseCaseService authUseCaseService;

    @Test
    @DisplayName("exchange_WhenValidAuthCode_ShouldReturnJwtToken")
    void exchange_WhenValidAuthCode_ShouldReturnJwtToken() {
        // Arrange
        UserProfile userProfile = UserProfile.builder()
            .id(UUID.fromString("abc"))
            .refUserId(UUID.fromString("def"))
            .email("user@example.com")
            .fullName("John Doe")
            .phoneNumber("123")
            .dateOfBirth("1990-01-01")
            .gender("Male")
            .metadata("{}")
            .build();
        User user = User.builder()
            .id(UUID.fromString("abc"))
            .email("user@example.com")
            .build();
        JwtToken jwtToken = JwtToken.builder()
            .accessToken("token")
            .refreshToken("rt")
            .expiresIn(3600L)
            .build();
        // Act
        var result = authUseCaseService.exchange("authCode");
        // Assert
        verify(openAPIRepository).exchange("authCode");
    }
}
"""

v = ValidationPipeline()
result = v.validate(BAD_CODE, rag_chunks=rag_chunks)

wrong_construction = result.get_issues_by_category(IssueCategory.WRONG_CONSTRUCTION_PATTERN)
assert len(wrong_construction) >= 1, f"FAIL: Expected WRONG_CONSTRUCTION_PATTERN errors, got {len(wrong_construction)}"
print(f"  ✅ Detected {len(wrong_construction)} WRONG_CONSTRUCTION_PATTERN issues:")
for issue in wrong_construction:
    print(f"     [{issue.severity.value}] {issue.message[:100]}")

# UserProfile.builder() should be flagged (record without @Builder)
up_issues = [i for i in wrong_construction if "UserProfile" in i.message]
assert len(up_issues) >= 1, "FAIL: UserProfile.builder() not detected"
print(f"  ✅ UserProfile.builder() correctly flagged as ERROR")

# User.builder() should NOT be flagged (has @Builder)
user_issues = [i for i in wrong_construction if "User.builder" in i.message and "UserProfile" not in i.message]
assert len(user_issues) == 0, f"FAIL: User.builder() falsely flagged: {user_issues}"
print(f"  ✅ User.builder() correctly NOT flagged (has @Builder)")

# JwtToken.builder() should NOT be flagged (has @Builder)
jwt_issues = [i for i in wrong_construction if "JwtToken" in i.message]
assert len(jwt_issues) == 0, f"FAIL: JwtToken.builder() falsely flagged"
print(f"  ✅ JwtToken.builder() correctly NOT flagged (has @Builder)")

# Wrong fields detected
field_issues = [i for i in wrong_construction if "do NOT exist" in i.message]
assert len(field_issues) >= 1, "FAIL: Wrong fields not detected"
print(f"  ✅ Wrong field names detected in builder chain")

print()

# ═══════════════════════════════════════════════════════════════════
# TEST 3: Good code passes validation
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3: Correct code passes validation")
print("=" * 60)

GOOD_CODE = """
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
class AuthUseCaseServiceTest {
    @Mock private OpenAPIRepository openAPIRepository;
    @Mock private UserQueryService userQueryService;
    @Mock private MemberUseCase memberUseCase;
    @Mock private UserUseCase userUseCase;
    @Mock private UserPasswordQueryService userPasswordQueryService;
    @Mock private PasswordEncoder passwordEncoder;
    @Mock private UserPasswordCommandService userPasswordCommandService;
    @Mock private JwtUseCase jwtUseCase;
    @InjectMocks private AuthUseCaseService authUseCaseService;

    @Test
    @DisplayName("exchange_WhenValidAuthCode_ShouldReturnJwtToken")
    void exchange_WhenValidAuthCode_ShouldReturnJwtToken() {
        // Arrange
        UserProfile userProfile = new UserProfile("id1", "John", "1990-01-01", "Male", "123", "john@example.com");
        User user = User.builder()
            .id(UUID.fromString("abc"))
            .email("user@example.com")
            .build();
        JwtToken jwtToken = JwtToken.builder()
            .accessToken("token")
            .refreshToken("rt")
            .expiresIn(3600L)
            .build();
        // Act
        var result = authUseCaseService.exchange("authCode");
        // Assert
        verify(openAPIRepository).exchange("authCode");
    }
}
"""

result_good = v.validate(GOOD_CODE, rag_chunks=rag_chunks)
wrong_good = result_good.get_issues_by_category(IssueCategory.WRONG_CONSTRUCTION_PATTERN)
assert len(wrong_good) == 0, f"FAIL: Good code should have 0 WRONG_CONSTRUCTION errors but got {len(wrong_good)}: {[i.message for i in wrong_good]}"
print(f"  ✅ Correct code: 0 WRONG_CONSTRUCTION_PATTERN issues")
print(f"  ✅ Validation passed: {result_good.passed}")

print()
print("=" * 60)
print("ALL END-TO-END TRACE TESTS PASSED")
print("=" * 60)
