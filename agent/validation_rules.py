"""
Default validation rules for Java test generation.
Separates framework-specific patterns from the core validation logic.
"""

from .validation_schema import IssueSeverity, IssueCategory

# Forbidden patterns (ERROR — must fix)
FORBIDDEN_PATTERNS = [
    ("@SpringBootTest", "Use @ExtendWith(MockitoExtension.class) instead"),
    ("@DataJpaTest", "Use @ExtendWith(MockitoExtension.class) instead"),
    ("@WebMvcTest", "Use @ExtendWith(MockitoExtension.class) instead"),
    ("@SpringExtension", "Use MockitoExtension instead"),
    ("@ContextConfiguration", "Not needed for unit tests"),
    ("@RunWith(SpringRunner", "Use @ExtendWith(MockitoExtension.class)"),
    ("@Autowired", "Use @InjectMocks / @Mock instead"),
    ("@MockBean", "Use @Mock instead of @MockBean"),
    ("TestRestTemplate", "Use direct service calls with mocks"),
    ("MockMvc", "Use direct service calls with mocks"),
]

# Required patterns with severity
REQUIRED_PATTERNS = [
    ("@ExtendWith(MockitoExtension.class)", IssueSeverity.ERROR, IssueCategory.MISSING_ANNOTATION),
    ("@Mock", IssueSeverity.ERROR, IssueCategory.MISSING_ANNOTATION),
    ("@InjectMocks", IssueSeverity.ERROR, IssueCategory.MISSING_ANNOTATION),
    ("@Test", IssueSeverity.ERROR, IssueCategory.MISSING_TEST),
    ("@DisplayName", IssueSeverity.WARNING, IssueCategory.MISSING_DISPLAY_NAME),
]

# Anti-pattern definitions (Regex based)
ANTI_PATTERNS = [
    {
        "id": "security_context_no_mock",
        "pattern": r"\bSecurityContextHolder\b",
        "exclude": r"MockedStatic\s*<\s*SecurityContextHolder\s*>",
        "message": "SecurityContextHolder used without MockedStatic — will leak state between tests",
        "severity": IssueSeverity.ERROR,
        "category": IssueCategory.STATIC_CALL_WITHOUT_MOCK,
        "suggestion": "Use MockedStatic<SecurityContextHolder> with try-with-resources: try (MockedStatic<SecurityContextHolder> m = mockStatic(SecurityContextHolder.class)) { ... }"
    },
    {
        "id": "missing_mock_import_for_static",
        "pattern": r"\bSecurityContextHolder\b",
        "required_import": "import org.mockito.MockedStatic",
        "message": "Missing import for MockedStatic (needed for SecurityContextHolder)",
        "severity": IssueSeverity.WARNING,
        "category": IssueCategory.MISSING_IMPORT,
        "suggestion": "Add: import org.mockito.MockedStatic;"
    },
    {
        "id": "inject_mocks_no_mocks",
        "pattern": r"@InjectMocks\s+(?:private\s+)?(?P<class>\w+)\s+\w+",
        "check_func": "check_inject_mocks_has_mocks",
        "message": "@InjectMocks {class} has no @Mock fields — all dependencies will be null",
        "severity": IssueSeverity.ERROR,
        "category": IssueCategory.MISSING_MOCK_FIELD,
        "suggestion": "Add @Mock field for every constructor parameter of the service under test"
    }
]

# Common static utilities that usually need mocking
STATIC_UTILS = [
    {"pattern": r"\bLocalDateTime\.now\(\)", "name": "LocalDateTime.now()", "class": "LocalDateTime"},
    {"pattern": r"\bLocalDate\.now\(\)", "name": "LocalDate.now()", "class": "LocalDate"},
    {"pattern": r"\bInstant\.now\(\)", "name": "Instant.now()", "class": "Instant"},
    {"pattern": r"\bUUID\.randomUUID\(\)", "name": "UUID.randomUUID()", "class": "UUID"},
]
