"""
Main entry point for the AI Coding Agent.
"""

import os
import sys
import ssl
import warnings

# =============================================================================
# CORPORATE PROXY & SSL BYPASS - Configurable via environment
# Set DISABLE_SSL_VERIFY=true only in corporate environments with proxy issues
# =============================================================================

# Sentence transformers model path
os.environ['SENTENCE_TRANSFORMERS_HOME'] = os.getenv('SENTENCE_TRANSFORMERS_HOME', './models')

# HuggingFace telemetry
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

# Check if SSL bypass is enabled (default: False for production safety)
DISABLE_SSL_VERIFY = os.getenv('DISABLE_SSL_VERIFY', 'false').lower() == 'true'

if DISABLE_SSL_VERIFY:
    # WARNING: Only use in corporate environments with proxy issues
    print("⚠️  WARNING: SSL verification is DISABLED. This is insecure for production!", file=sys.stderr)
    
    # Disable SSL verification globally
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['SSL_CERT_DIR'] = ''
    os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'

    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.filterwarnings('ignore', message='Unverified HTTPS request')
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    # Monkey-patch SSL context để bypass verification
    _original_create_default_context = ssl.create_default_context

    def _create_unverified_context(*args, **kwargs):
        context = _original_create_default_context(*args, **kwargs)
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    ssl.create_default_context = _create_unverified_context
    ssl._create_default_https_context = ssl._create_unverified_context

    # Patch requests để không verify SSL
    try:
        import requests
        from requests.adapters import HTTPAdapter
        
        class SSLAdapter(HTTPAdapter):
            def init_poolmanager(self, *args, **kwargs):
                kwargs['ssl_context'] = ssl._create_unverified_context()
                return super().init_poolmanager(*args, **kwargs)
        
        # Patch Session
        _original_session_init = requests.Session.__init__
        
        def _patched_session_init(self, *args, **kwargs):
            _original_session_init(self, *args, **kwargs)
            self.verify = False
            self.mount('https://', SSLAdapter())
        
        requests.Session.__init__ = _patched_session_init
    except ImportError:
        pass

# =============================================================================

import structlog
import logging
import uvicorn
from dotenv import load_dotenv

# Set root logger level from env (defaults to INFO — silences DEBUG from httpx, urllib3, etc.)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(message)s")
# Quiet noisy third-party loggers
for _name in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def main():
    """Run the AI Agent server."""
    load_dotenv()

    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8080"))
    log_level = os.getenv("LOG_LEVEL", "INFO").lower()

    logger.info(
        "Starting AI Coding Agent",
        host=host,
        port=port,
        log_level=log_level,
    )

    uvicorn.run(
        "server.api:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=os.getenv("DEBUG", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()

