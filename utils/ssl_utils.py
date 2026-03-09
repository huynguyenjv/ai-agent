"""
SSL Utilities - Secure SSL configuration management.

Provides controlled SSL bypass for corporate environments without
compromising global security.
"""

import os
import ssl
import warnings
from typing import Optional

import httpx
import requests
import structlog

logger = structlog.get_logger()


class SSLConfig:
    """Manages SSL configuration for HTTP clients."""

    def __init__(self):
        self.disable_ssl_verify = os.getenv('DISABLE_SSL_VERIFY', 'false').lower() == 'true'
        
        if self.disable_ssl_verify:
            logger.warning(
                "⚠️  SSL verification disabled via DISABLE_SSL_VERIFY environment variable",
                recommendation="Only use in corporate environments with proxy issues"
            )
            # Suppress SSL warnings for cleaner logs
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            warnings.filterwarnings('ignore', message='Unverified HTTPS request')

    def create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context based on configuration."""
        if not self.disable_ssl_verify:
            return None
        
        # Create unverified context for corporate proxy environments
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context

    def configure_httpx_client(self, **kwargs) -> dict:
        """Configure httpx client with SSL settings."""
        if self.disable_ssl_verify:
            kwargs['verify'] = False
        return kwargs

    def configure_requests_session(self, session: requests.Session) -> requests.Session:
        """Configure requests session with SSL settings."""
        if self.disable_ssl_verify:
            session.verify = False
            
            # Custom adapter for SSL bypass — capture outer ssl_config reference
            outer_config = self

            class SSLAdapter(requests.adapters.HTTPAdapter):
                def init_poolmanager(self, *args, **kwargs):
                    kwargs['ssl_context'] = outer_config.create_ssl_context()
                    return super().init_poolmanager(*args, **kwargs)
            
            session.mount('https://', SSLAdapter())
        
        return session


# Global SSL config instance
ssl_config = SSLConfig()