"""HashVerifier — Section 9, Data Volatility.

Verifies that chunks retrieved from Qdrant are still current
by comparing stored file_hash against current MD5 of the file.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger("server.hash_verifier")


class HashVerifier:
    """Verifies freshness of RAG results via file hash comparison.

    Used when freshness_signal is true or mentioned_files is non-empty.
    Chunks with mismatched hashes are silently discarded.
    """

    @staticmethod
    def verify_chunks(
        chunks: list[dict],
        file_contents: dict[str, bytes],
    ) -> list[dict]:
        """Keep only chunks whose file_hash matches current file content.

        Args:
            chunks: RAG result chunks with file_hash in payload.
            file_contents: Map of file_path to current file bytes.

        Returns:
            List of verified chunks.
        """
        verified = []
        for chunk in chunks:
            file_path = chunk.get("file_path", "")
            stored_hash = chunk.get("file_hash", "")

            if file_path not in file_contents:
                # Cannot verify — keep the chunk (server-side, no local fs access)
                verified.append(chunk)
                continue

            current_hash = hashlib.md5(file_contents[file_path]).hexdigest()
            if current_hash == stored_hash:
                verified.append(chunk)
            else:
                logger.debug(
                    "Discarding stale chunk %s (hash mismatch)",
                    chunk.get("symbol_name", "unknown"),
                )

        return verified
