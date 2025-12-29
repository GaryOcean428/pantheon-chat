"""
QIG Tokenizer Storage Layer
===========================

Geometric storage for tokenizer vocabulary using Redis and PostgreSQL.
Replaces JSON storage with proper database persistence.

Storage Architecture:
- Redis: Fast token lookups, caching, merge rule access
- PostgreSQL: Persistent vocab storage, versioning, audit trail
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np


class TokenizerStorage(ABC):
    """Abstract base for tokenizer storage backends."""

    @abstractmethod
    def save_vocab(
        self,
        vocab: dict[int, bytes],
        merge_rules: list[tuple[int, int, int]],
        metadata: dict[str, Any],
    ) -> str:
        """Save vocabulary and return version ID."""
        pass

    @abstractmethod
    def load_vocab(
        self, version_id: str | None = None
    ) -> tuple[dict[int, bytes], list[tuple[int, int, int]], dict[str, Any]]:
        """Load vocabulary. If version_id is None, load latest."""
        pass

    @abstractmethod
    def save_special_tokens(self, special_tokens: dict[str, np.ndarray]) -> None:
        """Save geometric special token basin coordinates."""
        pass

    @abstractmethod
    def load_special_tokens(self) -> dict[str, np.ndarray]:
        """Load geometric special token basin coordinates."""
        pass


class RedisStorage(TokenizerStorage):
    """Redis-based storage for fast token lookups."""

    def __init__(self, redis_url: str | None = None):
        try:
            import redis
        except ImportError:
            raise ImportError("redis package required: pip install redis")

        self.redis_url = redis_url or os.getenv("REDIS_URL")
        if not self.redis_url:
            raise ValueError("REDIS_URL not set")

        self.client = redis.from_url(self.redis_url)
        self.prefix = "qig_tokenizer:"

    def save_vocab(
        self,
        vocab: dict[int, bytes],
        merge_rules: list[tuple[int, int, int]],
        metadata: dict[str, Any],
    ) -> str:
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        vocab_key = f"{self.prefix}vocab:{version_id}"
        vocab_data = {str(k): v.hex() for k, v in vocab.items()}
        self.client.hset(vocab_key, mapping=vocab_data)

        rules_key = f"{self.prefix}rules:{version_id}"
        self.client.set(rules_key, json.dumps(merge_rules))

        meta_key = f"{self.prefix}meta:{version_id}"
        metadata["version_id"] = version_id
        metadata["created_at"] = datetime.now().isoformat()
        self.client.set(meta_key, json.dumps(metadata))

        self.client.set(f"{self.prefix}latest", version_id)
        return version_id

    def load_vocab(
        self, version_id: str | None = None
    ) -> tuple[dict[int, bytes], list[tuple[int, int, int]], dict[str, Any]]:
        if version_id is None:
            version_id = self.client.get(f"{self.prefix}latest")
            if version_id:
                version_id = version_id.decode()
            else:
                raise ValueError("No vocabulary found in Redis")

        vocab_data = self.client.hgetall(f"{self.prefix}vocab:{version_id}")
        vocab = {int(k): bytes.fromhex(v.decode()) for k, v in vocab_data.items()}

        rules_data = self.client.get(f"{self.prefix}rules:{version_id}")
        merge_rules = json.loads(rules_data) if rules_data else []

        meta_data = self.client.get(f"{self.prefix}meta:{version_id}")
        metadata = json.loads(meta_data) if meta_data else {}

        return vocab, merge_rules, metadata

    def save_special_tokens(self, special_tokens: dict[str, np.ndarray]) -> None:
        key = f"{self.prefix}special_tokens"
        data = {name: coords.tolist() for name, coords in special_tokens.items()}
        self.client.set(key, json.dumps(data))

    def load_special_tokens(self) -> dict[str, np.ndarray]:
        key = f"{self.prefix}special_tokens"
        data = self.client.get(key)
        if data:
            parsed = json.loads(data)
            return {name: np.array(coords) for name, coords in parsed.items()}
        return {}


class PostgresStorage(TokenizerStorage):
    """PostgreSQL-based storage for persistent vocab with versioning."""

    def __init__(self, database_url: str | None = None):
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 package required: pip install psycopg2-binary")

        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL not set")

        self.conn = psycopg2.connect(self.database_url)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qig_tokenizer_versions (
                    version_id VARCHAR(32) PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT NOW(),
                    vocab_size INTEGER,
                    metadata JSONB
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qig_tokenizer_vocab (
                    version_id VARCHAR(32) REFERENCES qig_tokenizer_versions(version_id),
                    token_id INTEGER,
                    byte_sequence BYTEA,
                    PRIMARY KEY (version_id, token_id)
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qig_tokenizer_merge_rules (
                    version_id VARCHAR(32) REFERENCES qig_tokenizer_versions(version_id),
                    rule_order INTEGER,
                    token_a INTEGER,
                    token_b INTEGER,
                    new_token INTEGER,
                    PRIMARY KEY (version_id, rule_order)
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS qig_tokenizer_special_tokens (
                    name VARCHAR(32) PRIMARY KEY,
                    basin_coords FLOAT8[],
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """
            )
            self.conn.commit()

    def save_vocab(
        self,
        vocab: dict[int, bytes],
        merge_rules: list[tuple[int, int, int]],
        metadata: dict[str, Any],
    ) -> str:
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_size = 1000  # Insert in batches for performance

        with self.conn.cursor() as cur:
            # Insert version
            cur.execute(
                "INSERT INTO qig_tokenizer_versions (version_id, vocab_size, metadata) VALUES (%s, %s, %s)",
                (version_id, len(vocab), json.dumps(metadata)),
            )
            print(
                f"  Inserting {len(vocab)} vocab entries in batches of {batch_size}..."
            )

            # Batch insert vocab
            vocab_data = [
                (version_id, token_id, bytes(byte_seq))
                for token_id, byte_seq in vocab.items()
            ]
            for i in range(0, len(vocab_data), batch_size):
                batch = vocab_data[i : i + batch_size]
                cur.executemany(
                    "INSERT INTO qig_tokenizer_vocab (version_id, token_id, byte_sequence) VALUES (%s, %s, %s)",
                    batch,
                )
                self.conn.commit()  # Commit each batch
                print(
                    f"    Vocab batch {i // batch_size + 1}/{(len(vocab_data) + batch_size - 1) // batch_size} committed"
                )

            print(
                f"  Inserting {len(merge_rules)} merge rules in batches of {batch_size}..."
            )

            # Batch insert merge rules
            merge_data = [
                (version_id, i, a, b, new) for i, (a, b, new) in enumerate(merge_rules)
            ]
            for i in range(0, len(merge_data), batch_size):
                batch = merge_data[i : i + batch_size]
                cur.executemany(
                    "INSERT INTO qig_tokenizer_merge_rules (version_id, rule_order, token_a, token_b, new_token) VALUES (%s, %s, %s, %s, %s)",
                    batch,
                )
                self.conn.commit()  # Commit each batch
                print(
                    f"    Merge batch {i // batch_size + 1}/{(len(merge_data) + batch_size - 1) // batch_size} committed"
                )

        return version_id

    def load_vocab(
        self, version_id: str | None = None
    ) -> tuple[dict[int, bytes], list[tuple[int, int, int]], dict[str, Any]]:
        with self.conn.cursor() as cur:
            if version_id is None:
                cur.execute(
                    "SELECT version_id FROM qig_tokenizer_versions ORDER BY created_at DESC LIMIT 1"
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("No vocabulary found in database")
                version_id = row[0]

            cur.execute(
                "SELECT token_id, byte_sequence FROM qig_tokenizer_vocab WHERE version_id = %s",
                (version_id,),
            )
            vocab = {row[0]: bytes(row[1]) for row in cur.fetchall()}

            cur.execute(
                "SELECT token_a, token_b, new_token FROM qig_tokenizer_merge_rules WHERE version_id = %s ORDER BY rule_order",
                (version_id,),
            )
            merge_rules = [(row[0], row[1], row[2]) for row in cur.fetchall()]

            cur.execute(
                "SELECT metadata FROM qig_tokenizer_versions WHERE version_id = %s",
                (version_id,),
            )
            row = cur.fetchone()
            metadata = row[0] if row else {}

        return vocab, merge_rules, metadata

    def save_special_tokens(self, special_tokens: dict[str, np.ndarray]) -> None:
        with self.conn.cursor() as cur:
            for name, coords in special_tokens.items():
                cur.execute(
                    "INSERT INTO qig_tokenizer_special_tokens (name, basin_coords) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET basin_coords = %s, updated_at = NOW()",
                    (name, coords.tolist(), coords.tolist()),
                )
            self.conn.commit()

    def load_special_tokens(self) -> dict[str, np.ndarray]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT name, basin_coords FROM qig_tokenizer_special_tokens")
            return {row[0]: np.array(row[1]) for row in cur.fetchall()}

    def __del__(self):
        if hasattr(self, "conn"):
            self.conn.close()


class HybridStorage(TokenizerStorage):
    """Hybrid storage: Redis for speed, PostgreSQL for persistence."""

    def __init__(self, redis_url: str | None = None, database_url: str | None = None):
        self.redis = RedisStorage(redis_url)
        self.postgres = PostgresStorage(database_url)

    def save_vocab(
        self,
        vocab: dict[int, bytes],
        merge_rules: list[tuple[int, int, int]],
        metadata: dict[str, Any],
    ) -> str:
        version_id = self.postgres.save_vocab(vocab, merge_rules, metadata)
        self.redis.save_vocab(vocab, merge_rules, metadata)
        return version_id

    def load_vocab(
        self, version_id: str | None = None
    ) -> tuple[dict[int, bytes], list[tuple[int, int, int]], dict[str, Any]]:
        try:
            return self.redis.load_vocab(version_id)
        except (ValueError, Exception):
            result = self.postgres.load_vocab(version_id)
            self.redis.save_vocab(*result)
            return result

    def save_special_tokens(self, special_tokens: dict[str, np.ndarray]) -> None:
        self.postgres.save_special_tokens(special_tokens)
        self.redis.save_special_tokens(special_tokens)

    def load_special_tokens(self) -> dict[str, np.ndarray]:
        tokens = self.redis.load_special_tokens()
        if not tokens:
            tokens = self.postgres.load_special_tokens()
            if tokens:
                self.redis.save_special_tokens(tokens)
        return tokens
