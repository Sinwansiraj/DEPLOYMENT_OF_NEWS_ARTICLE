"""
Database utility module for logging predictions to PostgreSQL RDS.
Manages connection pooling, table creation, and record insertion.
"""

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2 import pool, OperationalError, DatabaseError
from psycopg2.extras import RealDictCursor

from config import DB_CONFIG

logger = logging.getLogger(__name__)

_POOL: Optional[pool.ThreadedConnectionPool] = None


def init_db_pool(minconn: int = 1, maxconn: int = 10) -> None:
    """
    Initialize the PostgreSQL connection pool.
    Must be called once at application startup.

    Args:
        minconn: Minimum number of pooled connections.
        maxconn: Maximum number of pooled connections.
    """
    global _POOL
    try:
        _POOL = pool.ThreadedConnectionPool(
            minconn,
            maxconn,
            host=DB_CONFIG.host,
            port=DB_CONFIG.port,
            dbname=DB_CONFIG.name,
            user=DB_CONFIG.user,
            password=DB_CONFIG.password,
            connect_timeout=10,
        )
        logger.info("Database connection pool initialized.")
    except OperationalError as exc:
        logger.error("Failed to initialize DB pool: %s", exc)
        _POOL = None


@contextmanager
def get_db_connection():
    """
    Context manager that provides a connection from the pool.
    Automatically returns the connection to the pool on exit.

    Yields:
        psycopg2 connection object.

    Raises:
        RuntimeError: If pool is not initialized.
    """
    if _POOL is None:
        raise RuntimeError("DB pool not initialized. Call init_db_pool() first.")
    conn = _POOL.getconn()
    try:
        yield conn
        conn.commit()
    except DatabaseError as exc:
        conn.rollback()
        logger.error("Database error, transaction rolled back: %s", exc)
        raise
    finally:
        _POOL.putconn(conn)


def create_tables() -> None:
    """
    Create required database tables if they do not exist.
    Safe to call on every startup (idempotent).
    """
    ddl = """
        CREATE TABLE IF NOT EXISTS predictions (
            id              SERIAL PRIMARY KEY,
            input_text      TEXT NOT NULL,
            predicted_label VARCHAR(64) NOT NULL,
            confidence      FLOAT NOT NULL,
            all_scores      JSONB,
            model_version   VARCHAR(128),
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions (created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_predictions_label
            ON predictions (predicted_label);
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
        logger.info("Database tables verified / created.")
    except Exception as exc:
        logger.error("Failed to create tables: %s", exc)


def log_prediction(
    input_text: str,
    predicted_label: str,
    confidence: float,
    all_scores: dict,
    model_version: str = "v1.0",
) -> Optional[int]:
    """
    Insert a prediction record into the predictions table.

    Args:
        input_text: Raw article text submitted by user.
        predicted_label: Human-readable predicted category.
        confidence: Confidence score of the top prediction.
        all_scores: Full score dict for all categories.
        model_version: Identifier for the model used.

    Returns:
        Inserted row ID, or None on failure.
    """
    import json

    sql = """
        INSERT INTO predictions
            (input_text, predicted_label, confidence, all_scores, model_version)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        input_text[:4000],   # guard against oversized text
                        predicted_label,
                        confidence,
                        json.dumps(all_scores),
                        model_version,
                    ),
                )
                row_id = cur.fetchone()[0]
        logger.info("Prediction logged with ID=%d", row_id)
        return row_id
    except Exception as exc:
        logger.error("Failed to log prediction: %s", exc)
        return None


def fetch_recent_predictions(limit: int = 50) -> list:
    """
    Retrieve the most recent prediction records.

    Args:
        limit: Number of records to fetch.

    Returns:
        List of dicts representing prediction rows.
    """
    sql = """
        SELECT id, input_text, predicted_label, confidence, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT %s;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (limit,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        logger.error("Failed to fetch predictions: %s", exc)
        return []