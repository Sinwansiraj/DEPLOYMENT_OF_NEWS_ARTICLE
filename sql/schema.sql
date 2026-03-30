-- ============================================================
-- News Categorizer — PostgreSQL Schema
-- Run once on your RDS instance to initialize the database.
-- ============================================================

-- Create the database (run as superuser)
-- CREATE DATABASE news_categorizer;

-- Connect to the database before running the rest:
-- \c news_categorizer

-- Predictions table: stores every inference request
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    input_text      TEXT                     NOT NULL,
    predicted_label VARCHAR(64)              NOT NULL,
    confidence      FLOAT                    NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    all_scores      JSONB,
    model_version   VARCHAR(128)             DEFAULT 'v1.0',
    created_at      TIMESTAMPTZ              NOT NULL DEFAULT NOW()
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_predictions_created_at
    ON predictions (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_label
    ON predictions (predicted_label);

CREATE INDEX IF NOT EXISTS idx_predictions_confidence
    ON predictions (confidence DESC);

-- Stats view: category distribution
CREATE OR REPLACE VIEW category_stats AS
SELECT
    predicted_label,
    COUNT(*)                         AS total_predictions,
    ROUND(AVG(confidence)::NUMERIC, 4) AS avg_confidence,
    MAX(created_at)                  AS last_seen
FROM predictions
GROUP BY predicted_label
ORDER BY total_predictions DESC;

-- Daily volume view
CREATE OR REPLACE VIEW daily_prediction_volume AS
SELECT
    DATE(created_at AT TIME ZONE 'UTC') AS prediction_date,
    COUNT(*)                             AS total,
    COUNT(DISTINCT predicted_label)      AS unique_categories
FROM predictions
GROUP BY prediction_date
ORDER BY prediction_date DESC;