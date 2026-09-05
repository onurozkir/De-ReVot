"""SQLite DDL schema and initialization for meeting persistence."""

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS meetings (
    id                   TEXT PRIMARY KEY,
    title                TEXT,
    started_at           TEXT NOT NULL,
    ended_at             TEXT,
    status               TEXT NOT NULL CHECK (status IN ('starting','running','stopped','error')),
    save_enabled         INTEGER NOT NULL DEFAULT 1 CHECK (save_enabled IN (0,1)),
    asr_backend          TEXT NOT NULL,
    asr_model            TEXT NOT NULL,
    mt_tr_en_backend     TEXT NOT NULL,
    mt_tr_en_model       TEXT NOT NULL,
    mt_en_tr_backend     TEXT NOT NULL,
    mt_en_tr_model       TEXT NOT NULL,
    tts_backend          TEXT NOT NULL,
    tts_model            TEXT NOT NULL,
    voice_profile_id     TEXT,
    config_snapshot_json TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS utterances (
    id                TEXT PRIMARY KEY,
    meeting_id        TEXT NOT NULL,
    direction         TEXT NOT NULL CHECK (direction IN ('outgoing','incoming')),
    sequence          INTEGER NOT NULL CHECK (sequence >= 0),
    source_language   TEXT NOT NULL,
    text              TEXT NOT NULL,
    state             TEXT NOT NULL CHECK (state IN ('partial','committed')),
    started_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    committed_at      TEXT,
    audio_path        TEXT,
    audio_metadata_json TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (meeting_id, direction, sequence)
);

CREATE TABLE IF NOT EXISTS translations (
    id                TEXT PRIMARY KEY,
    utterance_id      TEXT NOT NULL,
    target_language   TEXT NOT NULL,
    text              TEXT NOT NULL,
    state             TEXT NOT NULL CHECK (state IN ('partial','committed')),
    backend           TEXT NOT NULL,
    model             TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    committed_at      TEXT,
    FOREIGN KEY (utterance_id) REFERENCES utterances(id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (utterance_id, target_language)
);

CREATE TABLE IF NOT EXISTS latency_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id        TEXT NOT NULL,
    utterance_id      TEXT,
    direction         TEXT CHECK (direction IN ('outgoing','incoming')),
    event_type        TEXT NOT NULL,
    occurred_at       TEXT NOT NULL,
    monotonic_ns      INTEGER NOT NULL,
    duration_ms       REAL,
    queue_age_ms      REAL,
    metadata_json     TEXT,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS voice_profiles (
    id                       TEXT PRIMARY KEY,
    display_name             TEXT NOT NULL UNIQUE,
    backend                  TEXT NOT NULL,
    reference_audio_path     TEXT NOT NULL,
    reference_text           TEXT,
    reference_language       TEXT NOT NULL,
    target_language          TEXT NOT NULL,
    is_default               INTEGER NOT NULL DEFAULT 0 CHECK (is_default IN (0,1)),
    conditioning_cache_path  TEXT,
    metadata_json            TEXT NOT NULL DEFAULT '{}',
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key          TEXT PRIMARY KEY,
    value_json   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meeting_statistics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id   TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    aggregation  TEXT NOT NULL,
    value        REAL NOT NULL,
    unit         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (meeting_id, metric_name, aggregation)
);

CREATE INDEX IF NOT EXISTS idx_meetings_started_at ON meetings(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_utterances_meeting_order ON utterances(meeting_id, direction, sequence);
CREATE INDEX IF NOT EXISTS idx_latency_meeting_time ON latency_events(meeting_id, monotonic_ns);
"""


def initialize_database(db_path: str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path.resolve()), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn

