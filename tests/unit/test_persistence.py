import os
import tempfile
import pytest
from voice_translator.config.models import PersistenceConfig
from voice_translator.core.types import Direction, UtteranceEvent, UtteranceState
from voice_translator.persistence.database import PersistenceWorker
from voice_translator.persistence.schema import initialize_database


def test_schema_initialization():
    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as f:
        temp_db = f.name

    try:
        conn = initialize_database(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "meetings" in tables
        assert "utterances" in tables
        assert "translations" in tables
        assert "latency_events" in tables
        conn.close()
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)

