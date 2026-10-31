"""Run while the app/workers are stopped: python scripts/migrate_tool_queue.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings
from app.core.database import engine
from app.core.tool_schema import migrate_tool_schema

if __name__ == '__main__':
    migrate_tool_schema(engine, get_settings())
    print('Tool queue schema migrated. Historical records retained.')
