from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # DB금융공모전
sys.path.insert(0, str(PROJECT_ROOT))
import DB.db_conn
