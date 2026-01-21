import sys
from pathlib import Path

# Add parent directory to path so we can import rag module
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from rag.db import get_engine
from rag.diagnostics import check_postgres_status

engine = get_engine()
st = check_postgres_status(engine)
print(st)
