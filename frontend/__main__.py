import sys
from pathlib import Path

if __package__:
    from .app import main
else:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from frontend.app import main

raise SystemExit(main())
