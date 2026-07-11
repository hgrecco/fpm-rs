"""Build and serve the exact static documentation deployed to Pages."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, "scripts/build_docs.py"], cwd=ROOT, check=True)
    handler = partial(SimpleHTTPRequestHandler, directory=ROOT / "site")
    server = ThreadingHTTPServer(("127.0.0.1", 8000), handler)
    print("Previewing the complete site at http://127.0.0.1:8000/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
