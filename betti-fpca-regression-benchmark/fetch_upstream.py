"""Clone upstream into ``upstream/`` at the pinned commit, or check the clone that is there.

Upstream (``design.md``) is the public code and data of Aggarwal, Nichols and
Turner. It is data only: this script runs git on it and never imports,
executes or puts on ``sys.path`` anything inside it. From this folder:

    $PY fetch_upstream.py [--source PATH_OR_URL]

``--source`` defaults to ``UPSTREAM_URL``. A local clone can stand in for it;
the new clone's ``origin`` then names ``UPSTREAM_URL`` all the same. Writes
``results/upstream.json``: the SHA, the source, whether this run cloned (if
``upstream/`` was there already, the run only checked it and used no
source), the date and ``provenance``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Sequence

from common import HERE, RESULTS, UPSTREAM, UPSTREAM_SHA, UPSTREAM_URL, _git, check_topocyte, provenance


def ensure_upstream(source: str = UPSTREAM_URL) -> str:
    """The pinned clone's SHA, cloned from ``source`` if ``UPSTREAM`` is absent.

    A new clone is checked out at ``UPSTREAM_SHA``, detached, and its
    ``origin`` set to ``UPSTREAM_URL``. ``RuntimeError`` unless ``UPSTREAM``
    is then a git clone whose ``HEAD`` is ``UPSTREAM_SHA``, with a clean tree.
    """
    if not UPSTREAM.exists():
        subprocess.run(["git", "clone", "--no-checkout", source, str(UPSTREAM)], check=True)
        _git(UPSTREAM, "checkout", "--quiet", "--detach", UPSTREAM_SHA)
        _git(UPSTREAM, "remote", "set-url", "origin", UPSTREAM_URL)
    if not (UPSTREAM / ".git").exists():  # else git would read the repository around it
        raise RuntimeError(f"{UPSTREAM} is not a git clone; move it aside, then fetch again")
    head = _git(UPSTREAM, "rev-parse", "HEAD")
    changed = _git(UPSTREAM, "status", "--porcelain")
    if head != UPSTREAM_SHA or changed:
        raise RuntimeError(
            f"upstream clone {UPSTREAM} is at {head[:7]} (tree changed: {bool(changed)}), not a clean {UPSTREAM_SHA}"
        )
    return head


def main(argv: Sequence[str] | None = None) -> None:
    check_topocyte()
    parser = argparse.ArgumentParser(description="Clone upstream into upstream/ at the pinned commit, or check it.")
    parser.add_argument(
        "--source", default=UPSTREAM_URL, metavar="PATH_OR_URL", help="where to clone from (default: %(default)s)"
    )
    args = parser.parse_args(argv)
    cloned = not UPSTREAM.exists()
    sha = ensure_upstream(args.source)
    record = {
        "description": "upstream, the paper's code and data, cloned into upstream/ at the pinned commit (data only)",
        "sha": sha,
        "source": args.source,
        "cloned": cloned,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provenance": provenance(HERE / "fetch_upstream.py", HERE / "common.py"),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "upstream.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"upstream/ is a clean clone at {sha}" + (f", cloned from {args.source}" if cloned else ", already there"))


if __name__ == "__main__":
    main()
