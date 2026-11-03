"""
version.py
The CHORD version, shown in the app and written into every report and Methods paragraph, so that a report
says which code made it (2 November 2026: a user's report could not be tied to a version).

VERSION is raised with every change pushed to GitHub. The git commit is read from the .git folder when the
code is a git clone (also inside Docker, which copies the folder into the image), with the time the code on
that computer last changed through git (clone, pull or commit). A ZIP download has no .git folder; then the
report gives VERSION only.
"""
from datetime import datetime, timezone
from pathlib import Path

VERSION = "2026.11.02"
ROOT = Path(__file__).resolve().parents[1]


def _git_commit(root):
    """The commit the code is at, read from .git without calling git (git is not in the Docker image)."""
    g = Path(root) / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1].strip()
        if (g / ref).exists():
            return (g / ref).read_text().strip()
        for line in (g / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def _git_updated(root):
    """When the code on this computer last changed through git: the time of the last line of .git/logs/HEAD."""
    try:
        last = (Path(root) / ".git" / "logs" / "HEAD").read_text().strip().splitlines()[-1]
        stamp = int(last.split("\t")[0].split()[-2])
        t = datetime.fromtimestamp(stamp, tz=timezone.utc)     # Docker runs on UTC; say so, never a bare time
        return f"{t.day} {t:%B %Y, %H:%M} UTC"
    except (OSError, IndexError, ValueError):
        return None


def describe(root=ROOT):
    """{"version", "commit" (7 characters or None), "updated" (or None)}."""
    sha = _git_commit(root)
    return {"version": VERSION, "commit": sha[:7] if sha else None, "updated": _git_updated(root)}


def version_text(root=ROOT, with_update=False):
    """For example 'CHORD 2026.11.02 (commit 1a2b3c4)', or 'CHORD 2026.11.02' for a ZIP download."""
    d = describe(root)
    extra = [f"commit {d['commit']}"] if d["commit"] else []
    if with_update and d["updated"]:
        extra.append(f"code updated on this computer {d['updated']}")
    return f"CHORD {d['version']}" + (f" ({'; '.join(extra)})" if extra else "")
