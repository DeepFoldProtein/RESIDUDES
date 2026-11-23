#!/usr/bin/env python3
"""Every relative link in the repository's Markdown resolves to a file (stdlib only; run in CI).

    python3 docs/check_links.py          # exit 1 and list the broken ones

Skipped: fenced code blocks, inline code that is not a link's text, URLs, anchors, e-mail addresses, and the upstream models' own READMEs
vendored under lantern-engine/core/models/ (kept as their authors wrote them).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = ("lantern-engine/core/models/",)


def broken() -> list[tuple[str, str]]:
    files = subprocess.run(["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    out = []
    for f in files:
        if f.startswith(SKIP):
            continue
        text = re.sub(r"```.*?```", "", (ROOT / f).read_text(errors="ignore"), flags=re.S)
        text = re.sub(r"(?<!\[)`[^`\n]*`(?!\])", "", text)      # inline code that is not a link's text (CLI syntax)
        for m in re.finditer(r"\]\(([^)\s]+)\)", text):
            target = m.group(1)
            path = target.split("#")[0]
            if not path or re.match(r"^(https?:|mailto:)", target) or "@" in path:
                continue
            if not ((ROOT / f).parent / path).exists():
                out.append((f, target))
    return out


if __name__ == "__main__":
    bad = broken()
    for f, t in bad:
        print(f"{f}: {t}")
    print(f"{len(bad)} broken link(s)")
    sys.exit(1 if bad else 0)
