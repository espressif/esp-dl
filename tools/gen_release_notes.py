#!/usr/bin/env python3
"""Generate English GitHub release notes from git history.

Template sections: ## Improvements, ## Bugfix, ## Full Changelog.

Example:
    python tools/gen_release_notes.py
    python tools/gen_release_notes.py --from v3.3.7 --to HEAD
    python tools/gen_release_notes.py --to v3.3.7 --out /tmp/notes.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = Path(__file__).resolve().parent / "release_notes.template.md"
DEFAULT_REPO_URL = "https://github.com/espressif/esp-dl"

# Conventional-commit types that become user-facing bullets.
IMPROVEMENT_TYPES = {
    "feat",
    "feature",
    "enhance",
    "enhancement",
    "perf",
    "refactor",
    "add",
    "support",
    "improve",
    "update",
}
BUGFIX_TYPES = {"fix", "bugfix", "bug", "hotfix"}
SKIP_TYPES = {
    "ci",
    "chore",
    "docs",
    "doc",
    "test",
    "tests",
    "release",
    "build",
    "style",
    "revert",
    "merge",
}

CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[A-Za-z]+)(?P<scope>\([^)]*\))?(?P<breaking>!)?:\s*(?P<summary>.+)$"
)
MERGE_RE = re.compile(r"^Merge (branch|remote-tracking branch|pull request)\b", re.I)
DUP_PREFIX_RE = re.compile(r"^(?:feat|fix|feature|bugfix):\s*", re.I)


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str
    author_name: str
    author_email: str


class GitError(RuntimeError):
    pass


def git(args: list[str], cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def resolve_ref(ref: str) -> str:
    return git(["rev-parse", "--verify", f"{ref}^{{commit}}"])


def ref_exists(ref: str) -> bool:
    try:
        resolve_ref(ref)
    except GitError:
        return False
    return True


def known_tags(pattern: str = "v*", limit: int = 10) -> list[str]:
    try:
        raw = git(["tag", "--list", pattern, "--sort=-version:refname"])
    except GitError:
        return []
    return [tag for tag in raw.splitlines() if tag.strip()][:limit]


def previous_tag(to_ref: str, pattern: str = "v*") -> str | None:
    """Nearest version tag that is an ancestor of to_ref, excluding to_ref itself."""
    merged = git(
        ["tag", "--list", pattern, "--merged", to_ref, "--sort=-version:refname"]
    )
    if not merged:
        return None

    try:
        to_sha = resolve_ref(to_ref)
    except GitError:
        to_sha = ""

    for tag in merged.splitlines():
        tag = tag.strip()
        if not tag:
            continue
        try:
            if resolve_ref(tag) == to_sha:
                continue
        except GitError:
            continue
        return tag
    return None


def collect_commits(from_ref: str, to_ref: str) -> list[Commit]:
    fmt = "%H%x09%s%x09%an%x09%ae"
    raw = git(["log", "--no-merges", f"{from_ref}..{to_ref}", f"--pretty=format:{fmt}"])
    if not raw:
        return []

    commits: list[Commit] = []
    for line in raw.splitlines():
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        sha, subject, name, email = parts
        commits.append(
            Commit(
                sha=sha, subject=subject.strip(), author_name=name, author_email=email
            )
        )
    return commits


def author_handle(commit: Commit) -> str | None:
    email = commit.author_email.lower()
    local, _, domain = email.partition("@")
    if not local:
        return None
    if domain.endswith("users.noreply.github.com"):
        # 123456+login@users.noreply.github.com or login@users.noreply.github.com
        return local.split("+", 1)[-1]
    if local in {"github-actions", "noreply", "no-reply"}:
        return None
    if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", local):
        return local
    return None


def classify(subject: str) -> str | None:
    """Return 'improvements', 'bugfix', or None to skip."""
    text = subject.strip()
    if not text or MERGE_RE.match(text):
        return None

    match = CONVENTIONAL_RE.match(text)
    if match:
        ctype = match.group("type").lower()
        if ctype in SKIP_TYPES:
            return None
        if ctype in BUGFIX_TYPES:
            return "bugfix"
        if ctype in IMPROVEMENT_TYPES:
            return "improvements"
        return None

    lowered = text.lower()
    if lowered.startswith(("fix ", "bug", "hotfix")):
        return "bugfix"
    return "improvements"


def clean_summary(subject: str) -> str:
    text = subject.strip()
    match = CONVENTIONAL_RE.match(text)
    if match:
        text = match.group("summary").strip()
        # Drop duplicated type prefixes: "feat: feat: foo" / "feat: update ..."
        text = DUP_PREFIX_RE.sub("", text).strip()
    text = re.sub(r"\s+", " ", text).rstrip(".")
    if not text:
        return subject.strip()
    return text[0].upper() + text[1:]


def format_bullet(commit: Commit) -> str:
    summary = clean_summary(commit.subject)
    handle = author_handle(commit)
    if handle:
        return f"- {summary} by @{handle}"
    return f"- {summary}"


def dedupe_bullets(bullets: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for bullet in bullets:
        key = re.sub(r" by @\S+$", "", bullet).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(bullet)
    return out


def render_section(bullets: list[str]) -> str:
    if not bullets:
        return "- None."
    return "\n".join(bullets)


def compare_url(repo_url: str, from_ref: str, to_ref: str) -> str:
    return f"{repo_url.rstrip('/')}/compare/{from_ref}...{to_ref}"


def generate_notes(
    from_ref: str,
    to_ref: str,
    repo_url: str,
    template_path: Path,
) -> str:
    commits = collect_commits(from_ref, to_ref)
    improvements: list[str] = []
    bugfix: list[str] = []
    for commit in commits:
        bucket = classify(commit.subject)
        if bucket is None:
            continue
        bullet = format_bullet(commit)
        if bucket == "bugfix":
            bugfix.append(bullet)
        else:
            improvements.append(bullet)

    template = template_path.read_text(encoding="utf-8")
    return (
        template.format(
            improvements=render_section(dedupe_bullets(improvements)),
            bugfix=render_section(dedupe_bullets(bugfix)),
            compare_url=compare_url(repo_url, from_ref, to_ref),
        ).rstrip()
        + "\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from",
        dest="from_ref",
        default=None,
        help="Previous tag or commit (default: previous v* tag merged into --to)",
    )
    parser.add_argument(
        "--to",
        dest="to_ref",
        default="HEAD",
        help="Current tag or commit (default: HEAD)",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help=f"Remote used for the compare link (default: {DEFAULT_REPO_URL})",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_TEMPLATE,
        help="Markdown template path",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Write notes to this file (default: stdout)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    to_ref = args.to_ref
    if not ref_exists(to_ref):
        print(
            f"error: --to ref '{to_ref}' does not exist in this repository.",
            file=sys.stderr,
        )
        tags = known_tags()
        if tags:
            print(f"Known tags: {', '.join(tags)}", file=sys.stderr)
            print(
                "Run 'git fetch --tags' if the tag was created elsewhere.",
                file=sys.stderr,
            )
        return 1

    from_ref = args.from_ref or previous_tag(to_ref)
    if not from_ref:
        print(
            "Could not find a previous tag. Pass --from explicitly.",
            file=sys.stderr,
        )
        return 1
    if not ref_exists(from_ref):
        print(
            f"error: --from ref '{from_ref}' does not exist in this repository.",
            file=sys.stderr,
        )
        tags = known_tags()
        if tags:
            print(f"Known tags: {', '.join(tags)}", file=sys.stderr)
        return 1

    try:
        notes = generate_notes(from_ref, to_ref, args.repo_url, args.template)
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(notes, encoding="utf-8")
    else:
        sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
