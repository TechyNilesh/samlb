#!/usr/bin/env bash
#
# Publish wiki/*.md to the GitHub wiki.
#
# The wiki is a separate git repository. GitHub creates it only after the first
# page is saved through the web UI — if the clone below fails with "Repository
# not found", open https://github.com/TechyNilesh/samlb/wiki, click "Create the
# first page", save it, and re-run.
#
# Usage:  ./scripts/publish_wiki.sh [wiki-remote-url]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/wiki"
WIKI_URL="${1:-https://github.com/TechyNilesh/samlb.wiki.git}"

if [ ! -d "$SRC_DIR" ]; then
    echo "error: $SRC_DIR does not exist" >&2
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "==> cloning $WIKI_URL"
if ! git clone --quiet --depth 1 "$WIKI_URL" "$TMP_DIR/wiki"; then
    cat >&2 <<'MSG'
error: could not clone the wiki repository.

GitHub creates it only after the first page exists. Open the repository's Wiki
tab, click "Create the first page", save anything, then re-run this script.
MSG
    exit 1
fi

echo "==> copying pages"
# README.md documents this directory for repo readers; it is not a wiki page.
for page in "$SRC_DIR"/*.md; do
    [ "$(basename "$page")" = "README.md" ] && continue
    cp "$page" "$TMP_DIR/wiki/"
    echo "    $(basename "$page")"
done

cd "$TMP_DIR/wiki"
git add -A

if git diff --cached --quiet; then
    echo "==> wiki already up to date"
    exit 0
fi

git commit --quiet -m "Update wiki from repo wiki/ directory"
git push --quiet
echo "==> published: ${WIKI_URL%.wiki.git}/wiki"
