#!/bin/bash
set -e

if git ls-remote --tags origin "refs/tags/v${VERSION}" | grep -q "refs/tags/v${VERSION}"; then
    echo "Tag v${VERSION} already exists, skipping..."
else
    echo "Creating new tag v${VERSION}..."
    git config user.email "gitlab-ci@espressif.com"
    git config user.name "GitLab CI"
    git tag -a "v${VERSION}" -m "Release version v${VERSION}"
    git push origin "v${VERSION}"

    # Also push the new tag to GitHub mirror.
    # The master/release branch push pipeline has already run the full CI for
    # this commit, so we deliberately do NOT trigger a new tag pipeline here.
    if [ -n "${GITHUB_PUSH_TOKEN}" ]; then
        echo "Pushing tag v${VERSION} to GitHub..."
        git remote remove github &>/dev/null || true
        git remote add github "https://${GITHUB_PUSH_TOKEN}@github.com/espressif/esp-dl.git"
        git push github "v${VERSION}"
    else
        echo "Warning: GITHUB_PUSH_TOKEN not set, tag v${VERSION} was not pushed to GitHub"
    fi
fi
