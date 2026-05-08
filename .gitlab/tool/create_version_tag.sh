#!/bin/bash
if git ls-remote --tags origin "refs/tags/v${VERSION}" | grep -q "refs/tags/v${VERSION}"; then
    echo "Tag v${VERSION} already exists, skipping..."
else
    echo "Creating new tag v${VERSION}..."
    git config user.email "gitlab-ci@espressif.com"
    git config user.name "GitLab CI"
    git tag -a "v${VERSION}" -m "Release version v${VERSION}"
    git push origin "v${VERSION}"

    # Trigger a new pipeline for the newly created tag
    # This is needed because tags created by CI jobs don't automatically trigger pipelines
    if [ -n "${CI_TRIGGER_TOKEN}" ]; then
        echo "Triggering pipeline for tag v${VERSION}..."
        curl -X POST \
            -F token="${CI_TRIGGER_TOKEN}" \
            -F ref="v${VERSION}" \
            "${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/trigger/pipeline"
    else
        echo "Warning: CI_TRIGGER_TOKEN not set, pipeline for tag v${VERSION} needs to be triggered manually"
    fi
fi
