#!/usr/bin/env bash
# Apply the canonical GitHub repository description, homepage and topics.
#
# Repository metadata lives on GitHub, not in the working tree, so it cannot be
# committed. This script applies it reproducibly with the `gh` CLI.
#
# Requires: gh (https://cli.github.com/), authenticated with `gh auth login`.
#
#   bash scripts/apply_github_metadata.sh            # apply to RxWhizz/Corpus
#   REPO=owner/name bash scripts/apply_github_metadata.sh
#   DRY_RUN=1 bash scripts/apply_github_metadata.sh  # print without applying

set -euo pipefail

REPO="${REPO:-RxWhizz/Corpus}"

DESCRIPTION="TEM nanoparticle metrology, dataset curation and instance-segmentation toolkit built with Electron, OpenCV and Python."
HOMEPAGE="https://github.com/RxWhizz/Corpus#readme"
TOPICS=(
  electron-microscopy
  tem
  nanoparticles
  computer-vision
  image-analysis
  materials-science
  opencv
  instance-segmentation
  scientific-software
  yolo
)

echo "Repository:  $REPO"
echo "Description: $DESCRIPTION"
echo "Homepage:    $HOMEPAGE"
echo "Topics:      ${TOPICS[*]}"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo
  echo "DRY_RUN=1 set; nothing was applied."
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "error: the GitHub CLI (gh) is required. See https://cli.github.com/" >&2
  exit 1
fi

gh repo edit "$REPO" \
  --description "$DESCRIPTION" \
  --homepage "$HOMEPAGE"

# `gh repo edit --add-topic` is additive; replace the set so removed topics go away.
TOPIC_JSON=$(printf '%s\n' "${TOPICS[@]}" | jq -R . | jq -s '{names: .}')
gh api -X PUT "repos/$REPO/topics" \
  -H "Accept: application/vnd.github+json" \
  --input - <<<"$TOPIC_JSON" >/dev/null

echo
echo "Applied. Verify with: gh repo view $REPO"
