#!/bin/bash
# Sync the harness to the remote box. Excludes anything large or environment-specific.
set -e
REMOTE=ubuntu@129.146.71.64
KEY=~/.ssh/id_ed25519
rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude 'runs' --exclude '*.eval' \
  --exclude '.venv' --exclude 'notes' \
  -e "ssh -i $KEY -o BatchMode=yes" \
  ./ "$REMOTE:~/ac/"
echo "synced -> $REMOTE:~/ac/"
