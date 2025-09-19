#!/usr/bin/env bash
set -e

# lhotse
source /workspaces/dev/.venv/bin/activate
lhotse install-sph2pipe # for tedlium dataset
