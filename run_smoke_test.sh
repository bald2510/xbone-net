#!/usr/bin/env bash
set -euo pipefail
python tools/smoke_test_all.py --config-root configs --manifest tools/smoke_manifest.yaml --device cuda --batch-size 2
