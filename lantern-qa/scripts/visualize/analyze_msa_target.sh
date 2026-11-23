#!/bin/bash
# analyze_msa_target.sh — kept for the CASP17 command line; now runs lantern-qa's venv instead of the af3-viz container.
# Same options as before (--dpi, --no-plots/--no-msa/--no-csv, -v). See: lantern-qa visualize-msa --help
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/bin/lantern-qa" visualize-msa "$@"
