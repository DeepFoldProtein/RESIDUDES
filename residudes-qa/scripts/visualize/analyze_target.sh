#!/bin/bash
# analyze_target.sh — kept for the CASP17 command line; now runs residudes-qa's venv instead of the af3-viz container.
# Same options as before (--dpi, --no-plots/--no-msa/--no-csv, -v). See: residudes-qa visualize --help
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/bin/residudes-qa" visualize "$@"
