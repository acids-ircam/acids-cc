#!/usr/bin/env bash
# Run the test suite over a matrix of Python x torch versions using uv.
#
# Usage:
#   ./test_versions.sh                          # Python 3.9 -> 3.14, latest torch
#   ./test_versions.sh -p "3.10 3.12"           # selected Python versions
#   ./test_versions.sh -p 3.10 -t "2.0 2.2 2.4" # selected torch versions
#   ./test_versions.sh -p 3.10 --lowest         # lowest torch uv can install
#   ./test_versions.sh --tmp-cache ...          # throwaway uv cache per run
#                                               # (saves disk, re-downloads)
#
# Torch specs (-t):
#   latest      -> newest compatible torch
#   lowest      -> oldest torch with a compatible wheel (uv --resolution lowest)
#   2.1         -> torch==2.1.*
#   2.1.2       -> torch==2.1.2
#   ">=2.3"     -> passed through as-is
#
# Combinations that cannot be installed (no wheel for that Python/platform)
# are reported as N/A instead of FAIL.
set -u
cd "$(dirname "$0")"

PYTHONS=(3.9 3.10 3.11 3.12 3.13 3.14)
TORCHES=(latest)
TMP_CACHE=0
OWN_CACHE=""  # only ever delete a cache dir we created

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--python) read -r -a PYTHONS <<< "$2"; shift 2 ;;
    -t|--torch)  read -r -a TORCHES <<< "$2"; shift 2 ;;
    --lowest)    TORCHES=(lowest); shift ;;
    --tmp-cache) TMP_CACHE=1; shift ;;
    -h|--help)   sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

torch_req() {
  case "$1" in
    latest|lowest) echo "torch" ;;
    [0-9]*.[0-9]*.[0-9]*) echo "torch==$1" ;;
    [0-9]*) echo "torch==$1.*" ;;
    *) echo "torch$1" ;;
  esac
}

declare -a RESULTS
status=0
for py in "${PYTHONS[@]}"; do
  for t in "${TORCHES[@]}"; do
    if [ "$TMP_CACHE" = 1 ]; then
      [ -n "$OWN_CACHE" ] && rm -rf "$OWN_CACHE"
      OWN_CACHE=$(mktemp -d -t uv-cache)
      export UV_CACHE_DIR="$OWN_CACHE"
    fi
    req=$(torch_req "$t")
    extra=()
    if [ "$t" = lowest ]; then
      # resolve the lowest torch alone (pytest & co. must stay recent)
      req=$(echo torch | uv pip compile - -q --no-header --python-version "$py" \
              --resolution lowest-direct 2>/dev/null | grep '^torch==')
      [ -z "$req" ] && { RESULTS+=("py $py | lowest torch: N/A (no resolution)"); continue; }
    fi
    # --no-project: tests import cached_conv via conftest's sys.path tweak.
    # --isolated: fresh throwaway env per combination.
    UV=(uv run --isolated --no-project --python "$py" ${extra[@]+"${extra[@]}"}
        --with "$req" --with pytest)

    echo "=== Python $py | $req ${extra[*]+${extra[*]}} ==="
    if ! info=$("${UV[@]}" python -c \
        "import sys, torch; print(sys.version.split()[0], torch.__version__)" 2>&1); then
      echo "$info" | tail -5
      RESULTS+=("py $py | $req: N/A (could not install)")
      continue
    fi
    info=$(echo "$info" | tail -1)
    echo "resolved: python ${info% *} | torch ${info#* }"
    if "${UV[@]}" pytest -q -x -p no:cacheprovider tests; then
      RESULTS+=("py ${info% *} | torch ${info#* }: PASS")
    else
      RESULTS+=("py ${info% *} | torch ${info#* }: FAIL")
      status=1
    fi
  done
done

[ -n "$OWN_CACHE" ] && rm -rf "$OWN_CACHE"

echo; echo "=== Summary ==="
printf "%s\n" ${RESULTS[@]+"${RESULTS[@]}"}
exit $status
