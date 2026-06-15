#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-spatial_env}"
SKIP_PIP=0
SKIP_VCPKG=0
SKIP_TESTS=0
CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-pip)    SKIP_PIP=1; shift ;;
        --skip-vcpkg)  SKIP_VCPKG=1; shift ;;
        --skip-tests)  SKIP_TESTS=1; shift ;;
        --clean)       CLEAN=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

VCPKG_ROOT="$PROJECT_ROOT/dependencies/vcpkg"
VCPKG_TRIPLET="x64-linux"
BUILD_DIR="$PROJECT_ROOT/build"

step() { echo -e "\n== $1 =="; }
fail() { echo "ERROR: $1" >&2; exit 1; }

conda_run() {
    conda run -n "$CONDA_ENV" "$@"
}

PYTHON_BIN="$(conda_run which python3 2>/dev/null || echo "")"
[[ -z "$PYTHON_BIN" ]] && fail "Python not found in conda env '$CONDA_ENV'"

echo "== Environment =="
echo "Project:   $PROJECT_ROOT"
echo "Conda env: $CONDA_ENV"
echo "Python:    $PYTHON_BIN"

if [[ "$SKIP_PIP" -eq 0 ]]; then
    step "Install Python dependencies"
    conda_run python -m pip install -r requirements.txt
fi

if [[ "$SKIP_VCPKG" -eq 0 ]]; then
    step "Prepare vcpkg"
    if [[ ! -d "$VCPKG_ROOT" ]]; then
        mkdir -p "$PROJECT_ROOT/dependencies"
        git clone https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT"
    fi

    if [[ ! -f "$VCPKG_ROOT/vcpkg" ]]; then
        "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
    fi

    step "Install PhysX"
    "$VCPKG_ROOT/vcpkg" install "physx:$VCPKG_TRIPLET"
fi

[[ -f "$VCPKG_ROOT/vcpkg" ]] || fail "vcpkg executable not found"
TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
[[ -f "$TOOLCHAIN_FILE" ]] || fail "vcpkg toolchain file not found"

if [[ "$CLEAN" -eq 1 ]] && [[ -d "$BUILD_DIR" ]]; then
    step "Clean build directory"
    rm -rf "$BUILD_DIR"
fi

CONDA_PREFIX="$(conda_run python -c "import sys; print(sys.prefix)")"

step "Configure CMake"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DPython_EXECUTABLE="$PYTHON_BIN" \
    -DPython_ROOT_DIR="$CONDA_PREFIX" \
    -DCONDA_ENV_PATH="$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release

step "Build"
cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

export LD_LIBRARY_PATH="$BUILD_DIR/lib:$VCPKG_ROOT/installed/$VCPKG_TRIPLET/lib:${LD_LIBRARY_PATH:-}"

step "Smoke import"
"$PYTHON_BIN" -c "
import pathlib, sys
sys.path.insert(0, str(pathlib.Path.cwd() / 'python'))
import spatial_db
print('HAS_NATIVE_MODULE =', spatial_db.HAS_NATIVE_MODULE)
print('native =', getattr(spatial_db.native, '__name__', type(spatial_db.native).__name__))
print('init_physx =', spatial_db.native.init_physx(0))
"

if [[ "$SKIP_TESTS" -eq 0 ]]; then
    step "Run tests"
    conda_run python -m pytest tests -q
fi

step "Done"
