#!/usr/bin/env bash
# ===========================================================================
# beat-books-model -- Local Test Script
# Run from the repo root:  bash test.sh
# Or test a single module:  bash test.sh features
#                           bash test.sh models
#                           bash test.sh backtesting
# ===========================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

pass() { echo -e "  ${GREEN}PASS: $1${NC}"; }
fail() { echo -e "  ${RED}FAIL: $1${NC}"; FAILURES=$((FAILURES + 1)); }
skip() { echo -e "  ${YELLOW}SKIP: $1${NC}"; }
info() { echo -e "${CYAN}$1${NC}"; }
header() { echo -e "\n${BOLD}${YELLOW}=== $1 ===${NC}\n"; }

FAILURES=0
MODULE="${1:-all}"   # optional arg: features | models | backtesting | strategy | all

# ---------------------------------------------------------------------------
# Helper: run pytest on a test dir, handle "0 collected" gracefully
# ---------------------------------------------------------------------------
run_tests() {
    local dir="$1"
    local label="$2"

    if [ ! -d "$dir" ]; then
        skip "Skipped -- $dir not found on this branch"
        return
    fi

    local count
    count=$(find "$dir" -name 'test_*.py' -exec grep -l 'def test_' {} + 2>/dev/null | wc -l)
    if [ "$count" -eq 0 ]; then
        skip "Skipped -- no test functions in $dir"
        return
    fi

    if pytest "$dir" -v --tb=short -q; then
        pass "$label -- all tests passed"
    else
        fail "$label -- test failures"
    fi
}

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
header "Environment Setup"

export DATABASE_URL="${DATABASE_URL:-postgresql://test:test@localhost:5432/test}"
info "DATABASE_URL set (dummy -- no real DB needed for unit tests)"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}ERROR: Run this from the repo root (where requirements.txt is)${NC}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 1. Install deps
# ---------------------------------------------------------------------------
header "Installing Dependencies"

PIP_FLAGS=""
if [ -z "${VIRTUAL_ENV:-}" ]; then
    PIP_FLAGS="--break-system-packages"
fi

pip install $PIP_FLAGS -q -r requirements.txt 2>&1 | tail -1 || true
pip install $PIP_FLAGS -q ruff black mypy 2>&1 | tail -1 || true
pass "Dependencies installed"

# ---------------------------------------------------------------------------
# 2. Lint
# ---------------------------------------------------------------------------
if [ "$MODULE" = "all" ]; then
    header "Lint (ruff)"
    if ruff check src/; then
        pass "ruff -- no lint errors"
    else
        fail "ruff -- lint errors found"
    fi

    header "Formatting (black)"
    if black --check src/ 2>/dev/null; then
        pass "black -- all files formatted correctly"
    else
        fail "black -- formatting issues (run: black src/)"
    fi

    header "Type Check (mypy)"
    if mypy src/ --ignore-missing-imports 2>/dev/null; then
        pass "mypy -- no type errors"
    else
        fail "mypy -- type errors found"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Tests
# ---------------------------------------------------------------------------
if [ "$MODULE" = "all" ]; then
    header "Full Test Suite"
    echo ""
    if pytest tests/ \
        -v \
        --tb=short \
        --cov=src \
        --cov-report=term-missing \
        --cov-report=html:htmlcov \
        -q; then
        echo ""
        pass "All tests passed"
        info "  Coverage HTML: open htmlcov/index.html"
    else
        echo ""
        fail "Some tests failed"
    fi

    header "Feature Engineering Tests"
    run_tests "tests/test_features" "Feature engineering"

    header "Model Tests"
    run_tests "tests/test_models" "Models"

    header "Backtesting Tests"
    run_tests "tests/test_backtesting" "Backtesting"

    header "Strategy Tests"
    run_tests "tests/test_strategy" "Strategy"
else
    header "Testing: $MODULE"
    run_tests "tests/test_${MODULE}" "$MODULE"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Summary"

if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  All checks passed! Ready for PR.${NC}"
    echo ""
    exit 0
else
    echo -e "${RED}${BOLD}  $FAILURES check(s) failed. Fix issues above.${NC}"
    echo ""
    exit 1
fi
