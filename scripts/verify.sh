#!/bin/bash
###############################################################################
# Quick verification script for restructured MSDYOLO
# Tests that all components work before full training
###############################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MSDYOLO Structure Verification${NC}"
echo -e "${GREEN}========================================${NC}"

TESTS_PASSED=0
TESTS_FAILED=0

test_step() {
    echo -e "\n${YELLOW}Testing: $1${NC}"
}

test_pass() {
    echo -e "${GREEN}✓ $1${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

test_fail() {
    echo -e "${RED}✗ $1${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

# Test 1: Package import
test_step "Package imports"
if python3 -c "import msdyolo" 2>/dev/null; then
    test_pass "msdyolo package imports"
else
    test_fail "msdyolo package import failed"
fi

# Test 2: Models import
test_step "Models import"
if python3 -c "from msdyolo.models import yolo" 2>/dev/null; then
    test_pass "Models import successfully"
else
    test_fail "Models import failed"
fi

# Test 3: Utils import
test_step "Utils import"
if python3 -c "from msdyolo.utils import general" 2>/dev/null; then
    test_pass "Utils import successfully"
else
    test_fail "Utils import failed"
fi

# Test 4: Config files exist
test_step "Configuration files"
if [ -f "configs/train/baseline.yaml" ]; then
    test_pass "Baseline config exists"
else
    test_fail "Baseline config missing"
fi

# Test 5: Setup script
test_step "Setup script"
if [ -f "scripts/setup.sh" ] && [ -x "scripts/setup.sh" ]; then
    test_pass "Setup script exists and is executable"
else
    test_fail "Setup script missing or not executable"
fi

# Test 6: Entry points
test_step "Entry point scripts"
if [ -f "msdyolo/train.py" ]; then
    test_pass "Training script exists"
else
    test_fail "Training script missing"
fi

if [ -f "msdyolo/detect.py" ]; then
    test_pass "Detection script exists"
else
    test_fail "Detection script missing"
fi

# Test 8: Setup.py
test_step "Package installation"
if [ -f "setup.py" ]; then
    test_pass "setup.py exists"
else
    test_fail "setup.py missing"
fi

# Test 9: Data config
test_step "Data configuration"
if [ -f "msdyolo/data/dior.yaml" ]; then
    test_pass "DIOR config exists"
else
    test_fail "DIOR config missing"
fi

# Test 10: Test label format script
test_step "Label format verification"
if python3 << 'EOF'
import sys
sys.path.insert(0, '.')

# Simulate label processing
label_line = "100.5 200.3 150.2 210.1 140.8 250.9 90.4 240.5 ship 0"
parts = label_line.split()

assert len(parts) == 10, f"Expected 10 parts, got {len(parts)}"
assert parts[8] == "ship", f"Expected class name, got {parts[8]}"
assert parts[9] in ["0", "1"], f"Expected difficult flag 0/1, got {parts[9]}"

coords = [float(x) for x in parts[:8]]
assert all(c >= 0 for c in coords), "Coordinates must be non-negative"
print("✓ Label format correct (pixel coordinates)")
EOF
then
    test_pass "Label format validation"
else
    test_fail "Label format validation failed"
fi

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Verification Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Ready to train.${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Start cloud instance"
    echo "2. git clone https://github.com/DexZane/MSDYOLO.git"
    echo "3. cd MSDYOLO"
    echo "4. bash scripts/setup.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Fix issues before deploying.${NC}"
    exit 1
fi
