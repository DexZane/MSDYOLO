#!/bin/bash
###############################################################################
# Clean up old files after restructuring
###############################################################################

set -e

echo "Cleaning up old files..."

# Remove old log files
echo "Removing log files..."
rm -f *.log

# Archive old structure (don't delete, keep for reference)
echo "Archiving old files..."
mkdir -p .old_structure

# Move old top-level Python files (now in msdyolo/)
if [ -f "trainmsd.py" ]; then
    mv trainmsd.py .old_structure/
fi

# Move old models/ (now msdyolo/models/)
if [ -d "models" ] && [ -d "msdyolo/models" ]; then
    echo "Old models/ directory exists, keeping for reference"
    # Don't move, user can manually check differences
fi

# Move old utils/ (now msdyolo/utils/)
if [ -d "utils" ] && [ -d "msdyolo/utils" ]; then
    echo "Old utils/ directory exists, keeping for reference"
fi

# Replace README
if [ -f "README_NEW.md" ]; then
    mv README.md README_old.md
    mv README_NEW.md README.md
fi

echo ""
echo "✓ Cleanup complete!"
echo ""
echo "Old files archived in .old_structure/"
echo "You can delete them after verifying the new structure works."
echo ""
echo "Next step: git add -A && git commit -m 'Restructure project'"
