#!/usr/bin/env python3
"""
Fix all imports after restructuring
Changes:
  - models.* -> msdyolo.models.*
  - utils.* -> msdyolo.utils.*
  - data.* remains in msdyolo.data.*
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single Python file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Fix imports
    replacements = [
        # Direct imports
        (r'\bfrom models\.', 'from msdyolo.models.'),
        (r'\bimport models\.', 'import msdyolo.models.'),
        (r'\bfrom utils\.', 'from msdyolo.utils.'),
        (r'\bimport utils\.', 'import msdyolo.utils.'),

        # Relative imports in msdyolo package
        # (will be handled manually if needed)
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def fix_all_imports(root_dir='msdyolo'):
    """Fix imports in all Python files"""
    fixed_count = 0

    for root, dirs, files in os.walk(root_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != '__pycache__']

        for file in files:
            if file.endswith('.py'):
                filepath = Path(root) / file
                if fix_imports_in_file(filepath):
                    print(f"Fixed: {filepath}")
                    fixed_count += 1

    print(f"\n✓ Fixed {fixed_count} files")

if __name__ == '__main__':
    print("Fixing imports in msdyolo package...")
    fix_all_imports('msdyolo')
