#!/bin/bash
# Quick fix for Python 3.12 pkg_resources issue

echo "Fixing Python 3.12 compatibility..."

# Downgrade setuptools to fix pkg_resources
pip install setuptools==69.5.1 -q

echo "✓ Fixed"
