#!/usr/bin/env python3
"""Compatibility wrapper for the canonical MSDYOLO validation command."""

from msdyolo.val import main, parse_opt


if __name__ == "__main__":
    main(parse_opt())
