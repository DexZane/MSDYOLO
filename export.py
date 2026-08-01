#!/usr/bin/env python3
"""Compatibility wrapper for the canonical MSDYOLO export command."""

from msdyolo.export import main, parse_opt


if __name__ == "__main__":
    main(parse_opt())
