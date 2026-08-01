#!/usr/bin/env python3
"""Compatibility wrapper for the canonical MSDYOLO detection command."""

from msdyolo.detect import main, parse_opt


if __name__ == "__main__":
    main(parse_opt())
