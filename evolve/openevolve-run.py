#!/usr/bin/env python
"""
Entry point script for OpenEvolve
"""
import sys
from openevolve.cli import main
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    sys.exit(main())
