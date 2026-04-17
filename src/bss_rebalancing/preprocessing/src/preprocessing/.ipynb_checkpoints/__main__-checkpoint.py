"""
Entry point for running the preprocessing pipeline as a module.

Usage:
    python -m preprocessing --data-path data/
"""

from preprocessing.cli import main

if __name__ == "__main__":
    main()