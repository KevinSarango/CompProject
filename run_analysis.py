#!/usr/bin/env python3
"""
run_analysis.py

Wrapper entry point for the RL training correlation analyzer.
This script is compatible with the current project layout and will
find logs in the project root or in the data/ directory.

Usage:
    python3 run_analysis.py
"""

from analyze_correlation import main

if __name__ == '__main__':
    main()
