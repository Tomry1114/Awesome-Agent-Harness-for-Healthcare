"""Prepare the three study datasets for runner/run.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'runner'))
from adapters.user_study.conversion import main

if __name__ == '__main__':
    main()
