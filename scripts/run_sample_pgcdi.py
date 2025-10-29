import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pgcdi.sample import run_sample
if __name__=="__main__":
    run_sample()
