import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pgcdi.train import train
if __name__=="__main__":
    train()
