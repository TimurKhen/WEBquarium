import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ml.fish_brain import train

if __name__ == '__main__':
    train('ml/brain.json')