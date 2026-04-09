import sys
import os

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.skills import execute_skill

if __name__ == "__main__":
    execute_skill("set_kettle", affordance_pose= [0.0900, 0.3763, -0.1825, 3.0800, 0.1120, -1.8970])







