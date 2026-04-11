import os
import sys

# Ensure the root directory is in the python path
# This allows server/app.py to import env.py, models.py, etc. from the parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import main

if __name__ == "__main__":
    main()
