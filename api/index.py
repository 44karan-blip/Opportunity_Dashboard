import os
import sys

# Track the structural path layout to ensure internal app assets map cleanly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", "app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
    sys.exit(stcli.main())