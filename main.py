import subprocess
import sys

subprocess.run([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--server.address",
    "0.0.0.0",
    "--server.port",
    "3000",
    "--server.headless",
    "true",
], check=False)
