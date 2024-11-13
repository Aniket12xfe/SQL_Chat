import os
import subprocess

# Run the Streamlit app
subprocess.run(["streamlit", "run", "app.py", "--server.port", os.environ.get("PORT", "8080")])
