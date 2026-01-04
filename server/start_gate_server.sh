#!/bin/bash

# Navigate to the script's directory (absolute path to be safe during reboot)
cd "$(dirname "$0")"

# Activate the virtual environment
source tiny-env/bin/activate

# Start the server
# Using exec to replace the shell with the python process, creating a clearer process tree
exec python gate.py
