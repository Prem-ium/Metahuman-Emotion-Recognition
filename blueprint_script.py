import subprocess
import sys

# Set the directory where the script is located. Replace with your own path
directory = r"C:\Users\Frost\Desktop\CodingProjects\EmotionDetection-main"

# Set the name of the script to be started
script_name = "emotional-detection-main.py"

# Use subprocess to start the script in the specified directory
process = subprocess.Popen(['python', script_name], cwd=directory)

# Stop the script from running after calling the other script
sys.exit(0)