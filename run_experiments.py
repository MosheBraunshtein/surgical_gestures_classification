from __future__ import print_function
import subprocess
import os
import sys


users_to_test = ["B", "C", "D", "E", "F", "G", "H", "I"]

script_name = "train_and_summarize.py"

print("--- Starting Manager ---")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())

for user in users_to_test:
    print("\n>>> Testing User: {}".format(user))
    sys.stdout.flush()

    cmd = [sys.executable, script_name, "--test_users", user,"--hidden_layer_size", "512"]
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=None, # שולח ישירות למסך שלך
            stderr=None, # שולח שגיאות ישירות למסך שלך
            shell=True   # לפעמים הכרחי ב-Windows להרצת פקודות CMD
        )
        process.wait() # ממתין לסיום הריצה של המשתמש הנוכחי
        
        if process.returncode != 0:
            print("Process failed with code:", process.returncode)
    except Exception as e:
        print("An error occurred:", str(e))

print("\n--- Manager Finished ---")