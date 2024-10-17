import psutil
import subprocess
import sys
import time

RED = '\033[31m'
GREEN = '\033[32m'
RESET_COLOR = '\033[0m'


def is_application_running(app_name):
    """Check if a process with the given name is running."""
    for process in psutil.process_iter(['pid', 'name']):
        if process.info['name'] == app_name:
            return True
    return False

def start_application(command):
    """Start the application if it's not running."""
    try:
        subprocess.Popen(command, shell=True)
        print(GREEN + f"{command} started successfully.\n" + RESET_COLOR)
    except Exception as e:
        print(RED + f"Failed to start {command}: {e}\n" + RESET_COLOR)
        print(RED + 'start_command must equal path to executable\n' + RESET_COLOR)

def check_and_start(app_name, command):
    """Check if the application is running, and start it if not."""
    if is_application_running(app_name):
        print(GREEN + f"{app_name} is already running.\n" + RESET_COLOR)
    else:
        print(GREEN + f"{app_name} is not running. Starting it now...\n" + RESET_COLOR)
        start_application(command)

if __name__ == "__main__":
    # Replace with the name of the application and the command to start it
    application_name = "Ollama"
    start_command = ["/Applications/Ollama.app/Contents/MacOS/Ollama"]
    check_and_start(application_name, start_command)

    # code below checks if ollama is running every 60 seconds
    # while True:
    #     check_and_start(application_name, start_command)
    #     time.sleep(60)
