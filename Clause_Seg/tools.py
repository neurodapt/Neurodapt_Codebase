from datetime import datetime
import inspect
import os

def debug_print(contents):
    # Get the caller's frame
    frame = inspect.currentframe().f_back
    
    # Get file name (just the base name, not full path)
    file_name = os.path.basename(frame.f_globals.get("__file__", "unknown"))
    
    # Get function name
    function_name = frame.f_code.co_name
    
    now = datetime.now()
    date_time_info = now.strftime("%H:%M:%S.%f")[:-3]
    print(f"\n[{date_time_info}] [{file_name}] DEBUG: {contents}")

def add_to_log(message):
    file_path = "Experiments/Run_data.txt"
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"{message}\n")