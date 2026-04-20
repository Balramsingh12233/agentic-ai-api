import os
import datetime
from utils.security import encryptor

class EventLogger:
    def __init__(self):
        # We'll save the log file in the backend directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.log_file_path = os.path.join(base_dir, 'events.log')
        
    def log_event(self, module: str, status: str, details: str):
        """
        Logs an event with a timestamp to the events.log file.
        The 'details' are encrypted for privacy before saving.
        """
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # ACT phase: Encrypt the sensitive details
        encrypted_details = encryptor.encrypt(details)
        
        log_entry = f"[{now}] [{module.upper()}] Status={status} | Encrypted_Details={encrypted_details}\n"
        
        # Open in append mode so we don't overwrite previous logs
        with open(self.log_file_path, 'a') as f:
            f.write(log_entry)
            
        print(f"Agent Logged Secure Event ({module}): {status}")

# export a single instance
logger = EventLogger()
