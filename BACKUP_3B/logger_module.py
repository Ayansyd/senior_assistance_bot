# logger_module.py
import datetime

class InteractionLogger:
    def __init__(self, log_file="assistant_log.txt"):
        self.log_file = log_file

    def log_interaction(self, user_text: str, system_text: str):
        """
        Log the conversation turn to a file.
        """
        timestamp = datetime.datetime.now().isoformat()
        log_entry = (
            f"[{timestamp}]\n"
            f"User: {user_text}\n"
            f"Assistant: {system_text}\n"
            "----------------------------------------\n"
        )
        self._write_to_file(log_entry)

    def log_error(self, error_msg: str):
        """
        Log errors or exceptions.
        """
        timestamp = datetime.datetime.now().isoformat()
        log_entry = (
            f"[{timestamp}] ERROR: {error_msg}\n"
            "----------------------------------------\n"
        )
        self._write_to_file(log_entry)

    def _write_to_file(self, text: str):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"[Logging Error]: {e}")
