# heartbeat_server.py
import logging
import threading
from flask import Flask, request, jsonify
from werkzeug.serving import make_server

# --- Shared Data ---
# Use a dictionary to store the latest reading, protected by a lock
latest_reading = {"heart_rate": None, "timestamp": None}
data_lock = threading.Lock() # Lock specifically for accessing latest_reading

# --- Flask App Setup ---
app = Flask(__name__)

# Configure Flask logging to be less verbose or match main app
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING) # Reduce Flask's default request logging noise

# --- API Endpoint ---
@app.route('/heartbeat', methods=['POST'])
def receive_heartbeat():
    """Receives heartbeat data via POST request."""
    global latest_reading
    if not request.is_json:
        logging.warning("[HeartbeatServer] Received non-JSON request")
        return jsonify({"status": "error", "message": "Request must be JSON"}), 400

    data = request.get_json()
    heart_rate = data.get('heart_rate')
    timestamp = data.get('timestamp') # Optional timestamp from source

    if heart_rate is None:
        logging.warning("[HeartbeatServer] Received JSON without 'heart_rate' key")
        return jsonify({"status": "error", "message": "Missing 'heart_rate' key"}), 400

    try:
        hr_value = int(heart_rate)
        # Optional: Add validation for realistic heart rate range (e.g., 30-250)
        if not (30 <= hr_value <= 250):
             logging.warning(f"[HeartbeatServer] Received unrealistic heart rate: {hr_value}")
             # Decide whether to store it or reject it
             # return jsonify({"status": "error", "message": "Heart rate out of range"}), 400

        # --- Update Shared Data Safely ---
        with data_lock:
            latest_reading["heart_rate"] = hr_value
            latest_reading["timestamp"] = timestamp or time.time() # Use provided or server time
            logging.info(f"[HeartbeatServer] Received Heart Rate: {hr_value}")

        return jsonify({"status": "success", "message": "Heart rate received"}), 200

    except (ValueError, TypeError):
        logging.warning(f"[HeartbeatServer] Invalid heart rate value received: {heart_rate}")
        return jsonify({"status": "error", "message": "Invalid 'heart_rate' value"}), 400
    except Exception as e:
        logging.error(f"[HeartbeatServer] Error processing request: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal server error"}), 500

# --- Function to get latest data safely ---
def get_latest_heart_rate_reading() -> dict | None:
    """Returns the latest reading (or None) safely."""
    with data_lock:
        # Return a copy to avoid external modification issues
        if latest_reading["heart_rate"] is not None:
            return latest_reading.copy()
        return None

# --- Server Thread ---
# Use threading.Thread to run Flask in the background
# Note: Flask's dev server is not recommended for production.
# Consider 'waitress' for a more robust WSGI server.
class ServerThread(threading.Thread):
    def __init__(self, flask_app, host='0.0.0.0', port=5002):
        threading.Thread.__init__(self)
        # Use make_server for better control, especially shutdown
        self.srv = make_server(host, port, flask_app, threaded=True)
        self.ctx = flask_app.app_context()
        self.ctx.push()
        logging.info(f"[HeartbeatServer] Server thread initialized for {host}:{port}")

    def run(self):
        logging.info('[HeartbeatServer] Starting Flask server...')
        self.srv.serve_forever()
        logging.info('[HeartbeatServer] Flask server stopped.')

    def shutdown(self):
        logging.info('[HeartbeatServer] Attempting to shut down Flask server...')
        # This is the correct way to shut down make_server
        self.srv.shutdown()

# --- Global server thread variable ---
server_thread = None

def start_server(host='0.0.0.0', port=5002):
    """Starts the Flask server in a background thread."""
    global server_thread
    if server_thread is None or not server_thread.is_alive():
        server_thread = ServerThread(app, host=host, port=port)
        server_thread.daemon = True # Allow main program to exit even if this thread is running
        server_thread.start()
    else:
        logging.warning("[HeartbeatServer] Server thread already running.")

def stop_server():
    """Stops the Flask server thread."""
    global server_thread
    if server_thread is not None and server_thread.is_alive():
        logging.info("[HeartbeatServer] Signaling server thread to stop...")
        server_thread.shutdown()
        server_thread.join(timeout=5) # Wait briefly for it to stop
        if server_thread.is_alive():
             logging.warning("[HeartbeatServer] Server thread did not shut down cleanly.")
        server_thread = None
    else:
        logging.info("[HeartbeatServer] Server thread not running or already stopped.")

# Import time for timestamp fallback
import time
