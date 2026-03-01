from flask import Flask, request, jsonify
import fhe_client
import requests
import random
import logging
import traceback
import sys
import time

# 1. Set up logging to track the handshake
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- SECURE STARTUP HANDSHAKE ---
def initialize_and_sync_keys():
    logger.info("--- Edge Client Booting Up ---")
    logger.info("Initializing OpenFHE Engine into Memory...")
    
    try:
        # 1. Boot the C++ engine into RAM
        engine = fhe_client.BiometricEngine() 
        logger.info("Engine Ready. Extracting Base64 Keys for the Cloud Server...")
        
        # 2. Extract the massive mathematical keys as Base64 strings
        payload = {
            "context": engine.export_context_base64(),
            "mult_keys": engine.export_mult_keys_base64(),
            "rot_keys": engine.export_rot_keys_base64()
        }
        
        logger.info("Keys successfully serialized! Searching for Cloud Server...")
        server_url = "http://fhe-server:8080/update_keys"
        
        # 3. Robust Retry Loop (Waits for Actix to finish compiling/booting)
        while True:
            try:
                response = requests.post(server_url, json=payload, timeout=5)
                if response.status_code == 200:
                    logger.info("SUCCESS: Cloud Server accepted the FHE Evaluation Keys!")
                    break
                else:
                    logger.warning(f"Server responded with {response.status_code}. Retrying in 3s...")
            except requests.exceptions.RequestException:
                logger.warning("Server not online yet. Retrying in 3 seconds...")
            
            time.sleep(3)
            
        return engine

    except Exception as e:
        logger.critical(f"FATAL: Failed to load FHE engine. Traceback:\n{traceback.format_exc()}")
        sys.exit(1)

# Run the handshake before Flask even starts accepting web requests
fhe_engine = initialize_and_sync_keys()


# --- THE WEB FRONTEND ---
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biometric FHE Scanner</title>
    <style>
        body { font-family: sans-serif; display: flex; flex-direction: column; align-items: center; margin-top: 50px; }
        .scanner-box { border: 2px dashed #ccc; padding: 30px; border-radius: 10px; text-align: center; }
        .btn-group { margin-top: 20px; display: flex; gap: 10px; justify-content: center; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="scanner-box">
        <h2>Secure FHE Biometric Edge Device</h2>
        <form action="/process_image" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" capture="user" required>
            <div class="btn-group">
                <button type="submit" name="action" value="enroll">Register New Face</button>
                <button type="submit" name="action" value="login">Login / Search</button>
            </div>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return HTML_UI

# --- THE BACKEND LOGIC ---
@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    action = request.form.get('action')
    logger.info(f"[CAMERA TRIGGERED] Action: {action.upper()}")
    
    # 1. Extract the 512 features
    arcface_vector = [random.uniform(0.1, 0.9) for _ in range(512)]
    
    # 2. Pack the vector based on the mathematical goal
    if action == "enroll":
        logger.debug("Mode: ENROLLMENT. Padding vector with zeros...")
        packed_vector = arcface_vector + [0.0] * (8192 - 512)
        target_endpoint = "http://fhe-server:8080/enroll"
    elif action == "login":
        logger.debug("Mode: LOGIN. Duplicating vector 16 times for 1-to-N search...")
        packed_vector = arcface_vector * 16
        target_endpoint = "http://fhe-server:8080/search"
    else:
        return jsonify({"error": "Invalid action"}), 400

    # 3. Push to Rust Engine
    try:
        # TODO: We will update this to return a real Base64 Ciphertext next!
        encryption_result = fhe_engine.encrypt(packed_vector)
        logger.info(f"FHE Success: {encryption_result}")
    except Exception as e:
        logger.error(f"FHE Encryption Failed! Traceback:\n{traceback.format_exc()}")
        return jsonify({"error": "Encryption engine crashed."}), 500

    # 4. Transmit to the correct Server endpoint
    try:
        response = requests.post(target_endpoint, json={"vector": packed_vector})
        return f"<h3>{action.capitalize()} Complete!</h3><p>Cloud Server said: {response.json()}</p><a href='/'>Back to Scanner</a>"
    except Exception as e:
        return f"<h3>Network Error</h3><p>Could not reach cloud server.</p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)