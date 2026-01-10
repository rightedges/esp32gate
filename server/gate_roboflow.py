
import io
import json
import numpy as np
from PIL import Image
from flask import Flask, request, Response
import urllib.request
import base64
import logging
from datetime import datetime
import os
import time
import subprocess
from inference_sdk import InferenceHTTPClient

# === Setup Logging ===
logging.basicConfig(
    filename='gate_roboflow_status.log',
    level=logging.INFO,
    format='%(asctime)s gate-detector: %(message)s',
    datefmt='%b %d %H:%M:%S'
)

app = Flask(__name__)

# === Roboflow Configuration ===
API_URL = "http://192.168.50.172:9001"
API_KEY = "KjYBr0eMHv9hYhoxNIWd"
WORKSPACE_NAME = "gate-sckfl"
WORKFLOW_ID = "custom-workflow-2"

client = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# === Camera URL and Credentials ===
gate_url = 'http://192.168.50.82/ISAPI/ContentMgmt/StreamingProxy/channels/801/picture?cmd=refresh'
gate_user = 'admin'
gate_password = 'pccw1234'
auth_str = f'{gate_user}:{gate_password}'.encode('ascii')
auth_b64 = base64.b64encode(auth_str).decode('ascii')
headers = {'Authorization': 'Basic ' + auth_b64}

def get_camera_image():
    """Fetches an image from the camera."""
    try:
        req = urllib.request.Request(gate_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            img_data = response.read()
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        msg = f"Error fetching image from camera: {e}"
        print(msg)
        logging.error(msg)
        
        # Fallback to CURL if Python network is blocked (e.g., VPN issues)
        print("Attempting fallback to curl...")
        try:
            # Construct curl command
            cmd = [
                "curl", "-s",
                "-u", f"{gate_user}:{gate_password}",
                gate_url,
                "--connect-timeout", "5"
            ]
            result = subprocess.run(cmd, capture_output=True, check=True)
            return Image.open(io.BytesIO(result.stdout))
        except Exception as curl_e:
            curl_msg = f"Curl fallback failed: {curl_e}"
            print(curl_msg)
            logging.error(curl_msg)
            return None


def predict_gate_status(image):
    """Predicts the gate status (open/closed) using Roboflow Inference Server."""
    if image is None:
        return "error"
    
    try:
        # Save image to temp buffer to send to Roboflow (SDK handles numpy/PIL usually, but explicit is safe)
        # Using the SDK's image handling: it accepts filepath or numpy array or PIL image
        
        result = client.run_workflow(
            workspace_name=WORKSPACE_NAME,
            workflow_id=WORKFLOW_ID,
            images={
                "image": np.array(image)
            },
            use_cache=True
        )
        
        print(f"Roboflow Result: {result}")
        logging.info(f"Roboflow Result: {result}")
        
        # Parse the result. 
        # Since I don't know the exact workflow output structure, I will attempt to find a classification or detection.
        # This is a placeholder parsing logic. The user might need to adjust based on actual workflow output.
        # Assuming workflow returns a list of outputs or a dictionary.
        
        # Heuristic: Check for class names "open" or "closed" in the string representation or recurse
        # A better way for a specific workflow is inspecting the output keys.
        # For now, if result is a list and has 'predictions', or root keys.
        
        # NOTE: This part is highly dependent on the workflow. 
        # I will return the RAW RESULT in the log for now, and try to make a guess if 'class' is present.
        
        # Example output structure for detection: [{'predictions': [{'class': 'open', ...}]}]
        # Example output structure for classification: [{'predictions': [{'class': 'open', 'confidence': 0.9}]}]
        
        if isinstance(result, list) and len(result) > 0:
            res = result[0] # First image result
            if 'predictions' in res:
                 preds = res['predictions']
                 # Workflow specific: 'predictions' could be a dictionary or list
                 if isinstance(preds, list) and len(preds) > 0:
                     # Take highest confidence or first
                     top_pred = preds[0] # Assuming sorted or just taking first
                     if 'class' in top_pred:
                         return top_pred['class'].lower()
                 elif isinstance(preds, dict): # Single classification
                      if 'class' in preds:
                          return preds['class'].lower()
                          
            # Custom workflow output might be just the keys
            if 'open' in str(res).lower() and 'closed' not in str(res).lower():
                return 'open'
            if 'closed' in str(res).lower() and 'open' not in str(res).lower():
                return 'closed'

        return "unknown (check logs)"
        
    except Exception as e:
        print(f"Prediction error: {e}")
        logging.error(f"Prediction error: {e}")
        return "error"

# === Image Saving Helper ===
def save_image(image, label="unknown"):
    """Saves the image to the data/train/<label> directory."""
    try:
        if image is None:
            return False, "No image to save"
        
        # Create directory if it doesn't exist
        save_dir = os.path.join("data", "train", label)
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        image.save(filepath)
        logging.info(f"Saved image to {filepath}")
        return True, filepath
    except Exception as e:
        err_msg = f"Error saving image: {e}"
        logging.error(err_msg)
        print(err_msg)
        return False, err_msg

# === Routes ===

@app.route("/capture", methods=['GET'])
def capture_image_route():
    """
    API endpoint to capture and save an image with a specific label.
    Usage: /capture?status=open|closed|low_confidence
    """
    status = request.args.get('status', 'unknown')
    
    # Clean status to be safe for filesystem
    status = "".join([c for c in status if c.isalnum() or c in ('_', '-')])
    if not status:
        status = "unknown"

    image = get_camera_image()
    
    if image is None:
         return Response(json.dumps({"status": "error", "message": "Failed to fetch image from camera"}),
                        status=500, mimetype='application/json')
                        
    success, result = save_image(image, status)
    
    if success:
        return Response(json.dumps({"status": "success", "file": result, "label": status}),
                        mimetype='application/json')
    else:
        return Response(json.dumps({"status": "error", "message": result}),
                        status=500, mimetype='application/json')


@app.route("/", methods=['GET'])
def get_gate_status():
    """
    Web service endpoint that returns the gate status as JSON.
    """
    image = get_camera_image()
    status = predict_gate_status(image)
    
    if "error" in status:
        return Response(json.dumps({"status": "error", "message": f"Failed to retrieve image or predict: {status}"}),
                        status=500, mimetype='application/json')

    log_msg = f"Gate status: {status}"
    logging.info(log_msg)

    response_data = {"status": status}
    return Response(json.dumps(response_data), mimetype='application/json')

if __name__ == "__main__":
    # Use 0.0.0.0 to listen on all interfaces
    # Port 5001 to match existing setup (ensure old server is killed if running)
    app.run(host='0.0.0.0', port=5001, debug=False)
