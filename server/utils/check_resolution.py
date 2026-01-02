import io
import urllib.request
import base64
from PIL import Image

gate_url = 'http://192.168.50.82/ISAPI/ContentMgmt/StreamingProxy/channels/801/picture?cmd=refresh'
gate_user = 'admin'
gate_password = 'pccw1234'
auth_str = f'{gate_user}:{gate_password}'.encode('ascii')
auth_b64 = base64.b64encode(auth_str).decode('ascii')
headers = {'Authorization': 'Basic ' + auth_b64}

def get_resolution():
    try:
        req = urllib.request.Request(gate_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            img = Image.open(io.BytesIO(response.read()))
            print(f"Server Camera Resolution: {img.size[0]}x{img.size[1]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_resolution()
