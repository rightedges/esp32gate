
import os
from inference_sdk import InferenceHTTPClient

# Configuration
IMAGE_DIR = "data/nolabel"
API_URL = "http://192.168.50.172:9001"
API_KEY = "KjYBr0eMHv9hYhoxNIWd"
WORKSPACE_NAME = "gate-sckfl"
WORKFLOW_ID = "custom-workflow-2"

def main():
    # Check if directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"Directory not found: {IMAGE_DIR}")
        return

    # Initialize client
    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY
    )

    # Get list of images
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    print(f"Found {len(images)} images to process...")

    for image_name in images:
        image_path = os.path.join(IMAGE_DIR, image_name)
        print(f"\nProcessing: {image_name}")
        
        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE_NAME,
                workflow_id=WORKFLOW_ID,
                images={
                    "image": image_path
                },
                use_cache=True
            )
            print(f"Result for {image_name}:")
            print(result)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

if __name__ == "__main__":
    main()
