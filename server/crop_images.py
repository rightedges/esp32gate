
import json
import os
from PIL import Image

# Configuration
ROI_FILE = 'roi.json'
INPUT_DIR = 'data/train'
OUTPUT_DIR = 'data/roboflow'

def load_roi(roi_path):
    with open(roi_path, 'r') as f:
        return json.load(f)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def crop_and_save_images(roi):
    x, y, x1, y1 = roi['x'], roi['y'], roi['x1'], roi['y1']
    box = (x, y, x1, y1)
    
    print(f"Cropping with box: {box}")

    for category in ['open', 'closed']:
        input_category_dir = os.path.join(INPUT_DIR, category)
        output_category_dir = os.path.join(OUTPUT_DIR, category)
        
        if not os.path.exists(input_category_dir):
            print(f"Directory not found: {input_category_dir}, skipping.")
            continue
            
        ensure_dir(output_category_dir)
        
        files = [f for f in os.listdir(input_category_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"Found {len(files)} images in {category}")

        for filename in files:
            input_path = os.path.join(input_category_dir, filename)
            output_path = os.path.join(output_category_dir, filename)
            
            try:
                with Image.open(input_path) as img:
                    cropped_img = img.crop(box)
                    cropped_img.save(output_path)
                    # print(f"Saved {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == '__main__':
    # Script assumes it's run from the server directory
    if not os.path.exists(ROI_FILE):
        print(f"Error: {ROI_FILE} not found. Please run this script from the 'server' directory.")
    else:
        roi_data = load_roi(ROI_FILE)
        crop_and_save_images(roi_data)
        print("Done.")
