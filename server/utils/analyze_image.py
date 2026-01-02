from PIL import Image
import os

img_path = "/Users/william/.gemini/antigravity/brain/a962a291-43ce-4bf1-9e71-d6b4f59b1d06/uploaded_image_1767251685323.jpg"

if os.path.exists(img_path):
    img = Image.open(img_path)
    print(f"Image Resolution: {img.size[0]}x{img.size[1]}")
    print(f"Format: {img.format}")
else:
    print("File not found.")
