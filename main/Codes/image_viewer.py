import cv2
import os
import glob

# Path to images
image_folder = "design_B_complex"
images = sorted(glob.glob(os.path.join(image_folder, "*.png")))

print(f"Found {len(images)} images in {image_folder}")
print("\nPress any key to view next image, 'q' to quit\n")

for img_path in images:
    img_name = os.path.basename(img_path)
    print(f"Viewing: {img_name}")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Could not load {img_name}")
        continue
    
    # Resize if too large
    h, w = img.shape[:2]
    max_h, max_w = 900, 1600
    if h > max_h or w > max_w:
        scale = min(max_w/w, max_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        img = cv2.resize(img, (new_w, new_h))
    
    cv2.imshow(f"Image Viewer - {img_name}", img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if key == ord('q'):
        break

print("\nDone!")
