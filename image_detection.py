from ultralytics import YOLO
import cv2
import math
import torch
import os

# Fix PyTorch model loading issue
import torch.serialization
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except:
    pass

# Alternative fix: patch torch.load to use weights_only=False
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

def image_detection(image_path):
    """Detect PPE in a single image"""
    try:
        # Load the model
        if os.path.exists("YOLO-Weights/ppe.pt"):
            model = YOLO("YOLO-Weights/ppe.pt")
        else:
            model = YOLO("ppe.pt")
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        
        # Run detection
        results = model(img, stream=False)
        
        classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                      'Safety Vest', 'machinery', 'vehicle']
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                
                if class_name == 'Mask' or class_name == 'Hardhat' or class_name == 'Safety Vest':
                    color = (0, 255, 0)
                elif class_name == 'NO-Hardhat' or class_name == 'NO-Mask' or class_name == 'NO-Safety Vest':
                    color = (0, 0, 255)
                elif class_name == 'machinery' or class_name == 'vehicle':
                    color = (0, 149, 255)
                else:
                    color = (85, 45, 255)
                    
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        
        return img
    except Exception as e:
        print(f"Error in image detection: {e}")
        return None

if __name__ == "__main__":
    # Test with available images
    test_images = [
        "static/files/building_workers_18.jpeg",
        "static/files/building_workers_24.jpeg", 
        "static/files/safety_ppe.jpg",
        "static/files/safety.jpg",
        "bus.jpg"
    ]
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"Processing {img_path}...")
            result = image_detection(img_path)
            if result is not None:
                output_path = f"detected_{os.path.basename(img_path)}"
                cv2.imwrite(output_path, result)
                print(f"Saved detection result to {output_path}")
            break
