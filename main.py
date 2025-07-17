from ultralytics import YOLO
import cv2
import cvzone
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



def ppe_detection(file): 
    try:
        if file is None : 
            cap = cv2.VideoCapture(0)  # For Webcam
            cap.set(3, 1280)
            cap.set(4, 720)
        else : 
            cap = cv2.VideoCapture(file)  # For Video
            
        if not cap.isOpened():
            print(f"Error: Could not open video source: {file}")
            return
            
        # Load model with error handling
        if os.path.exists("best.pt"):
            model = YOLO("best.pt")
        elif os.path.exists("YOLO-Weights/ppe.pt"):
            model = YOLO("YOLO-Weights/ppe.pt")
        else:
            print("Error: No model file found")
            return

        classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                    'Safety Vest', 'machinery', 'vehicle']
        myColor = (0, 0, 255)
        while True:
            success, img = cap.read()
            if not success:
                break
                
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                    w, h = x2 - x1, y2 - y1
                    # cvzone.cornerRect(img, (x1, y1, w, h))

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = classNames[cls]
                    print(currentClass)
                    if conf>0.5:
                        if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                            myColor = (0, 0,255) # blue 
                        elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                            myColor =(0,255,0) # green
                        else:
                            myColor = (255, 0, 0) # red 

                        cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                        (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                        colorT=(255,255,255),colorR=myColor, offset=5)
                        cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error in PPE detection: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Test with available video files
    test_files = [
        "static/files/ppe-1.mp4",
        "static/files/ppe-2.mp4", 
        "Videos/ppe-1.mp4",
        "Videos/ppe-2.mp4"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            print(f"Processing {file}...")
            ppe_detection(file)
            break
    else:
        print("Testing with webcam...")
        ppe_detection(None)

