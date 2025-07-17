from ultralytics import YOLO
import cv2
import math
import torch

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

def video_detection(path_x):
    video_capture = path_x
    
    # Check if path is valid
    if not path_x:
        print("Error: No video path provided")
        return
    
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    
    # Check if video capture is successful
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_capture}")
        return
        
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    # Fix PyTorch model loading with weights_only=False for trusted model
    try:
        model=YOLO("YOLO-Weights/ppe.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try with the alternative model
        try:
            model=YOLO("ppe.pt")
        except Exception as e2:
            print(f"Error loading alternative model: {e2}")
            return
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
                'Safety Vest', 'machinery', 'vehicle']
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'Mask' or class_name == 'Hardhat' or class_name == 'Safety Vest':
                    color=(0, 255,0)

                elif class_name == 'NO-Hardhat' or class_name == 'NO-Mask' or class_name == 'NO-Safety Vest':
                    color = (0,0,255)

                elif class_name == 'machinery' or class_name == 'vehicle':
                    color = (0, 149, 255)
                else:
                    color = (85,45,255)
                if conf>0.5:
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                    cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()