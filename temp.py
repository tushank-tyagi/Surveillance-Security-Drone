import cv2
import torch
from ultralytics import YOLO

# Ensure the model is loaded on the GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("runs\detect\tune3\weights\best.pt").to(device)

# Set up video capture (change index if necessary)
capture_device_index = 0
cap = cv2.VideoCapture(capture_device_index)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to RGB (YOLOv8 expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model.predict(source=rgb_frame, stream=True, conf=0.4)

    # Draw bounding boxes and labels on the frame
    for result in results:
        for bbox in result.boxes:
            # Convert bbox.xyxy tensor to a list and map to integers
            x1, y1, x2, y2 = map(int, bbox.xyxy.tolist()[0])
            confidence = bbox.conf.item()
            class_id = int(bbox.cls.item())
            label = f"{model.names[class_id]} {confidence:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Drone Live Feed with YOLOv8', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
