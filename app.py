from flask import Flask, render_template, request
import cv2
import torch
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()

# Function to detect vehicles on road and count the number of vehicles by category
def detect_vehicles(frame):
    # Convert frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert frame to PIL image
    pil_img = Image.fromarray(frame_rgb)
    # Perform inference
    results = model(pil_img)
    # Filter results to include vehicles (assuming class indices for vehicles are 2, 3, 5, 7, and 9)
    vehicle_results = results.pred[0][(results.pred[0][:, 5] == 2) | (results.pred[0][:, 5] == 3) |
                                      (results.pred[0][:, 5] == 5) | (results.pred[0][:, 5] == 7) |
                                      (results.pred[0][:, 5] == 9)]
    # Initialize counters for different vehicle categories
    car_count = 0
    bike_count = 0
    truck_count = 0
    bus_count = 0

    # Initialize total vehicle count
    total_count = 0

    # Count vehicles by category
    for vehicle in vehicle_results:
        total_count += 1
        if vehicle[5] == 2:  # Car
            car_count += 1
        elif vehicle[5] == 3:  # Bike
            bike_count += 1
        elif vehicle[5] == 5:  # Truck
            truck_count += 1
        elif vehicle[5] == 7:  # Bus
            bus_count += 1

    return car_count, bike_count, truck_count, bus_count, total_count

@app.route('/', methods=['GET', 'POST'])
def index():
    video_path = ""

    if request.method == 'POST':
        # Get the value of the clicked button
        clicked_button = request.form.get('camera')

        # Set the corresponding video path based on the button clicked
        if clicked_button == 'Camera 1':
            video_path = "D:/BDT Mini Project/Dataset/traffic-1.mp4"
        elif clicked_button == 'Camera 2':
            video_path = "D:/BDT Mini Project/Dataset/traffic-2.mp4"
        elif clicked_button == 'Camera 3':
            video_path = "D:/BDT Mini Project/Dataset/traffic-3.mp4"
        elif clicked_button == 'Camera 4':
            video_path = "D:/BDT Mini Project/Dataset/traffic-4.mp4"

    if video_path:
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Detect vehicles and count the number of vehicles by category
            car_count, bike_count, truck_count, bus_count, total_count = detect_vehicles(frame)

            # Display the total number of vehicles on the frame
            cv2.putText(frame, f'Total Vehicles: {total_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Vehicle Detection', frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture
        cap.release()
        cv2.destroyAllWindows()

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
