######
#Persisting Tracks Loop
import cv2
import os
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("trained_model/best_10_modify.pt")

# Open the video file
video_path = "videos/沒戴護目鏡.mp4"
cap = cv2.VideoCapture(video_path)
previous_boxes = None
# Loop through the video frames
count_tracking = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,conf=0.3,iou=0.5)

        # Print out object IDs and their corresponding boxes
        for i, box in enumerate(results[0].boxes):
          print(f'Object ID: {box.id}, Bounding box: {box.xyxy},class_anme: {box.cls}')
  
        count_tracking =  count_tracking + 1
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        processed_image_path = os.path.join("save_dir/tracking/", f"track_{count_tracking:04d}.jpg")
        cv2.imwrite(processed_image_path, annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLO8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
######


######
#Plotting Tracks Over Time
# from collections import defaultdict

# import cv2
# import numpy as np

# from ultralytics import YOLO

# # Load the YOLO11 model
# model = YOLO("trained_model/best_10_modify.pt")

# # Open the video file
# video_path = "videos/沒戴護目鏡.mp4"
# cap = cv2.VideoCapture(video_path)

# # Store the track history
# track_history = defaultdict(lambda: [])

# # Loop through the video frames
# while cap.isOpened():
#   # Read a frame from the video
#   success, frame = cap.read()

#   if success:
#     # Run YOLO11 tracking on the frame, persisting tracks between frames
#     results = model.track(frame, persist=True)
#     if results[0].boxes.id != None:
#     # Get the boxes and track IDs
#       boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
#       track_ids = results[0].boxes.id.cpu().numpy().astype(int)
#       # Visualize the results on the frame
#       annotated_frame = results[0].plot()

#       # Plot the tracks
#       for box, track_id in zip(boxes, track_ids):
#         x, y, w, h = box
#         track = track_history[track_id]
#         track.append((float(x), float(y)))  # x, y center point
#         if len(track) > 30:  # retain 90 tracks for 90 frames
#           track.pop(0)

#         # Draw the tracking lines
#         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#         cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

#         # Display the annotated frame
#         cv2.imshow("YOLO11 Tracking", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#           break
#   else:
#     # Break the loop if the end of the video is reached
#     break

# # Release the video capture object and close the display window
# cap.release()
# cv2.destroyAllWindows()

######