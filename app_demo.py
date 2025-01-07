import cv2
from ultralytics import YOLO

import os

# Load YOLO model
model_path = 'trained_model/best_10_modify.pt'  # Replace with your trained model path
model = YOLO(model_path)

# Video paths (replace with your video files)
video_paths = [
    "videos/沒戴護目鏡.mp4",
    # Add more video paths here
]

# Output directory
output_dir = "output_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Class mapping
class_event_mapping = {
    0: "No gloves",
    1: "No goggles",
}

# Define ROI for "No goggles" detection
roi_x1, roi_y1, roi_x2, roi_y2 = 700, 100, 1000, 400
#roi_x1, roi_y1, roi_x2, roi_y2 = 200, 100, 1900, 900

# Confidence threshold for detection
confidence_config = 0.6

def process_video(video_path, video_index):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Unable to open video: {video_path}")
        return

    # Get video properties
    # example shape is (1080, 1920, 3)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
    output_video_path = os.path.join(output_dir, f"processed_video_{video_index}.mp4")

    # Initialize VideoWriter
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video {video_path}.")
            break

        # Perform YOLO inference
        results = model(frame, conf=confidence_config)

        # Draw ROI on the frame with dashed lines
        line_type = cv2.LINE_AA #圓線
        for i in range(roi_x1, roi_x2, 10):
            #cv2.line (image,start_point,end_point,color,thick,line_type)
            cv2.line(frame, (i, roi_y1), (i + 5, roi_y1), (255, 255, 0), 2, line_type)
            cv2.line(frame, (i, roi_y2), (i + 5, roi_y2), (255, 255, 0), 2, line_type)
        for i in range(roi_y1, roi_y2, 10):
            cv2.line(frame, (roi_x1, i), (roi_x1, i + 5), (255, 255, 0), 2, line_type)
            cv2.line(frame, (roi_x2, i), (roi_x2, i + 5), (255, 255, 0), 2, line_type)
        cv2.putText(frame, "ROI", (roi_x1, roi_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
        #cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
        # fontFace 文字字型
        # fontScale 文字尺寸
        
        for result in results:
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            classes = result.boxes.cls

            for box, confidence, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = map(int, box)
                class_name = class_event_mapping.get(int(cls), "Unknown")

                # Check conditions for "No goggles" and "No gloves"
                if class_name == "No goggles" and (x1 >= roi_x1 and y1 >= roi_y1 and x2 <= roi_x2 and y2 <= roi_y2):
                    # Draw bounding box for "No goggles" (within ROI)
                    label = f"{class_name} {confidence:.2f}"
                    color = (255, 0, 0)  # Blue for No goggles
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                elif class_name == "No gloves":
                    # Draw bounding box for "No gloves" (anywhere)
                    label = f"{class_name} {confidence:.2f}"
                    color = (0, 255, 0)  # Green for No gloves
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # Write frame to the video file
        video_writer.write(frame)

        # Display the frame
        resized_frame = cv2.resize(frame, (640, 400))
        cv2.imshow(f"Video {video_index}", resized_frame)

        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyWindow(f"Video {video_index}")
    print(f"Processed video saved to {output_video_path}")

if __name__ == "__main__":
    for index, video_path in enumerate(video_paths):
        #依序處理路徑下的每個videos
        #enumerate 帶 index 跟 value
        process_video(video_path, index)
    cv2.destroyAllWindows()
    print("All videos processed.")
