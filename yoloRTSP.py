import os
import sys

# Temporarily comment out to allow error messages for debugging
# Suppress FFmpeg error messages
# sys.stderr = open(os.devnull, 'w')
# Set FFmpeg log level to quiet
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;quiet"

import cv2
import queue
import time
import threading
from ultralytics import YOLO
import requests

# Force detection to use CPU
device = "cpu"
# print(f"Using device: {device}")

# Alert settings
alert_threshold = 0.8  # Confidence threshold for alerts
alert_cooldown = 300  # Cooldown time in seconds
alert_records = []  # Store detection events for batch alerts
last_alert_times = {}  # Store the last alert time for each camera
alert_interval = 300  # Batch alert interval (5 minutes)

# Mutex lock to ensure thread safety
mutex = threading.Lock()

# Load YOLOv8 model (ensure this model is trained with the 7 specified classes)
model = YOLO('best.pt').to(device)  # Replace 'best.pt' with your model file

# Mapping from class index to event name (English)
class_event_mapping_en = {
    0: "Falling",
    1: "Fire",
    2: "No gloves",
    3: "Smoke",
    4: "Wear gloves",
    5: "Using cell phone",
    6: "Without goggles"
}

# Mapping from class index to event name (Chinese)
class_event_mapping_cn = {
    0: "倒臥",
    1: "火災",
    2: "未佩戴手套",
    3: "煙霧",
    4: "佩戴手套",
    5: "使用手機",
    6: "未佩戴護目鏡"
}

# List of RTSP camera URLs
camera_urls = [
    "rtsp://hikvision:Unitech0815!@10.40.11.20:554",
    "rtsp://hikvision:Unitech0815!@10.20.17.20:554",
    # Add more camera URLs here
]

# Define ROI for each camera as (x1, y1, x2, y2)
# Adjust these values according to your camera angle.
camera_rois = [
    (400, 0, 1200, 700),  # ROI for camera 0
    (600, 0, 1100, 700),  # ROI for camera 1
    # Add more ROIs corresponding to additional cameras
]

def send_alert(frame, camera_index, detections):
    """
    發送警報的函數，處理偵測到的事件並與 API 進行互動。
    detections 現在是一個包含字典的列表，每個字典包含 'xyxy', 'conf', 'cls'。
    """
    global last_alert_times
    current_time = time.time()

    # Initialize the last alert time of the camera.
    if camera_index not in last_alert_times:
        last_alert_times[camera_index] = 0

    if current_time - last_alert_times[camera_index] > alert_cooldown:
        last_alert_times[camera_index] = current_time
        print(f"Sending alert for camera {camera_index} to API!")

        # 在影像上繪製所有偵測到的邊界框和標籤（使用英文）
        for box in detections:
            confidence = float(box['conf'][0])
            cls = int(box['cls'][0])  # 取得類別索引
            event_name_en = class_event_mapping_en.get(cls, "Unknown Event")
            x1, y1, x2, y2 = box['xyxy']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{event_name_en} {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 設定位置根據 camera_id
        camera_id = camera_index + 1  # 假設 camera_index 是 0-based
        if camera_id == 1:
            location = "四廠鑽孔站"
        elif camera_id == 2:
            location = "二廠ATO站"
        else:
            location = f"未知位置_{camera_id}"

        # 格式化檔名為 "1-2024-12-12_17-53-11.jpg"
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        formatted_filename = f"{camera_id}-{timestamp}.jpg"

        # 保存警報截圖
        success = cv2.imwrite(formatted_filename, frame)
        if success:
            print(f"Saved screenshot: {formatted_filename}")
        else:
            print(f"Failed to save screenshot: {formatted_filename}")

        # 準備 API 請求的數據（使用中文事件名稱）
        api_url = "https://eip.pcbut.com.tw/File/UploadYoloImage"
        camera_model = {
            "cameraId": str(camera_id),
            "location": location,
            "eventName": "；".join([class_event_mapping_cn.get(int(box['cls'][0]), "未知事件") for box in detections]),
            "eventDate": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": f"{len(detections)} events detected with confidence > {alert_threshold}",
            "fileName": formatted_filename,
            "result": f"疑似發生 {'、'.join([class_event_mapping_cn.get(int(box['cls'][0]), '未知事件') for box in detections])}, 請同仁儘速查看"
        }

        # 發送包含影像和攝影機數據的 POST 請求
        try:
            with open(formatted_filename, 'rb') as img_file:
                files = {'files': (formatted_filename, img_file, 'image/jpeg')}
                response = requests.post(api_url, files=files, data=camera_model, verify=False)

            if response.status_code == 200:
                print(f"Successfully sent alert for camera {camera_index}. Response: {response.text}")
            else:
                print(f"Failed to send alert for camera {camera_index}. Status Code: {response.status_code}, Response: {response.text}")
        except Exception as e:
            print(f"Error sending alert for camera {camera_index}: {e}")

# Global stop event for all threads
stop_event = threading.Event()

def batch_alert():
    """
    處理批次警報的函數，每隔一段時間檢查一次 alert_records 並發送報告。
    """
    while not stop_event.is_set():
        time.sleep(alert_interval)
        with mutex:
            if alert_records:
                alert_message = f"Alert Report: {len(alert_records)} events detected."
                print(alert_message)
                # 在此添加批次警報郵件發送邏輯
                alert_records.clear()

def process_camera(camera_index, camera_url):
    """
    處理單個攝影機的接收和顯示函數。
    """
    q = queue.Queue(maxsize=10)

    def receive():
        print(f'Starting to receive from camera {camera_index}')
        cap = None
        try:
            while not stop_event.is_set():
                if cap is None or not cap.isOpened():
                    cap = cv2.VideoCapture(camera_url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        print(f"Failed to connect to camera {camera_index}, retrying...")
                        time.sleep(5)
                        continue

                ret, frame = cap.read()
                if not ret:
                    print(f"Connection lost to camera {camera_index}, reconnecting...")
                    cap.release()
                    cap = None
                    time.sleep(5)
                    continue

                if not q.full():
                    q.put(frame)

        except Exception as e:
            print(f"Exception in receive thread for camera {camera_index}: {e}")

        finally:
            if cap and cap.isOpened():
                cap.release()
            print(f"Stopped receiving from camera {camera_index}")

    def display():
        print(f"Starting to display camera {camera_index}")
        try:
            while not stop_event.is_set():
                if not q.empty():
                    frame = q.get()

                    if frame is not None:
                        # 打印影像尺寸以確認 ROI 是否有效
                        frame_height, frame_width = frame.shape[:2]
                        print(f"Camera {camera_index} - Frame size: {frame_width}x{frame_height}")

                        # Retrieve ROI for this camera
                        if camera_index < len(camera_rois):
                            roi = camera_rois[camera_index]
                            x1, y1, x2, y2 = roi
                            # Validate ROI coordinates against frame size
                            x1 = max(0, min(x1, frame_width - 1))
                            x2 = max(0, min(x2, frame_width))
                            y1 = max(0, min(y1, frame_height - 1))
                            y2 = max(0, min(y2, frame_height))

                            if x1 >= x2 or y1 >= y2:
                                print(f"Invalid ROI for camera {camera_index}: {roi}. Skipping ROI processing.")
                                roi = None
                                roi_frame = frame
                            else:
                                # Draw ROI rectangle on the frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                # Optionally, mask out the area outside ROI for detection
                                mask = frame.copy()
                                mask[:y1, :] = 0
                                mask[y2:, :] = 0
                                mask[:, :x1] = 0
                                mask[:, x2:] = 0
                                roi_frame = mask
                        else:
                            roi = None
                            roi_frame = frame  # If no ROI defined, use the whole frame

                        # Run YOLOv8 detection on the masked frame (only ROI area is active)
                        try:
                            results = model(roi_frame, verbose=False)
                        except Exception as e:
                            print(f"Error during model inference for camera {camera_index}: {e}")
                            continue

                        detections = []  # Store detections for alerts
                        with mutex:
                            for result in results:
                                for box in result.boxes:
                                    confidence = float(box.conf[0])
                                    cls = int(box.cls[0])  # Get class index
                                    if confidence > alert_threshold:
                                        # Get bounding box coordinates
                                        x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
                                        # If ROI is defined, ensure the box is within ROI
                                        if roi:
                                            if x1 <= x1_box <= x2 and x1 <= x2_box <= x2 and y1 <= y1_box <= y2 and y1 <= y2_box <= y2:
                                                adjusted_box = {
                                                    'xyxy': [x1_box, y1_box, x2_box, y2_box],
                                                    'conf': box.conf,
                                                    'cls': box.cls
                                                }
                                                detections.append(adjusted_box)
                                        else:
                                            adjusted_box = {
                                                'xyxy': [x1_box, y1_box, x2_box, y2_box],
                                                'conf': box.conf,
                                                'cls': box.cls
                                            }
                                            detections.append(adjusted_box)

                            if detections:
                                # 將警報處理放入獨立的線程，以避免阻塞
                                alert_thread = threading.Thread(target=send_alert, args=(frame.copy(), camera_index, detections), daemon=True)
                                alert_thread.start()

                            # Draw bounding boxes and labels on the display frame (using English)
                            for box in detections:
                                confidence = float(box['conf'][0])
                                cls = int(box['cls'][0])
                                event_name_en = class_event_mapping_en.get(cls, "Unknown Event")
                                x1_box, y1_box, x2_box, y2_box = box['xyxy']
                                cv2.rectangle(frame, (x1_box, y1_box), (x2_box, y2_box), (0, 255, 0), 2)
                                label = f'{event_name_en} {confidence:.2f}'
                                cv2.putText(frame, label, (x1_box, y1_box - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Resize frame for display
                        resized_frame = cv2.resize(frame, (640, 400))
                        cv2.imshow(f"Camera {camera_index}", resized_frame)

                # Check for stop event and close window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

        except Exception as e:
            print(f"Exception in display thread for camera {camera_index}: {e}")

        finally:
            cv2.destroyWindow(f"Camera {camera_index}")
            print(f"Stopped displaying camera {camera_index}")

    # Start receive and display threads for this camera
    receive_thread = threading.Thread(target=receive, daemon=True)
    display_thread = threading.Thread(target=display, daemon=True)
    receive_thread.start()
    display_thread.start()

    return receive_thread, display_thread

if __name__ == '__main__':
    # Start batch alert processing thread #daemon: 主程式結束時強制結束thread
    alert_thread = threading.Thread(target=batch_alert, daemon=True)
    alert_thread.start()

    # Start threads for all cameras
    camera_threads = []
    for index, url in enumerate(camera_urls):
        threads = process_camera(index, url)
        camera_threads.extend(threads)

    try:
        # Wait for all threads to complete
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        print("Interrupt received, stopping all threads...")
    finally:
        for thread in camera_threads:
            thread.join(timeout=2)
        alert_thread.join(timeout=2)
        cv2.destroyAllWindows()
        print("All resources released, exiting.")