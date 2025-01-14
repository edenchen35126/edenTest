import cv2
import os

video_path = "datasets/test_fire/2.mp4"
be_cropped_video_path = "videos/沒戴手套擦拭.mp4"

# 開啟影片檔案
cap = cv2.VideoCapture(video_path)
be_cropped_cap = cv2.VideoCapture(be_cropped_video_path)


if not cap.isOpened() or not be_cropped_cap.isOpened():
    print("Error: Unable to open video.")
    exit()



# Output directory
output_dir = "output_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ##讀取圖片每秒幾個frame
# fps = cap.get(cv2.CAP_PROP_FPS)
# print(f"Frames per second: {fps}")
video_index = 0

# Get video properties
# example shape is (1080, 1920, 3)
fps = int(be_cropped_cap.get(cv2.CAP_PROP_FPS))
width = int(be_cropped_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(be_cropped_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
output_video_path = os.path.join(output_dir, f"processed_video_{video_index}.mp4")
print(f"video fps : {fps} , width : {width} , height : {height}")

fps_2 = int(cap.get(cv2.CAP_PROP_FPS))
width_2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
#output_video_path = os.path.join(output_dir, f"processed_video_{video_index}.mp4")
print(f"fire_video fps : {fps_2} , width : {width_2} , height : {height_2}")
print("Finish...")
#exit()

video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width_2, height_2))

 # ROI 和疊加區域資訊
x1, y1, x2, y2 = (100, 100, 300, 400)
overlay_x, overlay_y = (0, 180)

# 處理影片的每一幀
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    be_cropped_ret, be_cropped_frame = be_cropped_cap.read()
    if not be_cropped_ret:
        print("End of video")
        break


    new_be_cropped_frame = cv2.resize(be_cropped_frame, (width_2, height_2))

    # 取得 video2 的 ROI
    roi = frame[y1:y2, x1:x2]

    # 確定 ROI 的大小是否超出 video1 範圍
    roi_height, roi_width = roi.shape[:2]
    if overlay_x + roi_width > width or overlay_y + roi_height > height:
        print("疊加範圍超出 video1 的尺寸，請調整位置或 ROI 大小。")
        break
     # 在 frame1 的指定位置貼上 ROI
    new_be_cropped_frame[overlay_y:overlay_y + roi_height, overlay_x:overlay_x + roi_width] = roi


    video_writer.write(new_be_cropped_frame)

    # Display the frame
    #resized_frame = cv2.resize(frame, (640, 400))
    cv2.imshow("Video", new_be_cropped_frame)


    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
be_cropped_cap.release()
video_writer.release()
cv2.destroyAllWindows()