import os
import cv2
import threading
from ultralytics import YOLO
import torch

# 定義推論函數
def infer_model(model, frame, results_list, lock, model_name):
    # 使用模型進行推論
    results = model.predict(frame, conf=0.5)
    # 確保多執行緒安全地寫入結果
    with lock:
        for r in results[0].boxes:
            # 提取屬性（確保存取正確的格式）
            box = r.xyxy.tolist()[0]  # 邊界框 (x1, y1, x2, y2)
            cls = int(r.cls.item())  # 類別索引
            conf = float(r.conf.item())  # 信心值
            # 存入結果列表
            results_list.append({
                "model": model_name,  # 添加模型名稱
                "class": cls,
                "confidence": conf,
                "box": box
            })

# 設定設備
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 載入模型
model_fire = YOLO('model_fire/model_collections_fire/weights/best.pt').to(device)
model_smoke = YOLO('model_smoke/model_collections_smoke/weights/best.pt').to(device)

# 初始化結果列表和執行緒鎖
results = []
lock = threading.Lock()

# 設定影片路徑
video_path = "datasets/test_fire/video1.mp4"
count_predictions = 0

# 創建資料夾
fire_dir = "save_dir/fire_predictions/"
smoke_dir = "save_dir/smoke_predictions/"
os.makedirs(fire_dir, exist_ok=True)
os.makedirs(smoke_dir, exist_ok=True)
fr_dir = "save_dir/fire_smoke_predictions/"

# 開啟影片檔案
cap = cv2.VideoCapture(video_path)

##設定暫停器，暫停存圖片
#ex:每秒30frmaes，設定300代表暫停10秒
fire_count_pause = 300
smoke_count_pause = 300

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

##讀取圖片每秒幾個frame
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

# 處理影片的每一幀
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # 影片結束

    count_predictions = count_predictions + 1

    results_list = []
    smoke_count_pause = smoke_count_pause + 1
    fire_count_pause = fire_count_pause + 1

    # 建立執行緒來同時推論火災和煙霧
    thread1 = threading.Thread(target=infer_model, args=(model_fire, frame, results_list, lock, "fire"))
    thread2 = threading.Thread(target=infer_model, args=(model_smoke, frame, results_list, lock, "smoke"))

    # 開始執行緒
    thread1.start()
    thread2.start()

    # 等待執行緒完成
    thread1.join()
    thread2.join()

    # 創建兩個副本來存儲分開的結果
    smoke_frame = frame.copy()  # 儲存煙霧的副本
    fire_frame = frame.copy()   # 儲存火災的副本

    # 合併顯示：繪製火災和煙霧的結果在同一張圖片上
    for r in results_list:
        model_name = r['model']
        x1, y1, x2, y2 = r['box']
        confidence = r['confidence']
        label = f"Class: {model_name}, Conf: {confidence:.2f}"

        # 合併顯示在原始畫面上
        if model_name == "smoke":
            color = (0, 0, 255)  # 煙霧結果使用紅色
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        elif model_name == "fire":
            color = (0, 255, 0)  # 火災結果使用綠色
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # image_path = os.path.join(fr_dir, f"processed_{count_predictions:04d}.jpg")
        # cv2.imwrite(image_path, frame)  # 儲存火災圖片

        # 儲存煙霧的偵測結果
        if model_name == "smoke":
            cv2.rectangle(smoke_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
            cv2.putText(smoke_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            if smoke_count_pause >= 300:
                smoke_image_path = os.path.join(smoke_dir, f"processed_{count_predictions:04d}.jpg")
                cv2.imwrite(smoke_image_path, smoke_frame)  # 儲存煙霧圖片
                smoke_count_pause = 0

        # 儲存火災的偵測結果
        if model_name == "fire":
            cv2.rectangle(fire_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
            cv2.putText(fire_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            if fire_count_pause >= 300:
                fire_image_path = os.path.join(fire_dir, f"processed_{count_predictions:04d}.jpg")
                cv2.imwrite(fire_image_path, fire_frame)  # 儲存火災圖片
                fire_count_pause = 0

    # 每張frame(包含未偵測)都會export
    # # 儲存煙霧和火災的結果到各自的資料夾
    # smoke_image_path = os.path.join(smoke_dir, f"processed_{count_predictions:04d}.jpg")
    # fire_image_path = os.path.join(fire_dir, f"processed_{count_predictions:04d}.jpg")

    # # 儲存煙霧偵測結果圖片
    # cv2.imwrite(smoke_image_path, smoke_frame)  # 儲存煙霧圖片

    # # 儲存火災偵測結果圖片
    # cv2.imwrite(fire_image_path, fire_frame)  # 儲存火災圖片


    
    # 顯示合併後的影像
    cv2.imshow("Video", frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()