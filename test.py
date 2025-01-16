from ultralytics import YOLO
import torch
import cv2
import os
import time
import threading
import time

# 設定設備
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 載入模型
#model_fire = YOLO('model_smoke/model_collections_smoke/weights/best.pt').to(device) #fire
model = YOLO("trained_model/best_10_modify.pt").to(device) # 10

cap = cv2.VideoCapture("videos/有戴護目鏡.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# testPath = "datasets/test_cellphone/images/"

# testFilePath = os.listdir(testPath)
# test_record = []

# for file in testFilePath:
#     test_record.append(file)
# #print(train_record)

# test_final = []
# for i in range(len(test_record)):
#     imagepath = "datasets/test_cellphone/images/"+test_record[i]+""
#     #img = cv2.imread("D:/AI/QTR_eden/QTR/dataset/train/"++)

#     img = cv2.imread(imagepath)

#     #print(img.shape)
#     # cv2.imshow("window_name", img)

#     # cv2.waitKey(0)

#     # # closing all open windows
#     # cv2.destroyAllWindows()


#     result = model(source=img,
#                   show=True, conf=0.5) 
#     cv2.waitKey(0)


# cap = cv2.VideoCapture("videos/有戴護目鏡.mp4")

# if not cap.isOpened():
#     print("Cannot open camera")
#     exit()
    
count = 0
count_predictions = 0
batch_size = 10  # 批次大小
batch_frames = []  # 暫存多幀影像


#################################################
### store video
output_dir = "output_videos"
# Get video properties
# example shape is (1080, 1920, 3)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 files
output_video_path = os.path.join(output_dir, f"alert_video.mp4")
#print(f"video fps : {fps} , width : {width} , height : {height}")

video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#################################################

##for 一次一張圖
#fps = 1 #跳過偵數  
##

# ### 一次一個frame
# ## 有分pop(較慢) 優點:ori & predictions images都會完整保存 / 清空frame再append(快) 缺點:最後幾個frame不會存到
# while True:
#     ##for 一次一張圖
#     #frame_index = int(count * fps)
#     #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # 設定讀取位置
#     ##
    
#   ret, frame = cap.read()             # 讀取影片的每一幀
#   if not ret:
#     print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
#     break
#   count = count + 1
    
#   batch_frames.append(frame)
    
#   # ori_image_path = os.path.join("save_dir/ori_image/", f"{count:04d}.jpg")
#   # cv2.imwrite(ori_image_path, frame)

#   if len(batch_frames) == batch_size:

#     results = model.predict(source = batch_frames,show=False, conf=0.5,max_det=2,
#                         verbose=False)

      
#     # processed_frame = results[0].plot()  # 提取可視化結果(單偵圖片結果)
#     # processed_image_path = os.path.join("save_dir/predictions/", f"processed_{count:04d}.jpg")
#     # cv2.imwrite(processed_image_path, processed_frame)
    
    
#     ############################ batch_frame的結構，但是在write predictions的時候一樣一張一張圖片寫
#     # processed_frame = results[-1].plot()
#     # count_predictions = count_predictions + 1
#     # cv2.imshow("0000000000000", processed_frame)
      
#     # # processed_image_path = os.path.join("save_dir/predictions/", f"processed_{count:04d}.jpg")
#     # # cv2.imwrite(processed_image_path, processed_frame)
#     # video_writer.write(processed_frame)


#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #             cap.release()
#     #             cv2.destroyAllWindows()
#     #             exit()
#     ############################
    
    
#     # #frame批次處理/偵測
#     # for result in results:
#     #   processed_frame = result.plot()  # 提取可視化結果
#     #   count_predictions = count_predictions + 1
#     #   cv2.imshow("0000000000000", processed_frame)
#     #   # processed_image_path = os.path.join("save_dir/predictions/", f"processed_{count_predictions:04d}.jpg")
#     #   # cv2.imwrite(processed_image_path, processed_frame)
            
#     #   # 等待 1 毫秒，模擬影片播放效果，按下 `q` 可退出
#     #   if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     cap.release()
#     #     cv2.destroyAllWindows()
#     #     exit()
    
#     batch_frames.pop()
    
# cap.release()

               
"""
#速度慢 一張一張存&預測  (優點:ori & predictions images都會完整保存 , 缺點:慢)

#可以存原圖跟預測結果
#單一張frame predictions
while True:

    
  #start_time = time.time()
    
  ret, frame = cap.read()             # 讀取影片的每一幀
  if not ret:
      print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
      break
  count = count + 1
    
    
  ori_image_path = os.path.join("save_dir/ori_image/", f"{count:04d}.jpg")
  cv2.imwrite(ori_image_path, frame)
    
  # 記錄影片讀取耗時
  #read_time = time.time()
  #print(f"Frame {count} - Video read time: {read_time - start_time:.4f} seconds")

    
  #cv2.imshow('oxxostudio', frame)     # 如果讀取成功，顯示該幀的畫面
  #cv2.imwrite("save_dir/ori_image"+str(count)+".jpg", frame)
    
    
  results = model.predict(source = frame,show=True, conf=0.5)
  #cv2.imwrite("save_dir/predictions/"+str(count)+".jpg", frame)
    
    
    
  #inference_time = time.time()
  #print(f"Frame {count} - YOLO inference time: {inference_time - read_time:.4f} seconds")


  processed_frame = results[0].plot()  # 提取可視化結果(單偵圖片結果)
  processed_image_path = os.path.join("save_dir/predictions/", f"processed_{count:04d}.jpg")
  cv2.imwrite(processed_image_path, processed_frame)
    
    
  #write_time = time.time()
  #print(f"Frame {count} - YOLO write time: {write_time - inference_time:.4f} seconds")
    
  #total_time = time.time()
  #print(f"Frame {count} - Total processing time: {total_time - start_time:.4f} seconds")
    
cap.release()
"""



# save specific folder
# result = model(source=f"videos/沒戴手套擦拭.mp4",
#                show=True, conf=0.5, save=True, save_frames=True, project="save_dir", name="result_ten")  


# result = model_fire(source=f"datasets/test_fire/processed_video_0.mp4",
#                show=True, conf=0.75)  




#################################################
### yolov8 object tracking constantly record id times of appearance independly

def draw_text(img, text,
          font=cv2.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size
    
batch_size = 10  # 批次大小
batch_frames = []  # 暫存多幀影像

last_alert_times = {}
alert_cooldown = 1
dictionary = {"0.0":"no gloves","1.0":"without goggles"}

cooldown_trackers = {
    "no gloves": 0, #0代表不是冷卻中
    "without goggles": 0
}  # 記錄每個類別的冷卻時間

color_dic = {"no gloves":(0, 255, 0),"without goggles":(0, 0, 255)}

pause_interval = 120

cv2.namedWindow('Object Detected', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detected', 800, 600)
### 一次一個frame
## 有分pop(較慢) 優點:ori & predictions images都會完整保存 / 清空frame再append(快) 缺點:最後幾個frame不會存到
while True:
    ret, frame = cap.read()  # 讀取影片的每一幀
    if not ret:
        print("Cannot receive frame")  # 如果讀取錯誤，印出訊息
        break

    batch_frames.append(frame)

    if len(batch_frames) == batch_size:
        # 更新冷卻時間
        for key in cooldown_trackers:
            if cooldown_trackers[key] > 0:
                cooldown_trackers[key] -= 1

        # 處理最新幀
        results = model(source=batch_frames[-1], show=False, conf=0.85, verbose=False)

        processed_frame = batch_frames[-1].copy()  # 保留原始影像作為底層
        input_frame = batch_frames[-1].copy()  # 保留原始影像作為底層
        detected_classes = set()  # 用於追蹤本次偵測到的類別

        # Frame 批次處理/偵測
        for result in results:
            # track_ids = result.boxes.id  # 追蹤 ID
            classes = result.boxes.cls  # 類別索引
            # processed_frame = result.plot()


            boxes = result.boxes.xyxy  # 矩形框 (x1, y1, x2, y2)
            scores = result.boxes.conf  # 信心分數

            # 自定義顏色和邊框厚度
            color = (0, 255, 0)  # 綠色
            thickness = 5  # 邊框厚度



            h = 0
            x, y = (100, 100)
            classes_len = len(classes.tolist())
            for i in classes.tolist():
                class_name = dictionary[str(i)]
                detected_classes.add(class_name)

                classes_len -= 1
                # 僅顯示未在冷卻期間的類別
                if cooldown_trackers[class_name] == 0:
                    
                    draw_text(processed_frame, f"class : {class_name}", pos=(x, y + h))
                    h += 30

                    # 啟動該類別的冷卻計時
                    cooldown_trackers[class_name] = alert_cooldown * pause_interval

                    #當1偵裡的draw_text都畫完後，最後再一次暫停寫影片跟顯示圖片(一次性)
                    if classes_len == 0:
                        
                        # 篩選出不在冷卻期間的框
                        mask = [dictionary[str(cls)] not in cooldown_trackers or cooldown_trackers[dictionary[str(cls)]] == alert_cooldown* pause_interval
                                for cls in classes.tolist()]
                        # 更新結果物件中的數據

                        # 篩選出符合條件的框
                        filtered_boxes = boxes[mask]
                        filtered_classes = classes[mask]
                        filtered_scores = scores[mask]

                        # 可以使用 OpenCV 自定義繪製框
                        for box, cls, score in zip(filtered_boxes, filtered_classes, filtered_scores):
                            print(f"box :{box},cls :{cls},score :{score}")
                            x1, y1, x2, y2 = box
                            class_name = dictionary[str(float(cls))]
                            # 繪製矩形框
                            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_dic[class_name], thickness)
                            # 添加標註
                            label = f"{class_name} {score:.2f}"
                            cv2.putText(processed_frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_PLAIN, 5, color_dic[class_name], 5)

                        # 使用篩選後的結果繪製
                        #processed_frame = result.plot()
                        
                        pause_time = time.time()
                        while time.time() - pause_time < alert_cooldown:
                            video_writer.write(processed_frame)  # 在暫停期間重複寫入當前幀
                            cv2.imshow("Object Detected", processed_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                cap.release()
                                video_writer.release()
                                cv2.destroyAllWindows()
                                exit()
                              

        cv2.imshow("Object Detected", input_frame)
        video_writer.write(input_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            video_writer.release()
            cv2.destroyAllWindows()
            exit()

        batch_frames.pop()  # 移除處理過的幀

 

cap.release()
video_writer.release()
cv2.destroyAllWindows()
#################################################


##pose detect
#results = model_fire("videos/有戴護目鏡.mp4",show=True)

##寫變數到文字檔
# count = 0
# with open("test.txt", 'w') as f:
#     f.write(str(record))


# while True:
#     ret, frame = cap.read()  # 讀取影片的每一幀
#     if not ret:
#         print("Cannot receive frame")  # 如果讀取錯誤，印出訊息
#         break

#     batch_frames.append(frame)

#     if len(batch_frames) == batch_size:
#         # 更新冷卻時間
#         for key in cooldown_trackers:
#             if cooldown_trackers[key] > 0:
#                 cooldown_trackers[key] -= 1

#         # 處理最新幀
#         results = model(source=batch_frames[-1], show=False, conf=0.85, verbose=False)

#         processed_frame = batch_frames[-1].copy()  # 保留原始影像作為底層
#         input_frame = batch_frames[-1].copy()  # 保留原始影像作為底層
#         detected_classes = set()  # 用於追蹤本次偵測到的類別

#         # Frame 批次處理/偵測
#         for result in results:
#             # track_ids = result.boxes.id  # 追蹤 ID
#             classes = result.boxes.cls  # 類別索引
#             processed_frame = result.plot()


#             h = 0
#             x, y = (100, 100)
#             classes_len = len(classes.tolist())
#             for i in classes.tolist():
#                 class_name = dictionary[str(i)]
#                 detected_classes.add(class_name)

#                 classes_len -= 1
#                 # 僅顯示未在冷卻期間的類別
#                 if cooldown_trackers[class_name] == 0:
                    
#                     draw_text(processed_frame, f"class : {class_name}", pos=(x, y + h))
#                     h += 30

#                     # 啟動該類別的冷卻計時
#                     cooldown_trackers[class_name] = alert_cooldown * 30

#                     #當1偵裡的draw_text都畫完後，最後再一次暫停寫影片跟顯示圖片(一次性)
#                     if classes_len == 0:
                    
#                         pause_time = time.time()
#                         while time.time() - pause_time < alert_cooldown:
#                             video_writer.write(processed_frame)  # 在暫停期間重複寫入當前幀
#                             cv2.imshow("Object Detected", processed_frame)
#                             if cv2.waitKey(1) & 0xFF == ord('q'):
#                                 cap.release()
#                                 video_writer.release()
#                                 cv2.destroyAllWindows()
#                                 exit()
                              

#         cv2.imshow("Object Detected", input_frame)
#         video_writer.write(input_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cap.release()
#             video_writer.release()
#             cv2.destroyAllWindows()
#             exit()

#         batch_frames.pop()  # 移除處理過的幀

 

# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()
