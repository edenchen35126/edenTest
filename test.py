from ultralytics import YOLO
import torch
import cv2
import os
import time


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = YOLO('trained_model/best_10_modify.pt').to(device)  ###



cap = cv2.VideoCapture("videos/沒戴護目鏡.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
count = 0
count_predictions = 0
batch_size = 10  # 批次大小
batch_frames = []  # 暫存多幀影像

##for 一次一張圖
#fps = 1 #跳過偵數  
##

### 一次一個frame
## 有分pop(較慢) 優點:ori & predictions images都會完整保存 / 清空frame再append(快) 缺點:最後幾個frame不會存到
while True:
    ##for 一次一張圖
    #frame_index = int(count * fps)
    #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)  # 設定讀取位置
    ##
    
  ret, frame = cap.read()             # 讀取影片的每一幀
  if not ret:
    print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
    break
  count = count + 1
    
  batch_frames.append(frame)
    
  ori_image_path = os.path.join("save_dir/ori_image/", f"{count:04d}.jpg")
  cv2.imwrite(ori_image_path, frame)

  if len(batch_frames) == batch_size:

    results = model.predict(source = batch_frames,show=False, conf=0.5,max_det=2,
                        verbose=False)

      
    # processed_frame = results[0].plot()  # 提取可視化結果(單偵圖片結果)
    # processed_image_path = os.path.join("save_dir/predictions/", f"processed_{count:04d}.jpg")
    # cv2.imwrite(processed_image_path, processed_frame)
    
    
    ############################ batch_frame的結構，但是在write predictions的時候一樣一張一張圖片寫
    # processed_frame = results[-1].plot()
    # count_predictions = count_predictions + 1
    # cv2.imshow("0000000000000", processed_frame)
      
    # processed_image_path = os.path.join("save_dir/predictions/", f"processed_{count:04d}.jpg")
    # cv2.imwrite(processed_image_path, processed_frame)
      
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #             cap.release()
    #             cv2.destroyAllWindows()
    #             exit()
    ############################
    
    
    #frame批次處理/偵測
    for result in results:
      processed_frame = result.plot()  # 提取可視化結果
      count_predictions = count_predictions + 1
      cv2.imshow("0000000000000", processed_frame)
      processed_image_path = os.path.join("save_dir/predictions/", f"processed_{count_predictions:04d}.jpg")
      cv2.imwrite(processed_image_path, processed_frame)
            
      # 等待 1 毫秒，模擬影片播放效果，按下 `q` 可退出
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    
    batch_frames = []
    
cap.release()

               
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

"""
result = model(source=f"videos/有戴護目鏡.mp4",
               show=True, conf=0.5)  """

