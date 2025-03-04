import cv2

multiTracker = cv2.legacy.MultiTracker_create()  # 建立多物件追蹤器
tracking = False                                 # 設定追蹤尚未開始
colors = [(0,0,255)]                 # 建立外框色彩清單

cap = cv2.VideoCapture("videos/十二手套沒帶.mp4")                        # 讀取攝影鏡頭

if not cap.isOpened():
    print("Cannot open camera")
    exit()



while True:
  ret, frame = cap.read()
  if not ret:
    print("Cannot receive frame")
    break
  
  # cv2.imshow('oxxostudio', frame)  
  # # 等待 1 毫秒，模擬影片播放效果，按下 `q` 可退出
  # if cv2.waitKey(1) & 0xFF == ord('q'):
  #   cap.release()
  #   cv2.destroyAllWindows()
  #   exit()
  
  frame = cv2.resize(frame,(800,630))         # 縮小尺寸加快速度
  keyName = cv2.waitKey(50)

  if keyName == ord('q'):
    break
  # 按下 a 的時候開始標記物件外框
  if keyName == ord('a'):
    for i in range(1):
      area = cv2.selectROI('oxxostudio', frame, showCrosshair=False, fromCenter=False)
      # 標記外框後設定該物件的追蹤演算法
      tracker = cv2.legacy.TrackerCSRT_create()
      # 將該物件加入 multiTracker
      multiTracker.add(tracker, frame, area)
      # 設定 True 開始追蹤
    tracking = True
  if tracking:
    # 更新 multiTracker
    success, points = multiTracker.update(frame)
    a = 0
    if success:
      for i in  points:
        p1 = (int(i[0]), int(i[1]))
        p2 = (int(i[0] + i[2]), int(i[1] + i[3]))
        # 標記物件外框
        cv2.rectangle(frame, p1, p2, colors[a], 3)
        a = a + 1
  cv2.imshow('oxxostudio', frame)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    cap.release()
    cv2.destroyAllWindows()
    exit()

cap.release()
cv2.destroyAllWindows()