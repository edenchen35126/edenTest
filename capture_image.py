import cv2

vc = cv2.VideoCapture("demo_0113/11課pos.mp4")
c = 0
if not vc.isOpened() :
    print("Error: Unable to open video.")
    exit()

timeF = 15
##讀取圖片每秒幾個frame
# fps = vc.get(cv2.CAP_PROP_FPS)
# print(f"Frames per second: {fps}")

while True:
    ret , frame = vc.read()

    if not ret:
        print("Cannot receive frame")   # 如果讀取錯誤，印出訊息
        break
    if (c % timeF == 0):
        cv2.imwrite(f"demo_0113/11/output_11_{str(c)}.jpg",frame)
    c = c + 1

cv2.waitKey(1)
vc.release()

