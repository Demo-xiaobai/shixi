import cv2
import numpy as np

# 读取视频
cap = cv2.VideoCapture('rtsp://admin:cd123456@172.18.110.51:554')

# 获取第一帧图像
ret, frame = cap.read()

# 创建窗口并命名
window_name = 'Video with ROI'
cv2.namedWindow(window_name)

# 定义ROI的多边形顶点坐标
points = np.array([[100, 50], [300, 50], [450, 200], [50, 200]])

while True:
    # 读取每一帧图像
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # 在每一帧图像上绘制ROI区域边界
    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    # 显示带有ROI区域的图像
    cv2.imshow(window_name, frame)
    
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放视频流和窗口
cap.release()
cv2.destroyAllWindows()
