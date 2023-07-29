import cv2
from ultralytics import YOLO
import os
import numpy as np
'''
    1、画出单独的检测框  cv2实现  
    2、只检测单独检测框内的车辆
    3、对逆行的车辆标出警示
    4、画出正向行驶的箭头
'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def reverse_Track(model_path,video_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        points = np.array([[523,376], [13, 581], [13,822], [1919, 822],[1522,376]], np.int32)         # ROI的范围大小
        success,frame = cap.read()   #frame 1080 * 1920 * 3
        if success:
        #设置感兴趣的ROI  
            mask = np.zeros(frame.shape[:2], dtype=np.uint8) #定义和图像大小一样的遮罩
            cv2.fillPoly(mask, [points], 255)# 在遮罩上绘制ROI区域
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=3)
            roi = cv2.bitwise_and(frame, frame, mask=mask)         # 将遮罩应用于原始图像  1要检测的区域


            # # cv2.rectangle(img=frame,pt1=(x,y),pt2=(x+w,y+h),color=(0,0,255),thickness=2)      # 绘制矩形
            
            # x, y, w, h = (100, 100, 400, 600)  # 以示例方式指定ROI的位置和大小
            # cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
            # #感兴趣区域ROI融合

            results = model.track(roi,persist=True)   #跟踪
            # results = results.cuda()
           

            annotated_frame = results[0].plot()
            new_image = cv2.bitwise_or(annotated_frame,frame)

            # w,h = annotated_frame.shape[:2]
            # scale = 0.5
            # new_w = int(w * scale) #960
            # new_h = int(h * scale) #540

            # show_img = cv2.resize(annotated_frame,(new_h,new_w))  # 用opencv的resize函数，重新设置图片。

            # cv2.imshow("YOLOv8 Inference",show_img)
            cv2.imshow("roi",new_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Yolo_path="E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt"      # 预训练模型路径
    video_path = "rtsp://admin:cd123456@172.18.110.51:554"                  # 视频流路径 
    reverse_Track(Yolo_path,video_path)      