import cv2
from ultralytics import YOLO
import os
import numpy as np
import torch
from queue import Queue
'''
    1、画出单独的检测框  cv2实现  
    2、只检测单独检测框内的车辆
    3、对逆行的车辆标出警示
    4、画出正向行驶的箭头
'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def warning():
    # 创建一个空白图像
    image = np.zeros((500, 500, 3), dtype=np.uint8)

    # 定义符号参数
    color = (0, 0, 255)  # 红色
    thickness = 2  # 符号线条宽度

    # 绘制红色的叉号
    cv2.line(image, (0, 0), (50, 50), color, thickness)  #起点坐标 终点坐标
    cv2.line(image, (0, 50), (50, 0), color, thickness)
    cv2.circle(image, (25,25), 25, color, thickness)
    return image

def reverse_Track(model_path,video_path):
    
    position_list = []      #记录所有检测框的 左上角 和 右下角的坐标
    mid_position_list = []   #记录所有检测框的 中心点坐标和 宽高
    cars_id = []            #记录所有检测框的ID

    queue = Queue()         #什么一个队列  往里装


    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        points = np.array([[523,258], [13, 581], [13,822], [1570, 822],[1222,258]], np.int32)         # ROI的范围大小
        success,frame = cap.read()   #frame 1080 * 1920 * 3  读取一帧
        if success:
        #设置感兴趣的ROI  
            mask = np.zeros(frame.shape[:2], dtype=np.uint8) #定义和图像大小一样的遮罩
            cv2.fillPoly(mask, [points], 255)# 在遮罩上绘制ROI区域
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=3) #绘制多边形
            roi = cv2.bitwise_and(frame, frame, mask=mask)         # 将遮罩应用于原始图像  1要检测的区域

            results = model.track(roi,classes=[2,5,7])   #跟踪
            boxes = results[0].boxes
            
            pre_boxs = []  #存储之前帧的检测框到队列中
            while not queue.empty():
                pre_boxs.append(queue.get())

            #当前的检测框的 坐标 id
            cur_center_position = torch.tensor([])
            cur_car_id = torch.tensor([])
            cur_position= torch.tensor([])
            dX = 0
            dY = 0
            for box in boxes:
                cur_center_position = box.xywh
                cur_car_id = box.id
                cur_position = box.xyxy
                queue.put(box)
                for pre_box in pre_boxs:    # 取出队里中的一个检测框
                    pre_center_position = pre_box.xywh
                    pre_car_id = pre_box.id
                    pre_position = pre_box.xyxy
                    # if pre_car_id == cur_car_id:   # 同一ID的检测目标
                    dX = pre_center_position[0][0] - cur_center_position[0][0]
                    dY = pre_center_position[0][1] - cur_center_position[0][1]
                    '''
                    if np.abs(dY) >= 12 and np.sign(dY) == -1:
                        dirX = "South"
                        cv2.putText(new_image, 'WRONG DIRECTION', (int(cur_position[0][2]), int(cur_position[0][3])), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1) 
                    else:
                        cv2.putText(new_image, 'GOOD', (int(cur_position[0][2]), int(cur_position[0][3])), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1) 
                    '''
            annotated_frame = results[0].plot(boxes=False,conf=False)
            
            # cv2.putText(annotated_frame, 'WRONG DIRECTION', (int(cur_position[0][0])-25, int(cur_position[0][1])-25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1) 
            # cv2.putText(annotated_frame, 'WRONG DIRECTION', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1)
            new_image = cv2.bitwise_or(annotated_frame,frame) #roi 和 当前图像求并集
            if np.abs(dY)>=10 and np.sign(dY) == -1:
                cv2.putText(new_image, 'WRONG DIRECTION', (int(cur_position[0][2]), int(cur_position[0][3])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2) 
                # cv2.
                # cv2.imshow("",warning())
            # 缩小显示的图像
            w,h = new_image.shape[:2]
            scale = 0.5
            new_w = int(w * scale) #960
            new_h = int(h * scale) #540
            show_img = cv2.resize(new_image,(new_h,new_w))  # 用opencv的resize函数，重新设置图片。
            cv2.imshow("YOLOv8 Inference",show_img)
            # cv2.imshow("Test......",roi)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Yolo_path="E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt"      # 预训练模型路径
    video_path = "rtsp://admin:cd123456@172.18.110.51:554"                  # 视频流路径 
    reverse_Track(Yolo_path,video_path)      