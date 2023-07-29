import cv2
from ultralytics import YOLO
import os
import numpy as np
import torch
from queue import Queue
'''cls
    1、画出单独的检测框  cv2实现  xls
    2、只检测单独检测框内的车辆
    3、对逆行的车辆标出警示
    4、画出正向行驶的箭头
'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def warning(img,x,y):       # 画×

    # 定义符号参数
    color = (0, 0, 255)  # 红色
    thickness = 4  # 符号线条宽度

    # 绘制红色的叉号
    cv2.line(img, (x,y), (x+50, y+50), color, thickness)  #起点坐标 终点坐标
    cv2.line(img, (x, y+50), (x+50, y), color, thickness)
    cv2.circle(img, (x+25,y+25), 35, color, thickness)
    return img

def reverse_Track(model_path,video_path):

    queue = Queue()         #队列放入每一帧的所有检测框
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0# 技数帧率
    process = 5
    while cap.isOpened():
        points = np.array([[523,258], [13, 581], [13,822], [1570, 822],[1222,258]], np.int32)         # ROI的范围大小
        success,frame = cap.read()   #frame 1080 * 1920 * 3  读取一帧
        if success:
        #设置感兴趣的ROI  
            mask = np.zeros(frame.shape[:2], dtype=np.uint8) #定义和图像大小一样的遮罩
            cv2.fillPoly(mask, [points], 255)# 在遮罩上绘制ROI区域
            cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=3) #绘制多边形
            roi = cv2.bitwise_and(frame, frame, mask=mask)         # 将遮罩应用于原始图像  要检测的区域

            results = model.track(roi,classes=[0,2,5,7],persist=True)             # 跟踪
            boxes = results[0].boxes

            pre_boxs = []                                          # 存储前一帧的检测框到队列中
            while not queue.empty():
                pre_boxs.append(queue.get())                       #将前一帧的检测框取出 并放入列表 pre_boxs中

            cur_center_position = torch.tensor([])
            cur_position= torch.tensor([])
            dX = 0
            dY = 0  # 记录坐标差 来判定是否逆行
            distance = 0
            draw_warinig = []
            # frame_count+=1
            # if (frame_count % process)==0:

            # 获取当前帧的检测框
            for box in boxes:
                cur_center_position = box.xywh      # 取得当前帧的检测框
                cur_car_id = box.id
                cur_position = box.xyxy             #
                queue.put(box)
                for pre_box in pre_boxs:    # 取出前一帧队列里中的一个检测框
                    pre_center_position = pre_box.xywh
                    pre_position = pre_box.xyxy
                    pre_car_id = pre_box.id

        # 记录下逆行的检测框 放入集合中  然后在全部画警告信息

                    if pre_car_id == cur_car_id:   # 同一ID的检测目标
                        dX = pre_center_position[0][0] - cur_center_position[0][0]
                        dY = pre_center_position[0][1] - cur_center_position[0][1]  #计算 前一帧和后一帧的 y轴坐标差
                            # 计算欧式距离
                        distance = torch.sqrt(dX**2+dY**2)
                            # 判断是否逆行
                        if np.sign(dY) == -1  and (np.abs(dY)>=12 and np.abs(dY)<=20):   #np.sign(dY) == -1 表示 < 0 逆行方向
                            if np.abs(dX)>=0 and np.abs(dX)<=25:
                                cv2.putText(roi, 'WRONG DIRECTION', (int(cur_position[0][0]), int(cur_position[0][1])+100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3) 
                                warning(roi,int(cur_position[0][0]),int(cur_position[0][1])) #绘制警告图标
                                # draw_warinig.append(pre_box)  # 出现逆行的检测框放入集合中
                    else:
                        continue
            # for bbox in draw_warinig:
                        # cv2.putText(roi, 'WRONG DIRECTION', (int(bbox.xyxy[0][0]), int(bbox.xyxy[0][1])+100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3) 
                        # warning(roi,int(bbox.xyxy[0][0]),int(bbox.xyxy[0][1])) #绘制警告图标
            annotated_frame = results[0].plot(boxes = False,conf=False)
            new_image = cv2.bitwise_or(annotated_frame,frame) # annotated_frame是追踪的roi 和 源图像求并集

            w,h = new_image.shape[:2]
            scale = 0.5
            new_w = int(w * scale) #960
            new_h = int(h * scale) #540
            show_img = cv2.resize(new_image,(new_h,new_w))  # 用opencv的resize函数，重新设置图片。

            cv2.line(show_img,(406,162),(360,376),(0, 255, 0),3)
            cv2.line(show_img,(406,162),(382,172),(0, 255, 0),3)
            cv2.line(show_img,(406,162),(430,172),(0, 255, 0),3)

            cv2.imshow("YOLOv8 Inference",show_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Yolo_path="E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt"      # 预训练模型路径
    # video_path = "rtsp://admin:cd123456@172.18.110.51:554"                  # 视频流路径 
    video_path = "E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\predit_video.mp4"
    reverse_Track(Yolo_path,video_path)      