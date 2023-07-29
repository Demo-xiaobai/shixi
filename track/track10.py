import cv2
from ultralytics import YOLO
import os
import numpy as np
import torch
import threading
from queue import Queue
import copy
'''cls
    1、画出单独的检测框  cv2实现 
    2、只检测单独检测框内的车辆
    3、对逆行的车辆标出警示
    4、画出正向行驶的箭头


    # 线程 
    # 检测框识别
        x 坐标大于 1 和 2的x
'''
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 画出警告红叉
def warning(img,x,y):
    '''
    函数功能

    args：
        img -> 图像 np.array
        x
        y

    return 

    ...

    '''

    # 定义符号参数
    color = (0, 0, 255)  # 红色
    thickness = 2  # 符号线条宽度

    # 绘制红色的叉号
    cv2.line(img, (x,y), (x+35, y+35), color, thickness)  #起点坐标 终点坐标
    cv2.line(img, (x, y+35), (x+35, y), color, thickness)
    cv2.circle(img, (x+20,y+20), 35, color, thickness)
    return img

def is_inside(x, y, polygon): # 判断点是否在多边形内部
    count = 0
    n = len(polygon)

    for i in range(n):
        # 获取多边形的相邻顶点坐标
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # 检查射线与边界是否相交
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            count += 1

    # 判断相交次数奇偶性
    return count % 2 == 1

# def Thread_control():


def track_goal(queue,model,points):
    global img_queue
    normal_direction_points = np.array([[406,162],[360,376]]) # 顺行方向坐标
    normal_direction_vector =  torch.from_numpy(normal_direction_points[0] - normal_direction_points[1]) # 顺行向量 AB
    while True:
        while not img_queue.empty():
                track_frame = img_queue.get() # 第一张图片 
                results = model.track(track_frame,classes=[0,2,5,7],persist=True)
                annotated_frame = results[0].plot(boxes=False,conf=False)
                cur_center_position = torch.tensor([])
                boxes = results[0].boxes  # 一直都是当前帧的box

                while not queue.empty():
                    pre_box = queue.get()  #  取出上一帧的一个检测框
                    pre_center_position = pre_box.xywh                                        # 取出前一帧一个检测框的
                    pre_car_id = pre_box.id
                    pre_position = pre_box.xyxy
                    # 上一帧的一个检测框和当前帧的检测框进行比对 计算
                    for box in boxes:   # 遍历当前帧的所有检测框
                        cur_car_id = box.id
                        cur_center_position = box.xywh
                        cur_position = box.xyxy
                        print("mmmmmmmmmmmmm-----------------------")
                        if cur_car_id == pre_car_id:
                            dY = pre_center_position[0][1] - cur_center_position[0][1]
                            reverse_vector = pre_center_position[0] - cur_center_position[0]# CD x1-x2=（0，0）
                            if (reverse_vector[0] != 0) and (reverse_vector[1] != 0): # 车辆不是静止情况下计算夹角
                                if(np.sign(dY)==-1) and ((np.abs(dY.cpu())>=2) and (np.abs(dY.cpu())<=50)):
                                    # if (np.abs(d))
                                    fenzi = (reverse_vector[0]*normal_direction_vector[0] + reverse_vector[1]*normal_direction_vector[1])
                                    fenmu = torch.sqrt(reverse_vector[0]**2 +reverse_vector[1]**2)*torch.sqrt(normal_direction_vector[0]**2+normal_direction_vector[1]**2)
                                    theta = torch.acos(fenzi/fenmu)   # 判断角度值
                                    if theta>=1 and theta<=60:  #判定逆行
                                        cv2.putText(annotated_frame, 'WRONG DIRECTION', (int(cur_position[0][0]), int(cur_position[0][1])+100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3) 
                                        warning(annotated_frame,int(cur_position[0][0]+80),int(cur_position[0][1])+10) #绘制警告图标
                        else:
                            continue


                if queue.empty(): # 针对的是 第一张的图片
                    for box in boxes:  
                        if is_inside(box.xywh[0][0],box.xywh[0][1],points): # 如果检测框的坐标在ROI的区域内就进行逆行检测                                               # 取得当前帧的一个检测框左上角和右下角坐标
                            queue.put(box)
                        else:
                            continue
                img_queue.put(annotated_frame)
    

def get_image(model_path,video_path):
    global img_queue
    cap = cv2.VideoCapture(video_path)
    # track_frame = np.array([])
    while cap.isOpened():
                success,frame = cap.read()   # frame 1080 * 1920 * 3  读取一帧
                # points = np.array([[523,258], [13, 581], [13,822], [1570, 822],[1222,258]], np.int32)         # ROI的范围坐标
                if success:
                    # track_frame = cv_frame  # 每一帧的图像
                    # # 绘图函数
                    if img_queue.empty() == True:   # 说明是当前帧
                        img_queue.put(frame)
                    else:
                        track_frame = img_queue.get() # 追踪线程执行完毕了
                    
                        cv2.polylines(track_frame, [points], isClosed=True, color=(255, 0, 0), thickness=3)       # 绘制蓝色的多边形检测区域 
                        
                        # 视频缩放
                        w,h = track_frame.shape[:2]
                        scale = 0.5
                        new_w = int(w * scale) #960
                        new_h = int(h * scale) #540
                        show_img = cv2.resize(track_frame,(new_h,new_w))  # 用opencv的resize函数，重新设置图片。

                        # 画出正确行驶路线
                        cv2.line(show_img,(406,162),(360,376),(0, 255, 0),3)
                        cv2.line(show_img,(406,162),(382,172),(0, 255, 0),3)
                        cv2.line(show_img,(406,162),(430,172),(0, 255, 0),3)

                        cv2.imshow("YOLOv8 Inference",show_img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break                
    cap.release()
    cv2.destroyAllWindows()

# global track_frame
# track_frame = np.array([])
img_queue = Queue()
if __name__ == '__main__':
    Yolo_path="E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt"      # 预训练模型路径
    video_path = "rtsp://admin:cd123456@172.18.110.51:554"                  # 视频流路径 
    model = YOLO(Yolo_path)
    queue = Queue()

    # lock = threading.Lock()
    # condition = threading.Condition(lock)
    # run_track_thread = True

    points = np.array([[523,258], [13, 581], [13,822], [1570, 822],[1222,258]], np.int32)

    cv_thread = threading.Thread(target=get_image,args=(Yolo_path,video_path))  # 读取视频帧的线程  返回出framek
    track_thread = threading.Thread(target=track_goal,args=(queue,model,points)) # 追踪处理的函数
    cv_thread.start()
    # cv_thread.join()
    track_thread.start()
    track_thread.join()







  