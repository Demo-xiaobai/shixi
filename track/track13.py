import cv2
from ultralytics import YOLO
import os
import numpy as np
import torch
import threading
from queue import Queue
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def warning(img,x,y):  # 画出警告红叉
    # 定义符号参数
    color = (0, 0, 255)     # 红色
    thickness = 3           # 符号线条宽度

    # 绘制红色的叉号
    cv2.line(img, (x,y), (x+35, y+35), color, thickness)  #起点坐标 终点坐标
    cv2.line(img, (x, y+35), (x+35, y), color, thickness)
    cv2.circle(img, (x+20,y+20), 35, color, thickness)
    return img

def is_inside(x, y, polygon): # 判断点是否在多边形内部
    count = 0
    # n = len(polygon)

    for i in range(5):
        # 获取多边形的相邻顶点坐标
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % 5]

        # 检查射线与边界是否相交
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            count += 1

    # 判断相交次数奇偶性
    return count % 2 == 1               # 奇数点在内部    偶数表示点在外部

def track_goal(model,points):
    global queue
    global frame_list 
    global stop                                              # 线程cv提取的每张图片
    # normal_direction_points = np.array([[406,162],[360,376]])       # 顺行方向坐标   
    normal_direction_points = np.array([[252,212],[452,212]])
    normal_direction_vector =  torch.from_numpy(normal_direction_points[0] - normal_direction_points[1]) # 顺行向量 AB
    while True:
        if np.size(frame_list) != 0:
                results = model.track(frame_list,classes=[0,2,5,7],persist=True)
                annotated_frame = results[0].plot(boxes=False,conf=False)
                boxes = results[0].boxes  # 一直都是当前帧的box

                while not queue.empty():
                    pre_box = queue.get()                                     # 取出前一帧的一个检测框
                    pre_center_position = pre_box.xywh                        
                    pre_car_id = pre_box.id
                    
                    # 上一帧的一个检测框和当前帧的检测框进行比对 计算
                    for box in boxes:   # 遍历当前帧的所有检测框
                        cur_car_id = box.id
                        cur_center_position = box.xywh
                        cur_position = box.xyxy
                        if (cur_car_id == pre_car_id):
                            dY = pre_center_position[0][1] - cur_center_position[0][1] # y
                            dX = pre_center_position[0][0] - cur_center_position[0][0] # x
                            reverse_vector = pre_center_position[0] - cur_center_position[0]    # 向量CD (x1,y1)-(x2,y2)  方向向下
                            
                            # if(torch.sign(dY)==-1) and ((torch.abs(dY)>=1) and (torch.abs(dY)<=40)):
                            if(torch.sign(dX)==-1) and (torch.abs(dX)>=6)and(torch.abs(dX)<=80):
                                if  ((torch.abs(dY)>=0) and (torch.abs(dY)<=5)) :
                                # if(((torch.sign(dX)==1) or (torch.sign(dX)==0)) )and ((torch.abs(dX)>=2) and (torch.abs(dX)<=25)) :
                                # 用公式计算向量夹角
                                    fenzi = (reverse_vector[0]*normal_direction_vector[0] + reverse_vector[1]*normal_direction_vector[1])
                                    fenmu = torch.sqrt(reverse_vector[0]**2 +reverse_vector[1]**2)*torch.sqrt(normal_direction_vector[0]**2+normal_direction_vector[1]**2)
                                    theta = torch.acos(fenzi/fenmu)   # 判断角度值
                                    if (torch.abs(theta)>=0) and (torch.abs(theta)<=12):  #判定逆行
                                        cv2.putText(annotated_frame, 'WRONG DIRECTION', (int(cur_position[0][0]), int(cur_position[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3) 
                                        warning(annotated_frame,int(cur_center_position[0][0]),int(cur_center_position[0][1])) # 绘制警告图标
                        else:
                            continue    # 不是同一个ID 就跳过循环 检测下一个框

                if queue.empty():   
                    for box in boxes:  
                        if is_inside(box.xywh[0][0],box.xywh[0][1],points)==True:   # 如果检测框的坐标在ROI的区域内就进行逆行检测                                               # 取得当前帧的一个检测框左上角和右下角坐标
                            queue.put(box) # 把当前帧的所有检测框入队
                        else:
                           continue    # 过滤掉不在ROI区域内的检测框

                cv2.polylines(annotated_frame, [points], isClosed=True, color=(255, 0, 0), thickness=3)       # 绘制蓝色ROI区域 
                
                # 视频缩放
                w,h = annotated_frame.shape[:2]
                scale = 0.5
                new_w = int(w * scale) #960
                new_h = int(h * scale) #540
                show_img = cv2.resize(annotated_frame,(new_h,new_w))  # 用opencv的resize函数，重新设置图片。
                

                # 画出正确行驶路线
                # cv2.line(show_img,(406,162),(360,376),(0, 255, 0),3)
                # cv2.line(show_img,(406,162),(382,172),(0, 255, 0),3)
                # cv2.line(show_img,(406,162),(430,172),(0, 255, 0),3)

                cv2.line(show_img,(252,212),(583,212),(0,255,0),2)
                cv2.line(show_img,(252,212),(282,198),(0,255,0),2)
                cv2.line(show_img,(252,212),(282,238),(0,255,0),2)
                
                
                # video_writer = cv2.VideoWriter('outputfile.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (640, 480))


                cv2.imshow("YOLOv8 Inference",show_img)               
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop = False
                    cv2.destroyAllWindows()
                    return

                
def get_image(model_path,video_path):
    global frame_list
    global stop
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success,frame = cap.read()   # frame 1080 * 1920 * 3  读取一帧
        if success and stop == True:
            time.sleep(0.00004)  # 睡一会 等待track线程执行完毕
            frame_list = frame                        
        else:
            return
                

if __name__ == '__main__':
    Yolo_path="E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt"      # 预训练模型路径
    video_path = "rtsp://admin:cd123456@172.18.110.51:554"                  # 视频流路径 
    
    frame_list = np.array([])
    model = YOLO(Yolo_path)
    queue = Queue()
    stop = True
    output = "outputfile.mp4"
    frame_rate = 30.0  # 视频的帧率
    points = np.array([[523,258], [13, 581], [13,822], [1570, 822],[1222,258]], np.int32)

    cv_thread = threading.Thread(target=get_image,args=(Yolo_path,video_path))  # 读取视频帧的线程  返回出framek
    cv_thread.start()
    track_thread = threading.Thread(target=track_goal,args=(model,points)) # 追踪处理的函数
    track_thread.start()
    # cv_thread.join()
    track_thread.join()








  