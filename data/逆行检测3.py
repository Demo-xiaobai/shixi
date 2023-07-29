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
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=3)
            roi = cv2.bitwise_and(frame, frame, mask=mask)         # 将遮罩应用于原始图像  1要检测的区域

            results = model.track(roi,persist=True,classes=[2,5,7])   #跟踪
            boxes = results[0].boxes
            

            #从队列中取出数据
            if queue.empty():
                print("这是第一帧！！！")
            else:
                # 取出前一帧的数据
                midmid_position_list = queue.get()
                position = queue.get()
                car_id = queue.get()

                # 得出当前帧的数据
                if(len(boxes) > 0): 
                    cur_position_list = []
                    cur_mid_position_list = []
                    cur_cars_id = []            # 存储每一辆车的ID
                    for box in boxes:
                        cur_position_list = box.xyxy
                        cur_mid_position_list = box.xywh
                        cur_cars_id = box.id
                # 计算x坐标差
                    # 先确定是同一辆车，我们有2个id列表 cur_cars_id 和  car_id 当前帧和上一帧  找出其中ID值相同的在进行坐标计算
                    # 遍历列表？？？?
                    if len(cur_cars_id)>=1 and len(car_id)>=1:
                        for cur_car_id in cur_cars_id:
                            for per_car_id in car_id:
                                if(cur_car_id.item() == per_car_id.item()):          #判断ID是否相等
                        # if(cur_cars_id[0] == car_id[0]):
                                    dX = cur_mid_position_list[0][1]-midmid_position_list[0][1]  #后一帧减去前一帧的数据
                                    dY = cur_mid_position_list[0][2]-midmid_position_list[0][2]
                                    if np.abs(dX)>=12:
                                        dirX = "North" if np.sign(dX) == 1 else "South"
                                        cv2.putText(roi, 'WRONG DIRECTION', (int(cur_mid_position_list[0][1])-25, int(cur_mid_position_list[0][2])-25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1)

            if(len(boxes) > 0):         #第一帧    
                position_lists = []      #记录所有检测框的 左上角 和 右下角的坐标
                mid_position_lists = []   #记录所有检测框的 中心点坐标和 宽高
                cars_id = []            #记录所有检测框的ID
                # package = torch.tensor([])
                for box in boxes:
                    # 用tensor张量                              
                    queue.put(position_lists.append(box.xyxy))         
                    queue.put(mid_position_lists.append(box.xywh))
                    queue.put(cars_id.append(box.id))



# 检测框入队
# 在计算距离



            # 画出结果
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, 'WRONG DIRECTION', (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 255), 1)

            new_image = cv2.bitwise_or(annotated_frame,frame) #将剪切的ROI区域和原始图像做并集
            # 缩小显示的图像
            w,h = new_image.shape[:2]
            scale = 0.5
            new_w = int(w * scale) #960
            new_h = int(h * scale) #540
            show_img = cv2.resize(new_image,(new_h,new_w))  # 用opencv的resize函数，重新设置图片。
            cv2.imshow("YOLOv8 Inference",show_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Yolo_path="E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt"      # 预训练模型路径
    video_path = "rtsp://admin:cd123456@172.18.110.51:554"                  # 视频流路径 
    reverse_Track(Yolo_path,video_path)      