import cv2
from ultralytics import YOLO
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def predit_sum(Yolo_path,video_path):                      # 定义预测函数
    model = YOLO(Yolo_path)                                # 加载yolo预训练好的模型
    cap = cv2.VideoCapture(video_path)                     # 打开视频流

    while cap.isOpened():                                  # 视频是否打开
        success, frame = cap.read()                        # 读取每一帧  frame视频帧 
        if success :                                       # 成功
            results = model.track(frame,persist=True)      # 加载模型,ID固定
            num = results[0].boxes.shape[0]                # 取得第一维的数据，（检测框总数）
    
            annotated_frame = results[0].plot()            # 绘制预测结果
                #修改显示视频的尺寸
            h,w = annotated_frame.shape[:2]                # 获取图片的高和宽
            scale = 1                                    # 设置缩放比例
            new_w = int(w * scale)                         # 将缩放后的数值取整
            new_h = int(h * scale)
            show_img = cv2.resize(annotated_frame,(new_w,new_h))  # 用opencv的resize函数，重新设置图片。

            cv2.putText(show_img, str(num), (360, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  #图片上绘制总数
            cv2.imshow("YOLOv8 Inference", show_img)       # 显示图片
            
            if cv2.waitKey(1) & 0xFF == ord("q"):          # 等待1毫秒，并检测是否有键盘输入。如果有键盘输入，并且输入的键为字母 'q'，则会跳出循环
                break
    model.predict(verbose = False)
    cap.release()                                          # 释放cap对象的资源
    cv2.destroyAllWindows()                                # 销毁窗口

if __name__ == '__main__':
    Yolo_path="E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt"      # 预训练模型路径
    # video_path = "rtsp://admin:cd123456@172.18.110.25:554"                  # 视频流路径 
    video_path = "rtsp://admin:cd123456@172.18.110.51:554"
    predit_sum(Yolo_path,video_path)            