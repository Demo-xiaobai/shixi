import threading

def increment_list(lst):
    for i in range(len(lst)):
        lst[i] += 1                #

# 创建一个包含数字的列表
my_list = [1, 2, 3, 4, 5]

# 创建线程
threads = []
for _ in my_list:
    t = threading.Thread(target=increment_list, args=(my_list,))
    threads.append(t)
    t.start()

# 等待所有线程执行完毕
for t in threads:
    t.join()

print(my_list)


def args(cur_position,cur_center_position):
    for pre_box in pre_boxs:    # 取出前一帧队列里中的一个检测框
                        pre_center_position = pre_box.xywh
                        pre_position = pre_box.xyxy
                        # if pre_car_id == cur_car_id:   # 同一ID的检测目标
                        dX = pre_center_position[0][0] - cur_center_position[0][0]
                        dY = pre_center_position[0][1] - cur_center_position[0][1]  #计算 前一帧和后一帧的 y轴坐标差
                    # 计算欧式距离
                        distance = torch.sqrt(dX**2+dY**2)
                    # 判断是否逆行
                        if np.sign(dY) == -1 and np.abs(dY)>=4:   #np.sign(dY) == -1 表示 < 0 逆行方向
                            # if (np.abs(dY)>=3 and np.abs(dY)<=10):
                            if (distance >= 3 and distance <=10):
                                cv2.putText(roi, 'WRONG DIRECTION', (int(cur_position[0][0]), int(cur_position[0][1])+100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3) 
                                warning(roi,int(cur_position[0][0]),int(cur_position[0][1])) #绘制警告图标
