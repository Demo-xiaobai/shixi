import threading

lock = threading.Lock()
condition = threading.Condition(lock)
should_run_a = True
  
a = 10
def thread_a():
    global should_run_a
    global a
    while True:
        with lock:
            # 等待B线程完成
            while not should_run_a:
                condition.wait()
            
            # 执行A线程工作
            a =100
            
            # 设置should_run_a为False，让B线程执行
            should_run_a = False
            # 通知B线程可以执行
            condition.notify()

def thread_b():
    global should_run_a
    global a
    while True:
        with lock:
            # 等待A线程完成
            while should_run_a:
                condition.wait()
            
            # 执行B线程工作
            print(a)
            
            # 设置should_run_a为True，让A线程执行
            should_run_a = True
            # 通知A线程可以执行
            condition.notify()

# 创建并启动A、B两个线程
thread_c = threading.Thread(target=thread_a)
thread_c.start()

thread_d = threading.Thread(target=thread_b)
thread_d.start()
thread_c.join()

