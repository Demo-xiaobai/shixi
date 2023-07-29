import os
import random
import shutil



# 定义原始数据集所在目录

def split(save_path,dataset_dir):

    # 创建保存目录  path
    os.makedirs(save_path+'\\train',exist_ok=True)
    os.makedirs(save_path+'\\val',exist_ok=True)
    os.makedirs(save_path+'\\test',exist_ok=True)

    # 拆分比例
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    random.seed(123)
    # 遍历原始目录
    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir,filename)

        rand_num = random.random()
        if rand_num < train_ratio:
            target_dir = (save_path+'\\train')
        elif rand_num < train_ratio + val_ratio:
            target_dir = (save_path+'\\val')
        else:
            target_dir = (save_path+'\\test')
        
        # 目标文件夹路径
        target_path = os.path.join(target_dir, filename)
        
        print(target_path)
        # 移动文件到目标文件夹ex
        shutil.move(file_path, target_path)


if __name__ == '__main__':

    '''
        train
            -- images
            
        val
            -- images
            
        test
            -- images
            
        一个数据集按照训练集、验证集、测试集，即以上的文件夹组织形式，按比例划分进这些文件夹
    
    '''

    # 保存路径
    save_img_path = 'E:\\Test\\VsCode_Test\\Python\\yolo8\\train\\mydata\\Dataset\\train'

    save_lab_path = 'E:\\Test\\VsCode_Test\\Python\\yolo8\\train\\mydata\\Dataset\\labels'

    data_img_path = 'C:\\Users\\XLiang\\Desktop\\shixi\\images' # 数据集路径
    data_lab_path = 'C:\\Users\\XLiang\\Desktop\\shixi\\labels'

    '''
        images
            -- test
    
    '''


    split(save_img_path,data_img_path)
    split(save_lab_path,data_lab_path)
