import os
import random
import shutil


def split_data(save_img_path,dataset_dir):
    # 定义原始数据集所在目录
    # dataset_dir = 'path/to/dataset'  # 假设原始数据集目录为 path/to/dataset

    # 创建保存目录
    os.makedirs(save_img_path+'\\train\\images', exist_ok=True)
    os.makedirs(save_img_path+'\\train\\labels', exist_ok=True)
    os.makedirs(save_img_path+'\\val\\images', exist_ok=True)
    os.makedirs(save_img_path+'\\val\\labels', exist_ok=True)
    os.makedirs(save_img_path+'\\test\\images', exist_ok=True)
    os.makedirs(save_img_path+'\\test\\labels', exist_ok=True)

    # 拆分比例（可根据需求进行修改）
    train_ratio = 0.8       # 训练集比例
    val_ratio = 0.1         # 验证集比例
    test_ratio = 0.1        # 测试集比例

    random.seed(123)
    # 遍历原始数据集目录中的图片文件
    for filename in os.listdir(os.path.join(dataset_dir, 'images')):
        
        # 获取完整文件路径
        image_file_path = os.path.join(dataset_dir, 'images', filename)

        # 随机分配到训练集、验证集或测试集
        rand_num = random.random()
        if rand_num < train_ratio:
            target_dir = (save_img_path+'\\train')
        elif rand_num < train_ratio + val_ratio:
            target_dir = (save_img_path+'\\val')
        else:
            target_dir = (save_img_path+'\\test')

        # 目标文件路径
        target_image_path = os.path.join(target_dir, 'images', filename)
        
        # 移动图片文件和标签文件到目标文件夹
        shutil.move(image_file_path, target_image_path)
    random.seed(123)
    for filename in os.listdir(os.path.join(dataset_dir, 'labels')):
        # 获取完整文件路径
        # image_file_path = os.path.join(dataset_dir, 'images', filename)
        
        label_file_path = os.path.join(dataset_dir, 'labels', filename)  # 假设标签文件与图片文件名相同

        # 随机分配到训练集、验证集或测试集
        rand_num = random.random()
        if rand_num < train_ratio:
            target_dir = (save_img_path+'\\train')
        elif rand_num < train_ratio + val_ratio:
            target_dir = (save_img_path+'\\val')
        else:
            target_dir = (save_img_path+'\\test')

        # 目标文件路径
        # target_image_path = os.path.join(target_dir, 'images', filename)
        target_label_path = os.path.join(target_dir, 'labels', filename)

        # 移动标签文件到目标文件夹
        # shutil.move(image_file_path, target_image_path)
        shutil.move(label_file_path, target_label_path)



if __name__ == '__main__':
    save_img_path = 'E:\\Test\\VsCode_Test\\Python\\yolo8\\train\\mydata\\Dataset'

    save_lab_path = 'E:\\Test\\VsCode_Test\\Python\\yolo8\\train\\mydata\\Dataset'
    data_path = 'C:\\Users\\XLiang\\Desktop\\shixi\\data' # 数据集路径

    split_data(save_img_path,data_path)