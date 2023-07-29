# 准备数据集 labelimg 进行框选 得出类别信息

# 对训练集 验证集 测试集 进行划分分别存储文件路径到txt文件中

# 修改yaml文件

# 训练

from ultralytics import YOLO
def main():
# Load a model
#model = YOLO('yolov8n.yaml')  # build a new model from YAML
#model = YOLO('E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt')  # load a pretrained model (recommended for training)
    model = YOLO('E:\\Test\\VsCode_Test\\Python\\yolo8\\train\\mydata\\Dataset\\yolov8.yaml').load('E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# data = "E:\\Test\\VsCode_Test\\Python\\yolo8\\train\\mydata\\Dataset\\mydata.yaml"
    model.train(data = 'E:\\Test\\VsCode_Test\\Python\\yolo8\\train\\mydata\\Dataset\\mydata.yaml', epochs=5, imgsz=640,batch=5,device='CPU')

if __name__ =='__main__':
    main()