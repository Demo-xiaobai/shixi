import cv2
import numpy as np

# 读取图像
image = cv2.imread('E:\\Test\\VsCode_Test\\Python\\yolo8\\data\\bus.jpg')

# 创建一个与原始图像大小相同的黑色遮罩
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# 定义ROI的多边形顶点坐标
points = np.array([[100, 50], [300, 50], [450, 200], [50, 200]])

# 在遮罩上绘制ROI区域
cv2.fillPoly(mask, [points], 255)

# 将遮罩应用于原始图像
roi = cv2.bitwise_and(image, image, mask=mask)

# 显示原始图像和提取的ROI区域
cv2.imshow('Original Image', image)
cv2.imshow('ROI', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
