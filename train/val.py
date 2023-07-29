from ultralytics import YOLO


def vval():

    model = YOLO("C:\\Users\\XLiang\\runs\\detect\\train13\\weights\\best.pt")
    model.val(plots=True)

if __name__ == '__main__':
    vval()