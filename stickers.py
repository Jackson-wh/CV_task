import dlib
import cv2
import numpy as np
import math
from PIL import Image
from scipy import ndimage
import imutils

predictor_path = 'src/shape_predictor_68_face_landmarks.dat'

# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector() #人脸检测器
predictor = dlib.shape_predictor(predictor_path) #68点检测器

def align_face(adding,landmarks):
    p1 = np.array([landmarks.part(36).x,landmarks.part(36).y ]) # 左眼坐标
    p2 = np.array([landmarks.part(45).x,landmarks.part(45).y ]) # 右眼坐标
    dp = p1 - p2
    angle = np.arctan(dp[0] / dp[1])
    print(angle)
    rot_img = adding.rotate(-angle)

    img = cv2.cvtColor(np.asarray(rot_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('rot',img)
    cv2.waitKey(0)
    return rot_img


#添加头饰特效
def sticker(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    im = Image.fromarray(src[:, :, ::-1])
    adding = Image.open('stickers/shine.png').convert("RGBA")
    faces = detector(gray,0)   #返回人脸四点坐标
    for face in faces:
        # 获取面部模式
        landmarks = predictor(gray, face)
        # 定位面部左上角点坐标
        x1, y1 = landmarks.part(0).x, landmarks.part(0).y
        # 定位面部右上角点坐标
        x2, y2 = landmarks.part(16).x, landmarks.part(16).y
        # 计算面部宽度
        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # 根据面部宽度计算尺寸
        size = int(d / 236 * 439)
        # 对头发图片缩放
        resized = adding.resize((size, size))
        # 在合适位置添加图片
        im.paste(resized, (int(x1 - d * 86 / 236), int(y1 - d * 394 / 236-(x2-x1)*2/5)), resized)

    img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return img

#添加眼镜特效
def glasses(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    im = Image.fromarray(src[:, :, ::-1])
    adding = Image.open('stickers/glasses.jpg').convert("RGBA")
    faces = detector(gray,0)   #返回人脸四点坐标
    for face in faces:
        landmarks = predictor(gray, face)
        x1, y1 = landmarks.part(37).x, landmarks.part(37).y  #眼上
        x2, y2 = landmarks.part(41).x, landmarks.part(41).y  #眼下
        x3, y3 = landmarks.part(36).x, landmarks.part(36).y  #左眼最左边
        x4, y4 = landmarks.part(45).x, landmarks.part(45).y  #右眼最右边
        length= math.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2)  #图片长度
        height= math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  #图片宽度
        adding=align_face(adding,landmarks)
        # 对头发图片缩放
        resized = adding.resize((int(7*length/4), int(20*height)))
        # 在合适位置添加图片
        im.paste(resized, (int(x3-length * 90 / 236), int(y1 - 5*height * 788 / 472)), resized)
       # im.paste(resized, (int(x3 ), int(y1 - 3*height/5)), resized)

    img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    return img

def main():
    src=cv2.imread('img/04.jpg')
    glasses(src)
    cv2.waitKey(0)
    return 0


if __name__ == '__main__':
    main()