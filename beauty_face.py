
#-*- coding:gb18030 -*-

#�ο� https://blog.csdn.net/u011941438/article/details/82416470
#�ο� https://zhuanlan.zhihu.com/p/409565920
import dlib
import cv2
import numpy as np
import math
import imutils

predictor_path = 'src/shape_predictor_68_face_landmarks.dat'

# ʹ��dlib�Դ���frontal_face_detector��Ϊ���ǵ�������ȡ��
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        # for idx,point in enumerate(land_marks_node):
        #     # 68������
        #     pos = (point[0,0],point[0,1])
        #     print(idx,pos)
        #     # ����cv2.circle��ÿ�������㻭һ��Ȧ����68��
        #     cv2.circle(img_src, pos, 5, color=(0, 255, 0))
        #     # ����cv2.putText���1-68
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(img_src, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        land_marks.append(land_marks_node)

    return land_marks


'''
������ Interactive Image Warping �ֲ�ƽ���㷨
'''

def localTranslationWarp_big(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    # ���㹫ʽ�е�|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # ����õ��Ƿ����α�Բ�ķ�Χ֮��
            # �Ż�����һ����ֱ���ж��ǻ��ڣ�startX,startY)�ľ������
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

            if (distance < ddradius):
                # �������i,j�������ԭ����
                # ���㹫ʽ���ұ�ƽ������Ĳ���
                rnorm = math.sqrt(distance) / radius
                ratio = 1 - (rnorm - 1) * (rnorm - 1) * 0.5

                # ӳ��ԭλ��
                UX = startX + ratio * (i - startX)
                UY = startY + ratio * (j - startY)

                # ����˫���Բ�ֵ���õ�UX��UY��ֵ
                value = BilinearInsert(srcImg, UX, UY)
                # �ı䵱ǰ i ��j��ֵ
                copyImg[j, i] = value

    return copyImg

def localTranslationWarp_thin(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    # ���㹫ʽ�е�|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # ����õ��Ƿ����α�Բ�ķ�Χ֮��
            # �Ż�����һ����ֱ���ж��ǻ��ڣ�startX,startY)�ľ������
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

            if (distance < ddradius):
                # �������i,j�������ԭ����
                # ���㹫ʽ���ұ�ƽ������Ĳ���
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio

                # ӳ��ԭλ��
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)

                # ����˫���Բ�ֵ���õ�UX��UY��ֵ
                value = BilinearInsert(srcImg, UX, UY)
                # �ı䵱ǰ i ��j��ֵ
                copyImg[j, i] = value

    return copyImg


# ˫���Բ�ֵ��
def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


def face_thin_auto(src):
    landmarks = landmark_dec_dlib_fun(src)

    # ���δ��⵽�����ؼ��㣬�Ͳ���������
    if len(landmarks) == 0:
        return

    for landmarks_node in landmarks:
        left_landmark = landmarks_node[3]
        left_landmark_down = landmarks_node[5]
        left_landmark_jaw=landmarks_node[5]
        left_landmark_jaw_down=landmarks_node[6]
        right_landmark = landmarks_node[13]
        right_landmark_down = landmarks_node[15]
        right_landmark_jaw=landmarks_node[11]
        right_landmark_jaw_down=landmarks_node[12]

        endPt = landmarks_node[30]

        # �����4���㵽��6����ľ�����Ϊ��������
        r_left = math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

        # �����6���㵽��8����ľ�����Ϊ���°;���
        r_left_jaw = math.sqrt(
            (left_landmark_jaw[0, 0] - left_landmark_jaw_down[0, 0]) * (left_landmark_jaw[0, 0] - left_landmark_jaw_down[0, 0]) +
            (left_landmark_jaw[0, 1] - left_landmark_jaw_down[0, 1]) * (left_landmark_jaw[0, 1] - left_landmark_jaw_down[0, 1]))

        # �����14���㵽��16����ľ�����Ϊ��������
        r_right = math.sqrt(
            (right_landmark[0, 0] - right_landmark_down[0, 0]) * (right_landmark[0, 0] - right_landmark_down[0, 0]) +
            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))

        # �����10���㵽��12����ľ�����Ϊ���°;���
        r_right_jaw = math.sqrt(
            (right_landmark_jaw[0, 0] - right_landmark_jaw_down[0, 0]) * (right_landmark_jaw[0, 0] - right_landmark_jaw_down[0, 0]) +
            (right_landmark_jaw[0, 1] - right_landmark_jaw_down[0, 1]) * (right_landmark_jaw[0, 1] - right_landmark_jaw_down[0, 1]))

        # �������
        thin_image = localTranslationWarp_thin(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
                                          r_left)
        # ���ұ���
        thin_image = localTranslationWarp_thin(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
                                          endPt[0, 1], r_right)
        # ������°�
        thin_image = localTranslationWarp_thin(thin_image, left_landmark_jaw[0, 0], left_landmark_jaw[0, 1], endPt[0, 0], endPt[0, 1],
                                          r_left_jaw)
        # ���ұ���
        thin_image = localTranslationWarp_thin(thin_image, right_landmark_jaw[0, 0], right_landmark_jaw[0, 1], endPt[0, 0],
                                          endPt[0, 1], r_right_jaw)

    return thin_image

def big_eyes_auto(src):
    landmarks = landmark_dec_dlib_fun(src)

    # ���δ��⵽�����ؼ��㣬�Ͳ���������
    if len(landmarks) == 0:
        return

    for landmarks_node in landmarks:
        left_landmark = landmarks_node[38]
        left_landmark_down = landmarks_node[27]
        right_landmark = landmarks_node[43]
        right_landmark_down = landmarks_node[27]

        endPt = landmarks_node[30]
        r_left = math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))
        r_right = math.sqrt(
            (right_landmark[0, 0] - right_landmark_down[0, 0]) * (right_landmark[0, 0] - right_landmark_down[0, 0]) +
            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))

        thin_image = localTranslationWarp_big(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
                                          r_left)
        thin_image = localTranslationWarp_big(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
                                          endPt[0, 1], r_right)
        return thin_image


def beauty_face(src):
    '''
    Dest =(Src * (100 - Opacity) + (Src + 2 * GuassBlur(EPFFilter(Src) - Src + 128) - 256) * Opacity) /100 ;
    '''
    dst = np.zeros_like(src)
    #int value1 = 3, value2 = 1; ĥƤ�̶���ϸ�ڳ̶ȵ�ȷ��
    v1 = 3
    v2 = 1
    dx = v1 * 5 # ˫���˲�����֮һ
    fc = v1 * 12.5 # ˫���˲�����֮һ
    p = 0.1
    temp4 = np.zeros_like(src)
    temp1 = cv2.bilateralFilter(src,dx,fc,fc)
    temp2 = cv2.subtract(temp1,src)
    temp2 = cv2.add(temp2, (10,10,10,128))
    temp3 = cv2.GaussianBlur(temp2,(2*v2 - 1,2*v2-1),0)
    temp4 = cv2.subtract(cv2.add(cv2.add(temp3, temp3), src), (10, 10, 10, 255))
    dst = cv2.addWeighted(src,p,temp4,1-p,0.0)
    dst = cv2.add(dst, (10, 10, 10,255))
    return dst
