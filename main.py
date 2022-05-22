import numpy as np
import cv2 as cv
import imutils

import beauty_face
import stickers


def main():
    src=cv.imread('img/05.jpg')
    cv.imshow('src',imutils.resize(src,800))
    big=beauty_face.big_eyes_auto(src)
    src=stickers.sticker(big)
    result=beauty_face.beauty_face(src)
    cv.imwrite('result_05.jpg',result)
    cv.waitKey(0)
    return 0


if __name__ == '__main__':
    main()