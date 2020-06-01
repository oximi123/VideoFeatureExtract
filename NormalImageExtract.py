import cv2
import numpy as np


# SIFT 特征 产生的特征为 n * 128 维，n为关键点的个数，尺寸不确定
def sift(img, to_path):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img)
    kp, des = sift.compute(gray, kp)
    print(kp)
    des = np.array(des)
    print(des.shape)
    np.save(to_path, des)


# 相同大小图像特征维度相同
def hog(img, to_path):
    # todo add parameter to the descriptor
    hog = cv2.HOGDescriptor()
    feature = hog.compute(img)
    feature = feature.T
    np.save(to_path, feature)
    print(feature.shape)


if __name__ == '__main__':
    image = cv2.imread('video/1385.png')
    hog(image, 'hog_feature')
