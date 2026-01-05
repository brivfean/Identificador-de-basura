import cv2

def op_and(img1, img2):
    return cv2.bitwise_and(img1, img2)

def op_or(img1, img2):
    return cv2.bitwise_or(img1, img2)

def op_xor(img1, img2):
    return cv2.bitwise_xor(img1, img2)

def op_not(img):
    return cv2.bitwise_not(img)
