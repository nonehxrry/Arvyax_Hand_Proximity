import cv2
import numpy as np

def skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0, 30, 60], np.uint8)
    upper_hsv = np.array([20, 150, 255], np.uint8)

    lower_ycrcb = np.array([0, 133, 77], np.uint8)
    upper_ycrcb = np.array([255, 173, 127], np.uint8)

    mask1 = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask2 = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    mask = cv2.bitwise_and(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    return mask
