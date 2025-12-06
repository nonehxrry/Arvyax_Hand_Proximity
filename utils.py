import cv2
import numpy as np

LOWER_HSV = np.array([0, 40, 60], dtype="uint8")
UPPER_HSV = np.array([20, 150, 255], dtype="uint8")

LOWER_YCRCB = np.array([0, 133, 77], dtype="uint8")
UPPER_YCRCB = np.array([255, 173, 127], dtype="uint8")

MIN_CONTOUR_AREA = 1000

def get_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mask_hsv = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    mask_ycrcb = cv2.inRange(ycrcb, LOWER_YCRCB, UPPER_YCRCB)
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def get_hand_info(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > MIN_CONTOUR_AREA:
            M = cv2.moments(c)
            if M["m00"] == 0:
                return None, None, None
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
            ys = c[:, 0, 1]
            top_y = np.min(ys)
            top_x = c[np.argmin(ys)][0][0]
            fingertip = (top_x, top_y)
            return c, centroid, fingertip
    return None, None, None

def point_to_box_distance(point, box):
    x, y = point
    x_min, y_min, x_max, y_max = box
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    return np.sqrt(dx*dx + dy*dy)
