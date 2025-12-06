import cv2
import numpy as np
import time

CAMERA_INDEX = 0

LOWER_SKIN = np.array([0, 40, 60], dtype="uint8")
UPPER_SKIN = np.array([20, 150, 255], dtype="uint8")

VIRTUAL_BOX_NORM = (0.3, 0.3, 0.7, 0.7)

DISTANCE_THRESHOLD_WARNING = 150
DISTANCE_THRESHOLD_DANGER = 50

COLOR_SAFE = (0, 255, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_DANGER = (0, 0, 255)

MIN_CONTOUR_AREA = 1000

def preprocess_mask(hsv_frame):
    mask = cv2.inRange(hsv_frame, LOWER_SKIN, UPPER_SKIN)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

def get_hand_info(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > MIN_CONTOUR_AREA:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid = (cX, cY)
            else:
                return None, None, None
            ys = max_contour[:, 0, 1]
            top_y = np.min(ys)
            top_x = max_contour[np.argmin(ys)][0][0]
            fingertip_point = (top_x, top_y)
            return max_contour, centroid, fingertip_point
    return None, None, None

def point_to_box_distance(point, box):
    x, y = point
    x_min, y_min, x_max, y_max = box
    dx = max(x_min - x, 0, x - x_max)
    dy = max(y_min - y, 0, y - y_max)
    return np.sqrt(dx*dx + dy*dy)

def get_state(distance):
    if distance <= DISTANCE_THRESHOLD_DANGER:
        return "DANGER", COLOR_DANGER
    elif distance <= DISTANCE_THRESHOLD_WARNING:
        return "WARNING", COLOR_WARNING
    else:
        return "SAFE", COLOR_SAFE

def run_prototype():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x_min = int(VIRTUAL_BOX_NORM[0] * frame_width)
    y_min = int(VIRTUAL_BOX_NORM[1] * frame_height)
    x_max = int(VIRTUAL_BOX_NORM[2] * frame_width)
    y_max = int(VIRTUAL_BOX_NORM[3] * frame_height)
    VIRTUAL_BOX = (x_min, y_min, x_max, y_max)

    previous_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = preprocess_mask(hsv)
        
        contour, centroid, fingertip = get_hand_info(mask)
        
        current_state = "SAFE"
        state_color = COLOR_SAFE
        distance = float('inf')
        
        if fingertip:
            distance = point_to_box_distance(fingertip, VIRTUAL_BOX)
            current_state, state_color = get_state(distance)
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
            cv2.circle(frame, centroid, 5, (255, 255, 0), -1)
            cv2.circle(frame, fingertip, 8, (0, 255, 255), -1)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), state_color, 4)
        cv2.putText(frame, f"STATE: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 3)

        if fingertip:
            cv2.putText(frame, f"DISTANCE: {distance:.0f} px", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if current_state == "DANGER":
            cv2.putText(frame, "DANGER DANGER", (frame_width // 2 - 270, frame_height // 2),
                        cv2.FONT_HERSHEY_TRIPLEX, 2.0, COLOR_DANGER, 5)

        cv2.imshow("Arvyax Hand Proximity Detector", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_prototype()
