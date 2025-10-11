import math
import cv2
import mediapipe as mp
import time

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    max_num_hands=1
)
mp_drawing = mp.solutions.drawing_utils

# ---------- Utility helpers ----------
def to_pixel_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_point(p1, p2, t):
    return (int(lerp(p1[0], p2[0], t)), int(lerp(p1[1], p2[1], t)))

# landmark index helpers for readability
HL = mp_hands.HandLandmark

def is_finger_up(landmarks, tip_idx, pip_idx):
    # using normalized y (smaller y = finger is up in image coords)
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def is_open_palm(landmarks):
    # Check index, middle, ring, pinky are up
    return (
        is_finger_up(landmarks, HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value)
        and is_finger_up(landmarks, HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value)
        and is_finger_up(landmarks, HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value)
        and is_finger_up(landmarks, HL.PINKY_TIP.value, HL.PINKY_PIP.value)
    )

def is_closed_fist(landmarks):
    return (
        not is_finger_up(landmarks, HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.PINKY_TIP.value, HL.PINKY_PIP.value)
    )

def is_pointing(landmarks):
    # index up, other fingers down
    return (
        is_finger_up(landmarks, HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.PINKY_TIP.value, HL.PINKY_PIP.value)
    )

def is_peace_sign(landmarks):
    # index and middle up, ring and pinky down
    return (
        is_finger_up(landmarks, HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value)
        and is_finger_up(landmarks, HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value)
        and not is_finger_up(landmarks, HL.PINKY_TIP.value, HL.PINKY_PIP.value)
    )

# ---------- Simulated interactive objects ----------
# Simple placeholder objects (circle) to simulate 3D artifacts.
objects = [
    {"id": 0, "pos": (150, 200), "r": 36, "color": (10, 120, 200), "orig": (150, 200)},
    {"id": 1, "pos": (350, 180), "r": 44, "color": (20, 200, 100), "orig": (350, 180)},
    {"id": 2, "pos": (500, 300), "r": 40, "color": (200, 100, 40), "orig": (500, 300)},
]

grabbed = False
grabbed_obj_id = None
grab_offset = (0, 0)
move_smooth = 0.35         # smoothing factor for object follow (0-1)

# ---------- Webcam init ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

cv2.namedWindow("Gesture -> Object Interaction", cv2.WINDOW_NORMAL)

# ---------- Main loop ----------
last_reset_time = 0
reset_cooldown = 1.0  # seconds to avoid accidental repeated resets

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_text = "No gesture"
    hand_center_px = None
    index_tip_px = None

    if results.multi_hand_landmarks:
        # use the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = hand_landmarks.landmark

        # draw landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # pixel coords dictionary
        pts = {i: to_pixel_coords(landmarks[i], w, h) for i in range(len(landmarks))}

        # hand center as average of landmarks
        cx = int(sum([p[0] for p in pts.values()]) / len(pts))
        cy = int(sum([p[1] for p in pts.values()]) / len(pts))
        hand_center_px = (cx, cy)
        index_tip_px = pts[HL.INDEX_FINGER_TIP.value]

        # detect gestures
        if is_open_palm(landmarks):
            gesture_text = "Open Palm (Grab)"
            # if not holding something, try to pick nearest object
            if not grabbed:
                # find nearest object
                nearest = None
                min_d = 9999
                for obj in objects:
                    d = distance(hand_center_px, obj["pos"])
                    if d < min_d:
                        min_d = d
                        nearest = obj
                # threshold to pick (tune as needed)
                if nearest and min_d < 100:
                    grabbed = True
                    grabbed_obj_id = nearest["id"]
                    # store offset between hand center and object pos to maintain relative grab point
                    grab_offset = (nearest["pos"][0] - hand_center_px[0], nearest["pos"][1] - hand_center_px[1])
        elif is_closed_fist(landmarks):
            gesture_text = "Closed Fist (Drop)"
            # if holding, drop
            if grabbed:
                grabbed = False
                grabbed_obj_id = None
        elif is_pointing(landmarks):
            gesture_text = "Pointing"
            # highlight object under index finger if close
            for obj in objects:
                if distance(index_tip_px, obj["pos"]) < obj["r"] + 30:
                    cv2.putText(frame, f"Pointing -> object {obj['id']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, obj["color"], 2)
        elif is_peace_sign(landmarks):
            gesture_text = "Peace (Reset positions)"
            # small cooldown to avoid repeating resets every frame
            now = time.time()
            if now - last_reset_time > reset_cooldown:
                for obj in objects:
                    obj["pos"] = obj["orig"]
                last_reset_time = now
        else:
            gesture_text = "No gesture"

        # If an object is grabbed, move it toward hand center + offset with smoothing
        if grabbed and grabbed_obj_id is not None and hand_center_px is not None:
            # compute target pos
            target = (hand_center_px[0] + grab_offset[0], hand_center_px[1] + grab_offset[1])
            # apply smoothing
            obj = next((o for o in objects if o["id"] == grabbed_obj_id), None)
            if obj:
                newpos = lerp_point(obj["pos"], target, move_smooth)
                obj["pos"] = newpos

    # ---------- Draw simulated objects ----------
    for obj in objects:
        pos = obj["pos"]
        r = obj["r"]
        color = obj["color"]
        # if grabbed highlight differently
        if grabbed and grabbed_obj_id == obj["id"]:
            # draw a filled circle with ring
            cv2.circle(frame, pos, r + 6, (0, 255, 255), -1)
            cv2.circle(frame, pos, r, color, -1)
            cv2.circle(frame, pos, r, (255, 255, 255), 2)
            cv2.putText(frame, f"Grabbed {obj['id']}", (pos[0] - r, pos[1] - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            cv2.circle(frame, pos, r, color, -1)
            cv2.circle(frame, pos, r, (255, 255, 255), 2)
            cv2.putText(frame, f"Obj {obj['id']}", (pos[0] - r, pos[1] - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # ---------- HUD and debug ----------
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,200), 2)
    cv2.putText(frame, "Open palm: grab | Closed fist: drop | Point: highlight | Peace: reset", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # show frame
    cv2.imshow("Gesture -> Object Interaction", frame)

    # exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
