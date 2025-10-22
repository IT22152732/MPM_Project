# main.py
import os, math, time
import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import trimesh
import numpy as np

# ==================== CONFIG ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
PUZZLE_MODELS = [
    ["pilar1.obj", "pilar2.obj", "pilar3.obj", "pilar4.obj", "pilar5.obj"],  # Puzzle 1
    ["puzzle2_1.obj", "puzzle2_2.obj", "puzzle2_3.obj"],                      # Puzzle 2
    ["puzzle3_1.obj", "puzzle3_2.obj", "puzzle3_3.obj", "puzzle3_4.obj"]      # Puzzle 3
]

# ==================== MEDIAPIPE HANDS ====================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.5, 
    max_num_hands=1
)
HL = mp_hands.HandLandmark

# ==================== GESTURE HELPERS ====================
def is_finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def is_open_palm(lm):
    return all(is_finger_up(lm,t,p) for t,p in [
        (HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value),
        (HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value),
        (HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value),
        (HL.PINKY_TIP.value, HL.PINKY_PIP.value)
    ])

def is_closed_fist(lm):
    return not any(is_finger_up(lm,t,p) for t,p in [
        (HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value),
        (HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value),
        (HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value),
        (HL.PINKY_TIP.value, HL.PINKY_PIP.value)
    ])

def is_peace_sign(lm):
    return (is_finger_up(lm, HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value)
            and is_finger_up(lm, HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value)
            and not is_finger_up(lm, HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value)
            and not is_finger_up(lm, HL.PINKY_TIP.value, HL.PINKY_PIP.value))

def is_pointing(lm):
    return (is_finger_up(lm, HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value)
            and not is_finger_up(lm, HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value)
            and not is_finger_up(lm, HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value)
            and not is_finger_up(lm, HL.PINKY_TIP.value, HL.PINKY_PIP.value))

def is_thumb_up(lm):
    return lm[HL.THUMB_TIP.value].y < lm[HL.THUMB_IP.value].y and all(
        not is_finger_up(lm,t,p) for t,p in [
            (HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value),
            (HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value),
            (HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value),
            (HL.PINKY_TIP.value, HL.PINKY_PIP.value)
        ]
    )

def is_thumb_down(lm):
    return lm[HL.THUMB_TIP.value].y > lm[HL.THUMB_IP.value].y and all(
        not is_finger_up(lm,t,p) for t,p in [
            (HL.INDEX_FINGER_TIP.value, HL.INDEX_FINGER_PIP.value),
            (HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value),
            (HL.RING_FINGER_TIP.value, HL.RING_FINGER_PIP.value),
            (HL.PINKY_TIP.value, HL.PINKY_PIP.value)
        ]
    )

def is_ok_sign(lm):
    return math.hypot(
        lm[HL.INDEX_FINGER_TIP.value].x - lm[HL.THUMB_TIP.value].x,
        lm[HL.INDEX_FINGER_TIP.value].y - lm[HL.THUMB_TIP.value].y
    ) < 0.05 and is_finger_up(lm, HL.MIDDLE_FINGER_TIP.value, HL.MIDDLE_FINGER_PIP.value)

def is_pinch(lm):
    return math.hypot(
        lm[HL.INDEX_FINGER_TIP.value].x - lm[HL.THUMB_TIP.value].x,
        lm[HL.INDEX_FINGER_TIP.value].y - lm[HL.THUMB_TIP.value].y
    ) < 0.03

# ==================== UTILS ====================
def lerp(a,b,t): return a+(b-a)*t
def map_2d_to_3d(x2d,y2d,w,h,depth=0): return [(x2d/w-0.5)*6.0, -(y2d/h-0.5)*4.0, depth]
def map_3d_to_2d(x3d,y3d,w,h): return int((x3d/6.0+0.5)*w), int((-y3d/4.0+0.5)*h)
def distance_3d(a,b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

# ==================== PUZZLE SELECTION ====================
def select_puzzle():
    puzzles = ["Puzzle 1", "Puzzle 2", "Puzzle 3"]
    selection = 0
    cap_sel = cv2.VideoCapture(0)
    if not cap_sel.isOpened(): raise RuntimeError("Could not open webcam")
    
    while True:
        ret, frame = cap_sel.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        for i, p in enumerate(puzzles):
            color = (0,255,0) if i==selection else (0,0,255)
            cv2.rectangle(frame, (50,50+i*100), (400,120+i*100), color, -1)
            cv2.putText(frame, p, (70,100+i*100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.putText(frame,"Use W/S keys, ENTER to select, Q=quit",(50,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
        cv2.imshow("Select Puzzle", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'): selection = (selection - 1) % len(puzzles)
        elif key == ord('s'): selection = (selection + 1) % len(puzzles)
        elif key == 13: break
        elif key == ord('q'): selection = None; break
    cap_sel.release()
    cv2.destroyAllWindows()
    return selection

# ==================== LOAD PIECES ====================
def load_pieces(chosen_puzzle):
    pieces_data = []

    # Default start/correct positions per puzzle
    if chosen_puzzle == 0:  # Puzzle 1
        starts = [[-3,-1.4,0],[-1.5,-1.4,0],[0,-1.4,0],[1.5,-1.4,0],[3,-1.4,0]]
        corrects = [[-1,0,0],[-0.5,0,0],[0,0,0],[0.5,0,0],[1,0,0]]
    elif chosen_puzzle == 1:  # Puzzle 2
        starts = [[-2,-1.2,0],[0,-1.2,0],[2,-1.2,0]]
        corrects = [[-0.5,0,0],[0,0,0],[0.5,0,0]]
    else:  # Puzzle 3
        starts = [[-3,-1.2,0],[-1,-1.2,0],[1,-1.2,0],[3,-1.2,0]]
        corrects = [[-1,0,0],[-0.33,0,0],[0.33,0,0],[1,0,0]]

    MODEL_FILES = PUZZLE_MODELS[chosen_puzzle]
    for i,fname in enumerate(MODEL_FILES):
        pieces_data.append({
            "id": i,
            "file": os.path.join(MODEL_FOLDER,fname),
            "start": starts[i] if i<len(starts) else [-2+i,-1,0],
            "correct": corrects[i] if i<len(corrects) else [0,0,0]
        })

    # Load meshes
    missing = [p["file"] for p in pieces_data if not os.path.isfile(p["file"])]
    if missing: raise SystemExit(f"Missing model files:\n{missing}")

    for p in pieces_data:
        mesh = trimesh.load_mesh(p["file"], force='mesh')
        if isinstance(mesh, trimesh.Scene): mesh = trimesh.util.concatenate(mesh.dump())
        bbox = mesh.bounding_box.extents
        if max(bbox) > 2.5: mesh.apply_scale(2.5/max(bbox))
        p["vertices"] = np.array(mesh.vertices, dtype=np.float32)
        p["faces"] = np.array(mesh.faces, dtype=np.int32)
        p["pos"] = p["start"].copy()
        p["grabbed"] = False
        p["snapped"] = False

    return pieces_data

# ==================== OPENGL DRAW ====================
def draw_piece(piece):
    glPushMatrix()
    glTranslatef(*piece["pos"])
    glEnable(GL_LIGHTING); glEnable(GL_LIGHT0); glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE)
    verts,faces = piece["vertices"], piece["faces"]
    glBegin(GL_TRIANGLES)
    for face in faces:
        for idx in face:
            v = verts[idx]
            glVertex3f(*v)
    glEnd()
    glDisable(GL_COLOR_MATERIAL); glDisable(GL_LIGHT0); glDisable(GL_LIGHTING)
    glPopMatrix()

def select_nearest_piece(hand_world_pos, pieces_data, PICK_THRESHOLD):
    best,best_d = None,float("inf")
    for p in pieces_data:
        if p["snapped"]: continue
        d = distance_3d(hand_world_pos, p["pos"])
        if d < best_d: best_d, best = d, p
    if best is not None and best_d <= PICK_THRESHOLD: return best["id"]
    return None

# ==================== MAIN ====================
def main():
    chosen_puzzle = select_puzzle()
    if chosen_puzzle is None: exit("No puzzle selected.")

    pieces_data = load_pieces(chosen_puzzle)
    total_pieces = len(pieces_data)
    move_smooth = 0.25
    grabbed_obj_id = None
    PICK_THRESHOLD = 1.5
    score = 0
    last_reset = 0
    reset_cooldown = 1

    # OpenGL init
    pygame.init()
    display=(1100,700)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glViewport(0,0,display[0],display[1])
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, display[0]/display[1], 0.1, 100)
    glMatrixMode(GL_MODELVIEW); glEnable(GL_DEPTH_TEST)
    glClearColor(0.08,0.08,0.08,1)
    CAMERA_Z = -10

    # Webcam init
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): raise RuntimeError("Could not open webcam")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,540)

    print("Use gestures: open palm=grab, fist=drop, peace=reset, point/select, q=quit")

    running = True
    while running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame,1); h,w=frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_center_px = None
        open_palm=closed_fist=peace=point=thumb_up=thumb_down=ok_sign=pinch=False

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark
            cx = int(sum(lm[i].x*w for i in range(len(lm)))/len(lm))
            cy = int(sum(lm[i].y*h for i in range(len(lm)))/len(lm))
            hand_center_px = (cx,cy)

            # Gesture detection
            open_palm=is_open_palm(lm); closed_fist=is_closed_fist(lm)
            peace=is_peace_sign(lm); point=is_pointing(lm)
            thumb_up=is_thumb_up(lm); thumb_down=is_thumb_down(lm)
            ok_sign=is_ok_sign(lm); pinch=is_pinch(lm)

            hand_world = map_2d_to_3d(cx,cy,w,h,0)

            # Grab/Drop
            if open_palm and grabbed_obj_id is None:
                pick_id = select_nearest_piece(hand_world, pieces_data, PICK_THRESHOLD)
                if pick_id is not None:
                    grabbed_obj_id = pick_id
                    for p in pieces_data: p["grabbed"] = (p["id"]==grabbed_obj_id)
            if closed_fist and grabbed_obj_id is not None:
                for p in pieces_data:
                    if p["id"] == grabbed_obj_id: p["grabbed"]=False
                grabbed_obj_id = None

            # Reset
            if peace and time.time()-last_reset>reset_cooldown:
                for p in pieces_data: p["pos"]=p["start"].copy(); p["grabbed"]=False; p["snapped"]=False
                grabbed_obj_id=None; score=0; last_reset=time.time()

            # Feedback overlay
            feedbacks = [(point,"👉 Pointing detected"),(ok_sign,"👌 OK detected"),
                         (thumb_up,"👍 Thumbs Up"),(thumb_down,"👎 Thumbs Down"),(pinch,"🤏 Pinch detected")]
            for i,(cond,text) in enumerate(feedbacks):
                if cond: cv2.putText(frame,text,(10,90+30*i),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

            # Move grabbed piece
            if grabbed_obj_id is not None:
                piece = next((x for x in pieces_data if x["id"]==grabbed_obj_id),None)
                if piece and not piece["snapped"]:
                    target = map_2d_to_3d(cx,cy,w,h,piece["pos"][2])
                    piece["pos"][0] = lerp(piece["pos"][0], target[0], move_smooth)
                    piece["pos"][1] = lerp(piece["pos"][1], target[1], move_smooth)

        # Snap pieces
        score = sum(1 for p in pieces_data if p["snapped"])
        for p in pieces_data:
            if p["snapped"]: continue
            dx,dy,dz = [p["pos"][i]-p["correct"][i] for i in range(3)]
            if math.sqrt(dx*dx + dy*dy + dz*dz) < 0.35:
                p["pos"] = p["correct"].copy()
                p["grabbed"] = False
                p["snapped"] = True

        # Webcam overlay
        cv2.putText(frame,"Open palm=grab | Fist=drop | Peace=reset | q=quit",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(200,200,200),1)
        cv2.putText(frame,f"Score: {score}/{total_pieces}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)
        if score == total_pieces:
            cv2.putText(frame,"🎉 You Win! Puzzle Completed! 🎉",(w//4,h//2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

        for p in pieces_data:
            px,py = map_3d_to_2d(p["pos"][0], p["pos"][1], w, h)
            color = (0,255,0) if p["snapped"] else (0,0,255)
            cv2.circle(frame,(px,py),15,color,2)
            cv2.putText(frame,f"{p['id']}",(px-10,py-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)

        if grabbed_obj_id is not None and hand_center_px:
            hx,hy = hand_center_px
            cv2.circle(frame,(hx,hy),12,(0,255,255),-1)
            cv2.putText(frame,f"Object {grabbed_obj_id}",(hx+15,hy),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        cv2.imshow("Webcam (q=quit)",frame)

        # OpenGL rendering
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glLoadIdentity(); glTranslatef(0,0,CAMERA_Z)
        for p in pieces_data: draw_piece(p)
        pygame.display.flip(); pygame.time.wait(10)

        # Event handling
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: running=False
        if cv2.waitKey(1)&0xFF==ord('q'): running=False

    # Cleanup
    cap.release(); cv2.destroyAllWindows(); hands.close(); pygame.quit()

if __name__=="__main__":
    main()
