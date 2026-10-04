#동영상으로 실험해보기

import cv2
from ultralytics import YOLO


VIDEO_PAHT = "./video/walking1.mp4"
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(VIDEO_PAHT)

if not cap.isOpened():
    print("영상을 열 수 없습니다.", VIDEO_PAHT)
    exit()

boxes = []
state = {"cur": None, "selected":None,
         "sdrawing":False, "sstart":None, "scores":{}, "paused": False}



def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union == 0:
        return 0
    return inter / union

def select_by_point(x, y):
    state["scores"] = {}
    state["selected"] = None
    for tid, box in boxes:
        if point_in_box(x, y, box):
            state["selected"] = tid
            break
    if state["selected"] is None:
        print("선택 해제")
    else:
        print("선택: ID", state["selected"])

def select_by_drag(drag):
    state["scores"] = {}
    best_id = None
    best_score = 0
    for tid, box in boxes:
        score = iou(drag, box)
        state["scores"][tid] = score
        if score > best_score:
            best_score = score
            best_id = tid
    if best_score < 0.1:
        state["selected"] = None
        print("선택 해제 (많이 겹치는 사람 없음)")
    else:
        state["selected"] = best_id
        print("선택: ID", best_id, "점수:", round(best_score, 2))
# ----- 여기까지 7단계와 같음 -----

# 새로 바뀜: 모드 구분 없이 항상 선택만 함 (7단계의 선택 모드 부분과 같음)
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["sdrawing"] = True
        state["sstart"] = (x, y)
        state["cur"] = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["sdrawing"]:
        state["cur"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["sdrawing"]:
        state["sdrawing"] = False
        x0, y0 = state["sstart"]
        drag = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        if drag[2] - drag[0] > 3 and drag[3] - drag[1] > 3:
            select_by_drag(drag)
        else:
            select_by_point(x, y)

cv2.namedWindow("follow")
cv2.setMouseCallback("follow", on_mouse)

raw = None

while True:
    if not state["paused"] or raw is None:
        ok, raw = cap.read()
        if not ok:
            print("영상이 끝났습니다.")
            break

        if raw.shape[1] > 960:
            scale = 960 / raw.shape[1]
            raw = cv2.resize(raw, None, fx=scale, fy=scale)
        results = model.track(raw, persist= True, classes=[0],
                          conf=0.4, iou=0.5, verbose=False)


        boxes = []
        for b in results[0].boxes:
            if b.id is None:
                continue
            tid = int(b.id[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            boxes.append((tid, (x1, y1, x2, y2)))

    frame = raw.copy()

    found = False
    for tid, (x1, y1, x2, y2) in boxes:
        if tid == state["selected"]:
            color = (0, 0, 255)
            found = True
        else:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID{tid}", (x1, y1 -5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tid in state["scores"]:
            cv2.putText(frame, f"{state['scores'][tid]:.2f}", (x1, y2 +18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if state["selected"] is not None and not found:
        cv2.putText(frame, f"ID {state['selected']} LOST", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    if state["paused"]:
        w = frame.shape[1]
        cv2.putText(frame, "PAUSED", (w -130, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    if state["sdrawing"]:
        cv2.rectangle(frame, state["sstart"], state["cur"], (255,0,255),1)

    h = frame.shape[0]
    cv2.putText(frame, "Click/Drag: select C: clear ESC: quit",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

    cv2.imshow("follow", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == 32:
        state["paused"] = not state["paused"]
    if key == ord("c"):
        state["selected"] = None
        state["scores"] = {}
    if cv2.getWindowProperty("follow", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

