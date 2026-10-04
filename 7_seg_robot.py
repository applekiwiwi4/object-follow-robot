# 6단계: 세그멘테이션 + 사람 선택 + 로봇 따라가기 + 다시 잡기 (전체 통합본)
#
# 조작법
#   클릭        : 그 사람 선택
#   드래그      : 드래그한 영역과 가장 많이 겹치는 사람 선택 (IoU)
#   SPACE       : 일시정지 / 재생
#   C           : 선택 해제
#   ESC         : 종료

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack

# ============================================================
# 설정값 (자주 바꾸는 숫자는 여기 모아 둠)
# ============================================================
VIDEO_PATH = "./video/walking3.mp4"   # 웹캠을 쓰려면 0 으로 바꾸기

CONF = 0.4          # YOLO 확신도 기준 (높이면 사람이 덜 잡힘)
MIN_H = 80          # 키가 이보다 작은(멀리 있는) 사람은 무시

# 로봇 설정값
FOLLOW_DIST = 10    # 로봇이 사람과 유지할 거리 (픽셀)
ROBOT_SPEED = 2     # 로봇이 한 화면마다 움직이는 최대 거리
CENTER_TOL = 60     # 사람이 화면 가운데서 이만큼 벗어나면 회전
NEAR_RATIO = 0.8    # 사람 키가 화면 높이의 80% 이상이면 너무 가까움
FAR_RATIO = 0.4     # 사람 키가 화면 높이의 40% 이하이면 너무 멂

# 다시 잡기 설정값
REACQ_FRAMES = 90   # 놓친 뒤 몇 화면 동안 다시 찾아볼지
REACQ_DIST = 1.5    # 마지막 위치에서 (사람 폭 x 이 값) 안에 나타나야 같은 사람으로 봄

# 세그멘테이션 색칠 진하기 (0 ~ 1)
FILL_ALPHA = 0.4

# ============================================================
# 준비
# ============================================================
model = YOLO("yolov8n-seg.pt")   # 세그멘테이션 모델
BaseTrack.reset_id()             # 추적 번호를 1번부터 시작

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("영상을 열 수 없습니다. 경로를 확인하세요:", VIDEO_PATH)
    exit()

boxes = []      # [(추적번호, (x1, y1, x2, y2)), ...]
polys = {}      # {추적번호: 몸 윤곽선 점들}
id_map = {}     # {새 번호: 원래 번호}  다시 잡은 사람을 원래 번호로 보여주기용

state = {"cur": None, "selected": None,
         "sdrawing": False, "sstart": None, "scores": {},
         "paused": False}


# ============================================================
# 선택 관련 함수
# ============================================================
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


def show_id(tid):
    """화면에 보여줄 번호 (다시 잡은 사람은 원래 번호)"""
    return id_map.get(tid, tid)


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
        print("선택: ID", show_id(state["selected"]))


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
        print("선택: ID", show_id(best_id), "점수:", round(best_score, 2))


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


# ============================================================
# 로봇 관련 함수
# ============================================================
def decide_command(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    offset = center_x - frame_w / 2      # 음수면 왼쪽, 양수면 오른쪽
    h_ratio = (y2 - y1) / frame_h        # 사람 키가 화면에서 차지하는 비율

    if offset < -CENTER_TOL:
        return "TURN LEFT"
    if offset > CENTER_TOL:
        return "TURN RIGHT"
    if h_ratio > NEAR_RATIO:
        return "BACKWARD"
    if h_ratio < FAR_RATIO:
        return "FORWARD"
    return "STOP (good distance)"


def move_robot(robot, target):
    dx = target[0] - robot[0]
    dy = target[1] - robot[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist > FOLLOW_DIST:
        step = min(ROBOT_SPEED, dist - FOLLOW_DIST)
        robot[0] += dx / dist * step
        robot[1] += dy / dist * step


def draw_robot(frame, robot, target):
    rx, ry = int(robot[0]), int(robot[1])
    if target is not None:
        cv2.line(frame, (rx, ry), (int(target[0]), int(target[1])),
                 (255, 200, 0), 1)
    cv2.circle(frame, (rx, ry), 15, (255, 0, 0), -1)
    cv2.putText(frame, "ROBOT", (rx - 25, ry + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def draw_command(frame, cmd):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, "CMD: " + cmd, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


# ============================================================
# 다시 잡기 함수
# ============================================================
def find_reacquire(boxes, last_box, ids_at_lost):
    lx1, ly1, lx2, ly2 = last_box
    lcx = (lx1 + lx2) / 2
    lcy = (ly1 + ly2) / 2
    lw = lx2 - lx1
    lh = ly2 - ly1

    best_id = None
    best_dist = lw * REACQ_DIST

    for tid, (x1, y1, x2, y2) in boxes:
        if tid in ids_at_lost:              # 놓칠 때 이미 있던 사람은 제외
            continue
        h = y2 - y1
        if h < lh * 0.7 or h > lh * 1.3:    # 키가 너무 다르면 제외
            continue
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_id = tid

    return best_id


# ============================================================
# 메인 반복
# ============================================================
cv2.namedWindow("follow")
cv2.setMouseCallback("follow", on_mouse)

raw = None
robot = None
last_selected = None
last_cmd = None
last_box = None
lost_count = 0
ids_at_lost = set()

while True:
    new_frame = False

    # ---------- 1. 새 화면 읽기 + YOLO 추적 (일시정지면 건너뜀) ----------
    if not state["paused"] or raw is None:
        ok, raw = cap.read()
        if not ok:
            print("영상이 끝났습니다.")
            break
        new_frame = True

        if raw.shape[1] > 960:
            scale = 960 / raw.shape[1]
            raw = cv2.resize(raw, None, fx=scale, fy=scale)

        results = model.track(raw, persist=True, classes=[0],
                              conf=CONF, iou=0.5, verbose=False)

        boxes = []
        polys = {}
        r = results[0]
        for i, b in enumerate(r.boxes):
            if b.id is None:
                continue
            tid = int(b.id[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if (y2 - y1) < MIN_H:
                continue
            boxes.append((tid, (x1, y1, x2, y2)))
            if r.masks is not None and len(r.masks.xy[i]) > 0:
                polys[tid] = r.masks.xy[i].astype(np.int32)

    frame = raw.copy()
    h, w = frame.shape[:2]

    # ---------- 2. 몸 모양대로 반투명 색칠 ----------
    overlay = frame.copy()
    for tid, box in boxes:
        if tid in polys:
            if tid == state["selected"]:
                fill = (0, 0, 255)
            else:
                fill = (0, 255, 0)
            cv2.fillPoly(overlay, [polys[tid]], fill)
    frame = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0)

    # ---------- 3. 윤곽선, 번호 그리기 + 선택한 사람 찾기 ----------
    target_box = None
    for tid, (x1, y1, x2, y2) in boxes:
        if tid == state["selected"]:
            color = (0, 0, 255)
            target_box = (x1, y1, x2, y2)
        else:
            color = (0, 255, 0)

        if tid in polys:
            cv2.polylines(frame, [polys[tid]], True, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"ID {show_id(tid)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tid in state["scores"]:
            cv2.putText(frame, f"{state['scores'][tid]:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- 4. 놓친 사람 다시 잡기 ----------
    if state["selected"] is None:
        last_box = None
        lost_count = 0
    elif target_box is not None:
        last_box = target_box
        lost_count = 0
    elif last_box is not None and new_frame:
        if lost_count == 0:
            ids_at_lost = set()
            for tid, box in boxes:
                ids_at_lost.add(tid)
        lost_count += 1

        if lost_count <= REACQ_FRAMES:
            new_id = find_reacquire(boxes, last_box, ids_at_lost)
            if new_id is not None:
                print("다시 잡음: ID", show_id(state["selected"]),
                      "(내부 번호", state["selected"], "->", new_id, ")")
                id_map[new_id] = show_id(state["selected"])
                state["selected"] = new_id
                last_selected = new_id      # 로봇이 처음 위치로 돌아가지 않게
                lost_count = 0

    # ---------- 5. 로봇 ----------
    if state["selected"] != last_selected:
        robot = None
        last_selected = state["selected"]

    target = None
    if state["selected"] is None:
        cmd = "WAITING (select a person)"
    elif target_box is None:
        if lost_count <= REACQ_FRAMES:
            cmd = f"STOP (ID {show_id(state['selected'])} lost, searching)"
        else:
            cmd = f"STOP (ID {show_id(state['selected'])} lost, gave up)"
    else:
        x1, y1, x2, y2 = target_box
        target = ((x1 + x2) / 2, y2)        # 발 위치를 따라감
        if robot is None:
            robot = [w / 2, h - 40.0]       # 화면 아래 가운데에서 출발
        if new_frame:
            move_robot(robot, target)
        cmd = decide_command(target_box, w, h)

    if robot is not None:
        draw_robot(frame, robot, target)
    draw_command(frame, cmd)

    if cmd != last_cmd:
        print("명령:", cmd)
        last_cmd = cmd

    # ---------- 6. 기타 표시 ----------
    if state["paused"]:
        cv2.putText(frame, "PAUSED", (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if state["sdrawing"]:
        cv2.rectangle(frame, state["sstart"], state["cur"], (255, 0, 255), 1)

    cv2.putText(frame, "Click/Drag: select  SPACE: pause  C: clear  ESC: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("follow", frame)

    # ---------- 7. 키 입력 ----------
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
