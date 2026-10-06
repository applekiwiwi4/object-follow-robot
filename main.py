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
from config import *
from follow.detector import Detector
from follow.selector import Selector
from follow.reid import ReIdentifier
from follow.robot import Robot
# ============================================================
# 준비
# ============================================================
detector = Detector()            # 추적 번호를 1번부터 시작

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("영상을 열 수 없습니다. 경로를 확인하세요:", VIDEO_PATH)
    exit()

boxes = []      # [(추적번호, (x1, y1, x2, y2)), ...]
polys = {}      # {추적번호: 몸 윤곽선 점들}    
reid = ReIdentifier()
selector = Selector(reid.id_map)



# ============================================================
# 선택 관련 함수
# ============================================================
















# ============================================================
# 로봇 관련 함수
# ============================================================
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
# 메인 반복
# ============================================================
cv2.namedWindow("follow")
cv2.setMouseCallback("follow", selector.on_mouse)

raw = None
paused = False
last_cmd = None
robot = Robot()


while True:
    new_frame = False

    # ---------- 1. 새 화면 읽기 + YOLO 추적 (일시정지면 건너뜀) ----------
    if not paused or raw is None:
        ok, raw = cap.read()
        if not ok:
            print("영상이 끝났습니다.")
            break
        new_frame = True

        if raw.shape[1] > MAX_WIDTH:
            scale = MAX_WIDTH / raw.shape[1]
            raw = cv2.resize(raw, None, fx=scale, fy=scale)

        boxes, polys = detector.detect(raw)
        selector.boxes = boxes

    frame = raw.copy()
    h, w = frame.shape[:2]

    # ---------- 2. 몸 모양대로 반투명 색칠 ----------
    overlay = frame.copy()
    for tid, box in boxes:
        if tid in polys:
            if tid == selector.selected:
                fill = (0, 0, 255)
            else:
                fill = (0, 255, 0)
            cv2.fillPoly(overlay, [polys[tid]], fill)
    frame = cv2.addWeighted(overlay, FILL_ALPHA, frame, 1 - FILL_ALPHA, 0)

    # ---------- 3. 윤곽선, 번호 그리기 + 선택한 사람 찾기 ----------
    target_box = None
    for tid, (x1, y1, x2, y2) in boxes:
        if tid == selector.selected:
            color = (0, 0, 255)
            target_box = (x1, y1, x2, y2)
        else:
            color = (0, 255, 0)

        if tid in polys:
            cv2.polylines(frame, [polys[tid]], True, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, f"ID {reid.show_id(tid)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tid in selector.scores:
            cv2.putText(frame, f"{selector.scores[tid]:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    # ---------- 4. 놓친 사람 다시 잡기 ----------
    new_id = reid.update(selector.selected, target_box, boxes, polys,
                         raw, new_frame)
    if new_id is not None:
        selector.selected = new_id
        robot.keep_following(new_id)     # 로봇이 처음 위치로 돌아가지 않게
      
    # ---------- 5. 로봇 ----------
    cmd = robot.update(selector.selected,
                       reid.show_id(selector.selected),
                       target_box, w, h, new_frame,
                       reid.is_searching())

    if robot.pos is not None:
        draw_robot(frame, robot.pos, robot.target)
    draw_command(frame, cmd)
    if cmd != last_cmd:
        print("명령:", cmd)
        last_cmd = cmd

    # ---------- 6. 기타 표시 ----------
    if paused:
        cv2.putText(frame, "PAUSED", (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if selector.dragging:
        cv2.rectangle(frame, selector.drag_start, selector.drag_cur, (255, 0, 255), 1)

    cv2.putText(frame, "Click/Drag: select  SPACE: pause  C: clear  ESC: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("follow", frame)

    # ---------- 7. 키 입력 ----------
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == 32:
        paused = not paused
    if key == ord("c"):
       selector.clear()
    if cv2.getWindowProperty("follow", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
