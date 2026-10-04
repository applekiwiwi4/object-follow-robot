# -*- coding: utf-8 -*-
"""
객체 추적 + 로봇 따라가기 (웹캠 시뮬레이션 버전)

사용법
  S 키 : 따라갈 물체를 마우스로 드래그해서 선택 -> Enter(또는 Space)로 확정
  R 키 : 추적 초기화 (다시 선택하고 싶을 때)
  Q 키 : 프로그램 종료

화면 설명
  초록 네모  = 추적 중인 물체(목표)
  파란 원    = 가상 로봇 (목표를 일정 거리 뒤에서 따라감)
  맨 위 글씨 = 실제 로봇이라면 내렸을 명령 (FORWARD, TURN LEFT 등)
"""

import cv2
import math

# ------------------------------------------------------------
# 설정값 (숫자를 바꿔 보면서 동작 차이를 확인해 보세요)
# ------------------------------------------------------------
FOLLOW_DIST = 80    # 로봇이 목표와 유지할 거리 (픽셀 단위)
ROBOT_SPEED = 6     # 로봇이 한 프레임(화면 한 장)에 움직일 수 있는 최대 거리
CENTER_TOL = 60     # 목표가 화면 가운데에서 이만큼 벗어나면 "회전" 명령
NEAR_RATIO = 0.25   # 물체가 화면의 25% 이상을 차지하면 "너무 가까움"
FAR_RATIO = 0.05    # 물체가 화면의 5% 이하이면 "너무 멂"

WINDOW = "Object Follow Robot"


def create_tracker():
    """사용 가능한 추적기(Tracker)를 하나 만들어서 돌려준다.
    CSRT가 가장 정확하고, 없으면 KCF, 그것도 없으면 MIL을 쓴다."""
    for name in ["TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"]:
        if hasattr(cv2, name):
            return getattr(cv2, name)()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, name):
            return getattr(cv2.legacy, name)()
    raise RuntimeError("사용 가능한 추적기가 없습니다. opencv 설치를 확인하세요.")


def decide_command(x, bw, bh, frame_w, frame_h):
    """목표 위치와 크기를 보고 실제 로봇에게 내릴 명령을 정한다.
    - 좌우 위치  -> 회전 방향 결정
    - 박스 크기  -> 가까운지/먼지 판단 (가까울수록 크게 보임)"""
    center_x = x + bw / 2
    offset = center_x - frame_w / 2          # 음수면 왼쪽, 양수면 오른쪽
    area_ratio = (bw * bh) / (frame_w * frame_h)

    if offset < -CENTER_TOL:
        return "TURN LEFT"
    if offset > CENTER_TOL:
        return "TURN RIGHT"
    if area_ratio > NEAR_RATIO:
        return "BACKWARD"                    # 너무 가까우면 뒤로
    if area_ratio < FAR_RATIO:
        return "FORWARD"                     # 너무 멀면 앞으로
    return "STOP (good distance)"


def move_robot(robot, target):
    """가상 로봇을 목표 쪽으로 조금씩 이동시킨다.
    FOLLOW_DIST보다 가까워지면 멈춰서 '뒤에서 따라가는' 모양이 된다."""
    dx = target[0] - robot[0]
    dy = target[1] - robot[1]
    dist = math.hypot(dx, dy)                # 로봇과 목표 사이 거리
    if dist > FOLLOW_DIST:
        step = min(ROBOT_SPEED, dist - FOLLOW_DIST)
        robot[0] += dx / dist * step         # 방향(dx/dist)으로 step만큼 이동
        robot[1] += dy / dist * step


def draw_robot(frame, robot, target):
    """가상 로봇(파란 원)과 목표를 향한 선을 그린다."""
    rx, ry = int(robot[0]), int(robot[1])
    if target is not None:
        cv2.line(frame, (rx, ry), (int(target[0]), int(target[1])), (255, 200, 0), 1)
    cv2.circle(frame, (rx, ry), 15, (255, 0, 0), -1)
    cv2.putText(frame, "ROBOT", (rx - 25, ry + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def draw_command(frame, cmd):
    """화면 맨 위에 현재 명령을 표시한다."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, "CMD: " + cmd, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, "S:select  R:reset  Q:quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)                # 0번 카메라 = 노트북 웹캠
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 다른 프로그램이 카메라를 쓰고 있는지 확인하세요.")
        return

    tracker = None      # 추적기 (아직 물체를 선택하지 않았으면 None)
    robot = None        # 가상 로봇 위치 [x, y]
    last_cmd = None     # 직전 명령 (바뀔 때만 출력하려고 기억)

    while True:
        ok, frame = cap.read()               # 카메라에서 사진 한 장 읽기
        if not ok:
            print("카메라 화면을 읽지 못했습니다.")
            break

        clean = frame.copy()                 # 아무것도 안 그린 원본 (선택용)
        h, w = frame.shape[:2]
        cmd = "WAITING (press S)"
        target = None

        if tracker is not None:
            found, box = tracker.update(frame)   # 새 화면에서 물체 위치 찾기
            if found:
                x, y, bw, bh = [int(v) for v in box]
                target = (x + bw / 2, y + bh / 2)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (int(target[0]), int(target[1])), 5, (0, 255, 0), -1)

                if robot is None:                # 처음엔 화면 아래쪽에서 출발
                    robot = [target[0], h - 40.0]
                move_robot(robot, target)
                cmd = decide_command(x, bw, bh, w, h)
            else:
                cmd = "STOP (target lost)"       # 물체를 놓치면 멈춤

        if robot is not None:
            draw_robot(frame, robot, target)
        draw_command(frame, cmd)

        # 명령이 바뀔 때만 터미널에 출력
        # (나중에 실제 로봇을 연결하면 여기서 로봇에게 명령을 보내면 됨)
        if cmd != last_cmd:
            print("명령:", cmd)
            last_cmd = cmd

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            box = cv2.selectROI(WINDOW, clean, False, False)
            if box[2] > 0 and box[3] > 0:        # 제대로 드래그했을 때만
                tracker = create_tracker()
                tracker.init(clean, box)
                robot = None
        elif key == ord("r"):
            tracker = None
            robot = None
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
