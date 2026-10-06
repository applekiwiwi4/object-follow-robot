import cv2

import config

SELECTED_COLOR = (0, 0, 255)   # 선택한 사람: 빨강
NORMAL_COLOR = (0, 255, 0)     # 나머지: 초록


def draw_people(frame, boxes, polys, selected, label_fn, scores):
    """사람들을 그린다 (색칠, 윤곽선, 번호, 드래그 점수). 그린 화면을 돌려준다."""
    # 1. 몸 모양대로 반투명 색칠 (FILL_ALPHA가 0이면 건너뜀)
    if config.FILL_ALPHA > 0:
        overlay = frame.copy()
        for tid, box in boxes:
            if tid in polys:
                color = SELECTED_COLOR if tid == selected else NORMAL_COLOR
                cv2.fillPoly(overlay, [polys[tid]], color)
        frame = cv2.addWeighted(overlay, config.FILL_ALPHA,
                                frame, 1 - config.FILL_ALPHA, 0)

    # 2. 윤곽선(없으면 네모), 번호, 점수
    for tid, (x1, y1, x2, y2) in boxes:
        color = SELECTED_COLOR if tid == selected else NORMAL_COLOR
        if tid in polys:
            cv2.polylines(frame, [polys[tid]], True, color, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {label_fn(tid)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if tid in scores:
            cv2.putText(frame, f"{scores[tid]:.2f}", (x1, y2 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_robot(frame, pos, target):
    rx, ry = int(pos[0]), int(pos[1])
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


def draw_status(frame, paused, dragging, drag_start, drag_cur):
    """일시정지 표시, 드래그 네모, 아래쪽 안내 글자"""
    h, w = frame.shape[:2]
    if paused:
        cv2.putText(frame, "PAUSED", (w - 130, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if dragging:
        cv2.rectangle(frame, drag_start, drag_cur, (255, 0, 255), 1)
    cv2.putText(frame, "Click/Drag: select  SPACE: pause  C: clear  ESC: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)