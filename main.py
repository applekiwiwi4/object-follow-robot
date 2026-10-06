# 사람 추적 + 따라가는 로봇 (리팩터링 버전)
#
# 조작법: 클릭/드래그 = 선택, SPACE = 일시정지, C = 선택 해제, ESC = 종료

import cv2

import config
from follow import drawing
from follow.detector import Detector
from follow.reid import ReIdentifier
from follow.robot import Robot
from follow.selector import Selector


def resize_frame(frame):
    """화면이 너무 크면 가로 MAX_WIDTH에 맞춰 줄인다."""
    if frame.shape[1] > config.MAX_WIDTH:
        scale = config.MAX_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
    return frame


def main():
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    if not cap.isOpened():
        print("영상을 열 수 없습니다. 경로를 확인하세요:", config.VIDEO_PATH)
        return

    # 부품 준비
    detector = Detector()
    reid = ReIdentifier()
    selector = Selector(reid.id_map)
    robot = Robot()

    cv2.namedWindow("follow")
    cv2.setMouseCallback("follow", selector.on_mouse)

    raw = None
    boxes, polys = [], {}
    paused = False
    last_cmd = None

    while True:
        new_frame = False

        # 1. 새 화면 읽기 + 사람 감지 (일시정지면 건너뜀)
        if not paused or raw is None:
            ok, raw = cap.read()
            if not ok:
                print("영상이 끝났습니다.")
                break
            raw = resize_frame(raw)
            boxes, polys = detector.detect(raw)
            selector.boxes = boxes
            new_frame = True

        h, w = raw.shape[:2]

        # 2. 놓친 사람 다시 잡기
        target_box = selector.selected_box()
        new_id = reid.update(selector.selected, target_box, boxes, polys,
                             raw, new_frame)
        if new_id is not None:
            selector.selected = new_id
            robot.keep_following(new_id)

        # 3. 로봇 이동 + 명령 판단
        cmd = robot.update(selector.selected,
                           reid.show_id(selector.selected),
                           target_box, w, h, new_frame,
                           reid.is_searching())
        if cmd != last_cmd:
            print("명령:", cmd)
            last_cmd = cmd

        # 4. 그리기
        frame = drawing.draw_people(raw.copy(), boxes, polys,
                                    selector.selected, reid.show_id,
                                    selector.scores)
        if robot.pos is not None:
            drawing.draw_robot(frame, robot.pos, robot.target)
        drawing.draw_command(frame, cmd)
        drawing.draw_status(frame, paused, selector.dragging,
                            selector.drag_start, selector.drag_cur)
        cv2.imshow("follow", frame)

        # 5. 키 입력
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


if __name__ == "__main__":
    main()