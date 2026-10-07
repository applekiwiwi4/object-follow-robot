# 소지품 찾아서 다가가기 (Unity 로봇 자동 운전)
#
# 실행 순서: 이 프로그램 먼저 -> Unity ▶ -> Game 화면 클릭 -> Tab (자동 모드)
#
# Python 창 키
#   1 ~ 5 : 목표 소지품 바꾸기 (cup, remote, phone, book, bottle)
#   R     : 처음부터 다시 찾기
#   ESC   : 종료

import cv2

from approach import Approacher
from unity_detect import ITEM_NAMES, UNITY_COLOR, draw, items_from_unity
from unity_link import PORT, open_server, recv_packet, send_reply

TARGET = "cup"   # 처음 목표


def draw_status(frame, approacher, cmd):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame, f"target: {approacher.target}   state: {cmd['state']}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"move {cmd['move']:+.2f}  turn {cmd['turn']:+.2f}",
                (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "1-5: target  R: restart  ESC: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # 화면 가운데 세로선 (로봇이 맞추려는 기준)
    cv2.line(frame, (w // 2, 40), (w // 2, h - 50), (0, 200, 255), 1)


def main():
    approacher = Approacher(TARGET)
    server = open_server()

    while True:
        print(f"Unity 연결 기다리는 중... (포트 {PORT}) 이제 Unity에서 ▶ 를 누르세요")
        conn, addr = server.accept()
        print("Unity 연결됨:", addr, "-> Unity Game 화면을 클릭하고 Tab 을 누르면 자동 운전")
        approacher.restart()

        last_state = None
        quit_all = False
        while True:
            frame, info = recv_packet(conn)
            if frame is None:
                print("Unity 연결 끊김 (▶ 를 멈춘 경우 정상)")
                break

            items = items_from_unity(info)
            h, w = frame.shape[:2]
            cmd = approacher.update(items, w, h)

            if cmd["state"] != last_state:
                print("상태:", cmd["state"])
                last_state = cmd["state"]

            draw(frame, items, UNITY_COLOR, "unity")
            draw_status(frame, approacher, cmd)
            cv2.imshow("Unity Robot Eye - Fetch", frame)

            send_reply(conn, {"items": items, "cmd": cmd})

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                quit_all = True
                break
            if key in (ord("r"), ord("R")):
                approacher.restart()
                print("다시 찾기:", approacher.target)
            if ord("1") <= key <= ord("5"):
                approacher.restart(ITEM_NAMES[key - ord("1")])
                print("목표 변경:", approacher.target)

        conn.close()
        if quit_all:
            break

    server.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
