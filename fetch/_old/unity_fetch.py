# 심부름 로봇: 소지품 찾기 -> 집기 -> 어르신에게 가져가기 -> 내려놓기
#
# 실행 순서: 이 프로그램 먼저 -> Unity ▶ -> Game 화면 클릭 -> Tab (자동 모드)
#
# Python 창 키 (창을 클릭한 뒤)
#   1 ~ 5 : 요청 (1 cup, 2 remote, 3 phone, 4 book, 5 bottle)
#   C     : 취소
#   ESC   : 종료
#
# Unity 설정 필수: Edit > Project Settings > Player > Resolution and Presentation
#                  > Run In Background 체크 (Python 창을 클릭해도 Unity가 멈추지 않게)

import cv2
import command_box
from voice_input import VoiceInput

from fetch_task import ELDER_NAME, FetchTask
from unity_detect import ITEM_NAMES, UNITY_COLOR, draw
from unity_link import PORT, open_server, recv_packet, send_reply

WINDOW = "Unity Robot Eye - Fetch"
ELDER_COLOR = (200, 100, 255)   # 어르신: 분홍


def split_items(info):
    """Unity 정답을 소지품과 어르신으로 나누기"""
    items, elders = [], []
    for it in info["items"]:
        entry = {"name": it["name"], "conf": 1.0, "box": it["box"]}
        if it["name"] == ELDER_NAME:
            elders.append(entry)
        elif it["name"] in ITEM_NAMES:
            items.append(entry)
    return items, elders


def draw_status(frame, cmd, holding):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, cmd["state"], (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, f"holding: {holding if holding else '-'}   "
                       f"move {cmd['move']:+.2f}  turn {cmd['turn']:+.2f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "1 cup  2 remote  3 phone  4 book  5 bottle   C: cancel  ESC: quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.line(frame, (w // 2, 65), (w // 2, h - 25), (0, 200, 255), 1)


def main():
    task = FetchTask()
    voice = VoiceInput()
    server = open_server()

    # 창을 미리 만들어 두기 (마우스로 크기 조절 가능)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 480)

    state = {"quit": False}

    def handle_key(key):
        """키 처리. 메인 반복에서도, 기다리는 동안에도 같이 쓴다."""
        if key == 27:
            state["quit"] = True
        elif key in (ord("c"), ord("C")):
           command_box.put_cancel("keyboard")
        elif ord("1") <= key <= ord("5"):
            command_box.put_fetch(ITEM_NAMES[key - ord("1")], "keyboard")
        elif key in(ord("v"), ord("V")):
            voice.start_listening()


    def on_idle():
        """Unity 사진을 기다리는 동안: 창을 살려 두고 키도 받기"""
        handle_key(cv2.waitKey(20) & 0xFF)
        return state["quit"]

    # 연결 기다리는 동안에도 창이 얼지 않게
    server.settimeout(0.05)

    while not state["quit"]:
        print(f"Unity 연결 기다리는 중... (포트 {PORT}) 이제 Unity에서 ▶ 를 누르세요")
        conn = None
        while conn is None and not state["quit"]:
            try:
                conn, addr = server.accept()
            except OSError:
                on_idle()
        if conn is None:
            break

        print("Unity 연결됨:", addr, "-> Game 화면 클릭 후 Tab (자동 모드), 이 창에서 1~5 로 요청")
        conn.settimeout(0.05)

        while not state["quit"]:
            frame, info = recv_packet(conn, on_idle)
            if frame is None:
                if not state["quit"]:
                    print("Unity 연결 끊김 (▶ 를 멈춘 경우 정상)")
                break

            items, elders = split_items(info)
            holding = info.get("holding", "")
            h, w = frame.shape[:2]

            for c in command_box.get_all():
                print(f"[{c['source']}] 요청", c["type"], c.get("item", ""))
                if c["type"] == "fetch":
                    task.request(c["item"])
                elif c["type"] == "cancel":
                    task.cancel()
            cmd = task.update(items + elders, holding, w, h)

            draw(frame, items, UNITY_COLOR, "unity")
            draw(frame, elders, ELDER_COLOR, "unity")
            draw_status(frame, cmd, holding)

            cv2.putText(frame, voice.status, (w - 220, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(WINDOW, frame)

            send_reply(conn, {"items": items, "cmd": cmd})
            handle_key(cv2.waitKey(1) & 0xFF)

        conn.close()

    server.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()