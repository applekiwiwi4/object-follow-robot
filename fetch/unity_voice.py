# 음성 요청 전용: Python 은 "귀"만 맡고, 심부름 판단과 운전은 Unity 머리(FetchBrain)가 한다
#
#   V 키 -> 3초 동안 말하기 ("컵 가져다줘", "리모컨 가져와", "취소")
#   1 ~ 5 키, C 키도 됨 (이 창을 클릭한 뒤)
#   ESC : 종료
#
# 실행 순서: 이 프로그램 먼저 -> Unity ▶ -> 왼쪽 위가 [UNITY BRAIN + VOICE] 인지 확인 -> V
#
# unity_fetch.py 와 다른 점
#   unity_fetch.py : Python 이 운전까지 함 (Python 머리)
#   unity_voice.py : Python 은 요청만 보내고, Unity 머리가 거리 센서까지 써서 운전 (추천)

import time

import cv2

import command_box
from unity_detect import ITEM_NAMES, UNITY_COLOR, draw
from unity_link import PORT, open_server, recv_packet, send_reply
from voice_input import VoiceInput

WINDOW = "Unity Robot Eye - Voice"
ELDER_COLOR = (200, 100, 255)


def new_request_id():
    """요청마다 겹치지 않는 번호 (Python 을 다시 켜도 겹치지 않게 시간으로 만듦)"""
    return int(time.time() * 1000) % 2_000_000_000


def main():
    voice = VoiceInput()
    server = open_server()

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 640, 480)

    state = {"quit": False, "req": None, "sent": False, "last": "-"}

    def push(kind, item, source):
        """Unity 로 보낼 요청 만들기. 같은 요청은 Unity 가 id 로 한 번만 실행함"""
        state["req"] = {"id": new_request_id(), "type": kind, "item": item or ""}
        state["sent"] = False
        state["last"] = f"[{source}] {kind} {item or ''}"
        print(f"[{source}] 요청 -> Unity:", kind, item or "")

    def pump():
        """명령 상자(우편함)에 쌓인 요청을 Unity 로 보낼 요청으로 옮기기"""
        for c in command_box.get_all():
            push(c["type"], c.get("item"), c["source"])

    def handle_key(key):
        if key == 27:
            state["quit"] = True
        elif key in (ord("v"), ord("V")):
            voice.start_listening()
        elif key in (ord("c"), ord("C")):
            command_box.put_cancel("keyboard")
        elif ord("1") <= key <= ord("5"):
            command_box.put_fetch(ITEM_NAMES[key - ord("1")], "keyboard")

    def on_idle():
        """Unity 를 기다리는 동안에도 창과 음성이 살아 있게"""
        handle_key(cv2.waitKey(20) & 0xFF)
        pump()
        return state["quit"]

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

        print("Unity 연결됨:", addr, "-> 이 창을 클릭하고 V 를 누른 뒤 말하세요")
        conn.settimeout(0.05)
        if state["sent"]:     # 이전 Unity 에 이미 보낸 요청은 새 Unity 에 다시 보내지 않음
            state["req"] = None

        while not state["quit"]:
            frame, info = recv_packet(conn, on_idle)
            if frame is None:
                if not state["quit"]:
                    print("Unity 연결 끊김 (▶ 를 멈춘 경우 정상)")
                break

            pump()

            items = [it for it in info["items"] if it["name"] in ITEM_NAMES]
            elders = [it for it in info["items"] if it["name"] == "elder"]
            draw(frame, items, UNITY_COLOR, "unity")
            draw(frame, elders, ELDER_COLOR, "unity")

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, voice.status, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, "last: " + state["last"], (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "V: speak   1-5: item   C: cancel   ESC: quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(WINDOW, frame)

            # 운전 명령(cmd)은 보내지 않음 -> Unity 머리가 운전
            reply = {"items": items}
            if state["req"] is not None:
                reply["request"] = state["req"]
                state["sent"] = True
            send_reply(conn, reply)

            handle_key(cv2.waitKey(1) & 0xFF)

        conn.close()

    server.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()