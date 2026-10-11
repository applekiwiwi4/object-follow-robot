# Unity 한 화면으로 조종하기: Python 은 창 없이 뒤에서 "귀(음성)"와 "눈(손짓)"만 맡는다
#
#   이 프로그램은 실행만 해 두고 터미널은 최소화해도 됨
#   조종은 전부 Unity Game 화면에서:
#     V : 3초 동안 말하기 ("컵 가져다줘", "따라와", "그만", "취소")
#     H : 손짓 웹캠 켜기/끄기 (한 손 들기 = 따라와, 두 손 들기 = 멈춤)
#   웹캠 화면과 음성 상태는 Unity 화면에 나옴
#   종료: 터미널에서 Ctrl + C
#
# 실행 순서는 상관없음 (Unity 가 1초마다 다시 연결을 시도함)
#
# unity_voice.py 와 다른 점
#   unity_voice.py  : Python 창이 따로 뜸 (로봇 눈 화면 확인, 문제 찾을 때 좋음)
#   robot_helper.py : Python 창 없음. Unity 화면 하나로 조종 (발표, 시연용)

import base64
import time

import cv2

import command_box
from gesture_input import GestureInput
from unity_detect import ITEM_NAMES
from unity_link import PORT, open_server, recv_packet, send_reply
from voice_input import VoiceInput

CAM_W, CAM_H = 240, 180     # Unity 에 보낼 웹캠 작은 사진 크기
CAM_JPG_QUALITY = 60


def new_request_id():
    """요청마다 겹치지 않는 번호 (Python 을 다시 켜도 겹치지 않게 시간으로 만듦)"""
    return int(time.time() * 1000) % 2_000_000_000


def small_cam_jpg(frame):
    """손짓 웹캠 그림 -> 작은 JPG -> 글자(base64). JSON 에 넣으려고 글자로 바꿈"""
    small = cv2.resize(frame, (CAM_W, CAM_H))
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, CAM_JPG_QUALITY])
    if not ok:
        return ""
    return base64.b64encode(jpg.tobytes()).decode("ascii")


def main():
    # 포트(5005)를 먼저 잡기. 예전 Python 이 아직 켜져 있으면 여기서 실패함
    try:
        server = open_server()
    except OSError as e:
        print("[오류] 포트", PORT, "를 이미 다른 Python 이 쓰고 있어요.")
        print("       작업 관리자(Ctrl+Shift+Esc)에서 python.exe 를 모두 끄고 다시 ▶ 하세요.")
        print("       자세한 내용:", e)
        return

    voice = VoiceInput()
    gesture = GestureInput()

    state = {"req": None, "sent": False, "listen": None, "gesture_want": False}

    def push(kind, item, source):
        """Unity 로 보낼 요청 만들기. 같은 요청은 Unity 가 id 로 한 번만 실행함"""
        state["req"] = {"id": new_request_id(), "type": kind, "item": item or "", "source": source}
        state["sent"] = False
        print(f"[{source}] 요청 -> Unity:", kind, item or "")

    def pump():
        """명령 상자(우편함)에 쌓인 요청을 Unity 로 보낼 요청으로 옮기기"""
        for c in command_box.get_all():
            push(c["type"], c.get("item"), c["source"])

    def follow_unity_keys(info):
        """Unity 에서 누른 V, H 를 따라 하기"""
        # V: listen 숫자가 바뀌면 듣기 시작
        listen = info.get("listen", 0)
        if state["listen"] is None:
            state["listen"] = listen              # 처음 연결됐을 때 숫자는 그냥 기억만
        elif listen != state["listen"]:
            state["listen"] = listen
            voice.start_listening()

        # H: Unity 에서 켜짐/꺼짐이 "바뀌었을 때만" 따라 하기
        #    (웹캠이 고장 나서 꺼져도 계속 다시 켜려고 하지 않게)
        want = bool(info.get("gesture", False))
        if want != state["gesture_want"]:
            state["gesture_want"] = want
            if want != gesture.running:
                gesture.toggle()

    def on_idle():
        """Unity 를 기다리는 동안에도 음성, 손짓 결과를 모아 두기"""
        time.sleep(0.02)
        pump()
        return False

    server.settimeout(0.05)
    print("=" * 60)
    print(" robot_helper 실행 중. 이제 Unity 에서 ▶ 를 누르세요")
    print(" 조종은 Unity Game 화면에서: V = 말하기, H = 손짓 웹캠")
    print(" 종료: Ctrl + C")
    print("=" * 60)

    try:
        while True:
            conn = None
            while conn is None:
                try:
                    conn, addr = server.accept()
                except OSError:
                    on_idle()

            print("Unity 연결됨:", addr)
            conn.settimeout(0.05)
            state["listen"] = None
            state["gesture_want"] = False
            if state["sent"]:     # 이전 Unity 에 이미 보낸 요청은 새 Unity 에 다시 보내지 않음
                state["req"] = None

            while True:
                frame, info = recv_packet(conn, on_idle)
                if frame is None:
                    print("Unity 연결 끊김 (▶ 를 멈춘 경우 정상). 다시 ▶ 를 누르면 이어서 됨")
                    break

                follow_unity_keys(info)
                pump()

                items = [it for it in info.get("items", []) if it["name"] in ITEM_NAMES]
                reply = {
                    "items": items,
                    "status": {"voice": voice.status, "gesture": gesture.status},
                    "cam": "",
                }
                if gesture.running and gesture.frame is not None:
                    reply["cam"] = small_cam_jpg(gesture.frame)
                if state["req"] is not None:
                    reply["request"] = state["req"]
                    state["sent"] = True
                send_reply(conn, reply)

            conn.close()
            if gesture.running:       # Unity 가 꺼지면 웹캠도 끔
                gesture.toggle()
    except KeyboardInterrupt:
        print("종료합니다")
    finally:
        gesture.running = False
        server.close()


if __name__ == "__main__":
    main()