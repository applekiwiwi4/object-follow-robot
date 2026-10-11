# 손짓 요청: 웹캠으로 사람 관절(어깨, 손목)을 보고 -> 손을 들었는지 판단 -> 명령 상자에 넣기
#
#   한 손 들기 (1초 유지) -> "따라와"   (follow)
#   두 손 들기 (1초 유지) -> "모두 멈춤" (stop)
#
# 관절 찾기: YOLO 포즈 모델 (yolov8n-pose.pt, 처음 실행할 때 자동으로 내려받음)
#   이미 설치한 ultralytics 로 바로 됨. 추가 설치 없음
#
# 사용법: unity_voice.py 창에서 H 키 -> 웹캠 켜기/끄기

import os
import threading
import time

import cv2

import command_box

# ============================================================
# 설정값
# ============================================================
CAMERA_INDEX = 0          # 웹캠 번호 (안 되면 1 로 바꿔 보기)
MODEL_NAME = "yolov8n-pose.pt"
KP_CONF = 0.5             # 관절 점수가 이보다 낮으면 못 본 것으로 침
HOLD_SEC = 1.0            # 같은 손짓을 이만큼 유지해야 명령으로 인정
COOLDOWN_SEC = 2.0        # 명령을 보낸 뒤 이 시간 동안은 새 명령 안 받음
RAISE_MARGIN = 0.3        # 손목이 어깨보다 (어깨 너비 x 이 값) 이상 위에 있으면 "들었다"

# COCO 관절 번호 (YOLO 포즈 모델이 주는 17개 점 중 필요한 것)
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10

# 손짓 -> 명령
GESTURE_TO_COMMAND = {
    "one": "follow",     # 한 손
    "both": "stop",      # 두 손
}


def classify(kp, conf):
    """
    관절 하나(사람 1명)를 보고 손짓 판단.
    kp: (17, 2) 좌표, conf: (17,) 점수
    반환: "none" / "one" / "both"
    """
    # 어깨 두 개가 다 보여야 판단 가능
    if conf[L_SHOULDER] < KP_CONF or conf[R_SHOULDER] < KP_CONF:
        return "none"

    shoulder_y = (kp[L_SHOULDER][1] + kp[R_SHOULDER][1]) / 2
    shoulder_w = abs(kp[L_SHOULDER][0] - kp[R_SHOULDER][0])
    margin = max(shoulder_w * RAISE_MARGIN, 10)   # 너무 작아지지 않게 최소 10픽셀

    def raised(wrist):
        # 화면 좌표는 위로 갈수록 y 가 작다 -> 어깨보다 y 가 margin 이상 작으면 들었음
        return conf[wrist] >= KP_CONF and kp[wrist][1] < shoulder_y - margin

    up = int(raised(L_WRIST)) + int(raised(R_WRIST))
    if up == 2:
        return "both"
    if up == 1:
        return "one"
    return "none"


class GestureInput:
    """웹캠 + 포즈 모델을 따로 일꾼(스레드)에서 돌리고, 손짓이 확인되면 명령 상자에 넣는다."""

    def __init__(self):
        self.status = "gesture: off (H)"
        self.frame = None          # 화면에 보여 줄 최신 그림 (메인 창에서 imshow)
        self.running = False
        self._thread = None
        self._model = None

    # ---------------- 켜기 / 끄기 ----------------
    def toggle(self):
        if self.running:
            self.running = False
            self.status = "gesture: off (H)"
            print("[gesture] 웹캠 끔")
        else:
            self.running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    # ---------------- 일꾼이 하는 일 ----------------
    def _run(self):
        try:
            if self._model is None:
                self.status = "gesture: loading..."
                from ultralytics import YOLO
                self._model = YOLO(MODEL_NAME)

            # 윈도우에서는 DSHOW 가 웹캠을 빨리 엶
            if os.name == "nt":
                cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                print(f"[gesture] 웹캠 {CAMERA_INDEX} 번을 열 수 없어요 (CAMERA_INDEX 를 바꿔 보세요)")
                self.status = "gesture: no camera"
                self.running = False
                return

            print("[gesture] 웹캠 켜짐. 한 손 들기 = 따라와, 두 손 들기 = 멈춤 (1초 유지)")
            current = "none"         # 지금 하고 있는 손짓
            since = time.time()      # 그 손짓을 언제부터 했는지
            last_sent = 0.0          # 마지막으로 명령 보낸 시각
            fired = False            # 이번 손짓으로 이미 명령을 보냈는지 (손 내릴 때까지 한 번만)

            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)   # 거울처럼 보이게

                r = self._model(frame, verbose=False)[0]
                gesture = self._gesture_of_biggest_person(r)
                view = r.plot()              # 관절 그린 그림

                # 같은 손짓이 HOLD_SEC 동안 이어졌는지
                now = time.time()
                if gesture != current:
                    current = gesture
                    since = now
                    fired = False
                held = now - since

                if (current in GESTURE_TO_COMMAND and not fired
                        and held >= HOLD_SEC and now - last_sent >= COOLDOWN_SEC):
                    cmd = GESTURE_TO_COMMAND[current]
                    if cmd == "follow":
                        command_box.put_follow("gesture")
                    elif cmd == "stop":
                        command_box.put_stop("gesture")
                    print(f"[gesture] 손짓 '{current}' -> {cmd}")
                    last_sent = now
                    fired = True

                # 화면 글씨
                if current in GESTURE_TO_COMMAND and not fired:
                    bar = min(held / HOLD_SEC, 1.0)
                    text = f"{current} hand(s) up... {int(bar * 100)}%"
                elif fired:
                    text = f"sent: {GESTURE_TO_COMMAND[current]} (put hands down)"
                else:
                    text = "1 hand up = follow   2 hands up = stop"
                cv2.rectangle(view, (0, 0), (view.shape[1], 32), (0, 0, 0), -1)
                cv2.putText(view, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                self.frame = view
                self.status = f"gesture: {current}"

            cap.release()
        except Exception as e:
            print("[gesture] 오류:", e)
            self.status = "gesture: error"
            self.running = False

    def _gesture_of_biggest_person(self, r):
        """화면에서 가장 크게 보이는 사람(가장 가까운 사람) 한 명의 손짓"""
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return "none"
        boxes = r.boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(areas.argmax())

        kp = r.keypoints.xy[i].cpu().numpy()
        if r.keypoints.conf is None:
            return "none"
        conf = r.keypoints.conf[i].cpu().numpy()
        return classify(kp, conf)