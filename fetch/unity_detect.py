# Unity 로봇 눈 화면에서 소지품 찾기
#
# 실행 순서: 이 프로그램 먼저 -> Unity ▶
# 종료: 영상 창을 클릭하고 ESC
#
# DETECT_MODE
#   "unity" : Unity가 알려 준 정답 위치를 사용 (지금 추천. YOLO가 Unity 물건을 못 알아볼 때)
#   "yolo"  : YOLO로 직접 찾기
#   "both"  : 둘 다 화면에 표시해서 비교 (정답=파랑, YOLO=초록). 결과는 정답 사용

import os

import cv2

from unity_link import PORT, open_server, recv_packet, send_reply

# ============================================================
# 설정값
# ============================================================
DETECT_MODE = "unity"

MODEL_PATH = "yolov8s.pt"   # "yolo", "both" 모드에서 쓰는 모델
CONF = 0.3

# 소지품 이름 목록 (Unity ItemLabel의 itemName과 같게)
ITEM_NAMES = ["cup", "remote", "phone", "book", "bottle"]

# YOLO 이름 -> 우리 소지품 이름
YOLO_TO_ITEM = {
    "cup": "cup",
    "remote": "remote",
    "cell phone": "phone",
    "book": "book",
    "bottle": "bottle",
}

# 학습 자료 모으기 (나중에 YOLO 학습용). True면 사진과 정답을 저장
SAVE_DATASET = False
DATASET_DIR = "./dataset"
SAVE_EVERY = 5              # 몇 장마다 한 장씩 저장할지

UNITY_COLOR = (255, 150, 0)   # 정답: 파랑
YOLO_COLOR = (0, 255, 0)      # YOLO: 초록


# ============================================================
# 찾기
# ============================================================
def items_from_unity(info):
    """Unity가 알려 준 정답 위치"""
    return [{"name": it["name"], "conf": 1.0, "box": it["box"]}
            for it in info["items"] if it["name"] in ITEM_NAMES]


def items_from_yolo(model, frame):
    """YOLO로 찾은 소지품"""
    r = model(frame, conf=CONF, verbose=False)[0]
    found = []
    for b in r.boxes:
        name = model.names[int(b.cls[0])]
        if name not in YOLO_TO_ITEM:
            continue
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        found.append({"name": YOLO_TO_ITEM[name],
                      "conf": round(float(b.conf[0]), 2),
                      "box": [x1, y1, x2, y2]})
    return found


def draw(frame, items, color, tag):
    for it in items:
        x1, y1, x2, y2 = it["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{tag}:{it['name']}"
        if tag == "yolo":
            text += f" {it['conf']:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# ============================================================
# 학습 자료 저장 (YOLO 형식)
# ============================================================
def save_sample(frame, items, index):
    img_dir = os.path.join(DATASET_DIR, "images")
    lbl_dir = os.path.join(DATASET_DIR, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    h, w = frame.shape[:2]
    name = f"unity_{index:06d}"
    cv2.imwrite(os.path.join(img_dir, name + ".jpg"), frame)
    with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
        for it in items:
            x1, y1, x2, y2 = it["box"]
            cls = ITEM_NAMES.index(it["name"])
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ============================================================
# 메인
# ============================================================
def main():
    model = None
    if DETECT_MODE in ("yolo", "both"):
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)

    server = open_server()
    print("찾기 방식:", DETECT_MODE)

    count = 0
    saved = 0
    while True:
        print(f"Unity 연결 기다리는 중... (포트 {PORT}) 이제 Unity에서 ▶ 를 누르세요")
        conn, addr = server.accept()
        print("Unity 연결됨:", addr)

        last_names = None
        quit_all = False
        while True:
            frame, info = recv_packet(conn)
            if frame is None:
                print("Unity 연결 끊김 (▶ 를 멈춘 경우 정상)")
                break
            count += 1
            clean = frame.copy()

            unity_items = items_from_unity(info)
            yolo_items = items_from_yolo(model, frame) if model else []

            if DETECT_MODE == "yolo":
                items = yolo_items
            else:
                items = unity_items

            # 학습 자료 저장 (정답 기준, 소지품이 보일 때만)
            if SAVE_DATASET and unity_items and count % SAVE_EVERY == 0:
                save_sample(clean, unity_items, count)
                saved += 1
                if saved % 20 == 0:
                    print("학습 자료 저장:", saved, "장")

            # 보이는 소지품이 바뀔 때만 출력
            names = sorted(it["name"] for it in items)
            if names != last_names:
                print("보이는 소지품:", names if names else "없음")
                last_names = names

            if DETECT_MODE in ("unity", "both"):
                draw(frame, unity_items, UNITY_COLOR, "unity")
            if DETECT_MODE in ("yolo", "both"):
                draw(frame, yolo_items, YOLO_COLOR, "yolo")
            cv2.putText(frame, f"mode: {DETECT_MODE}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Unity Robot Eye - Detect", frame)

            send_reply(conn, {"items": items})

            if cv2.waitKey(1) & 0xFF == 27:
                quit_all = True
                break

        conn.close()
        if quit_all:
            break

    server.close()
    cv2.destroyAllWindows()
    if SAVE_DATASET:
        print("저장한 학습 자료:", saved, "장 ->", DATASET_DIR)


if __name__ == "__main__":
    main()