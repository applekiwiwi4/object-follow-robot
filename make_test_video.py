# 테스트 영상 만들기 (11단계 다시 잡기 + 12단계 옷 색깔 비교 확인용)
#
# ./image/3people.jpg 에서 YOLO 세그멘테이션으로 사람을 오려 낸 뒤,
# 기둥 뒤로 숨었다가 나오는 장면을 합성해서 ./video/test_reid.mp4 로 저장합니다.
#
# 등장인물
#   A : 사진의 맨 왼쪽 사람  -> 이 사람을 선택해서 따라가게 하세요
#   B : 사진의 두 번째 사람  -> 몸통을 파랗게 물들여 A와 옷 색이 확실히 다르게 함
#   C : 사진의 세 번째 사람  -> 오른쪽에 서 있기만 하는 구경꾼
#
# 장면
#   장면 1 (0 ~ 5.5초) : A가 기둥 뒤에 2초 숨었다가, 들어간 쪽(왼쪽)으로 다시 나옴
#                        -> 11단계: 같은 사람을 다시 잡는지 확인
#   장면 2 (5.5 ~ 12초): A가 다시 기둥 뒤로 숨음
#                        -> A보다 먼저 옷 색이 다른 B가 기둥 왼쪽으로 나옴 (함정)
#                        -> 잠시 뒤 진짜 A가 나옴
#                        -> 12단계: B는 무시하고 A를 다시 잡는지 확인

import os
import cv2
import numpy as np

# ============================================================
# 설정값
# ============================================================
SRC_IMG = "./image/3people.jpg"
OUT_PATH = "./video/test_reid.mp4"

W, H = 960, 540        # 영상 크기
FPS = 30               # 1초에 몇 장
DURATION = 12.0        # 영상 길이 (초)
FLOOR_Y = 500          # 바닥 높이 (사람 발 위치)
PERSON_H = 320         # 영상 속 사람 키 (픽셀)
PILLAR_CX = 480        # 기둥 가운데 x 위치

TINT_B = True                  # B의 몸통을 물들일지
TINT_COLOR = (255, 120, 0)     # 물들일 색 (BGR, 파란색 계열)
TINT_STRENGTH = 0.6            # 물들이는 정도 (0 ~ 1)


# ============================================================
# 1. 사진에서 사람 오려 내기
# ============================================================
def cut_people(img_path):
    from ultralytics import YOLO

    img = cv2.imread(img_path)
    if img is None:
        print("사진을 찾을 수 없습니다:", img_path)
        return []

    model = YOLO("yolov8n-seg.pt")
    r = model(img, classes=[0], conf=0.4, iou=0.5,
              retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []

    masks = r.masks.data.cpu().numpy()
    found = []
    for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = box.astype(int)
        m = masks[i]
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]))

        crop = img[y1:y2, x1:x2]
        alpha = (m[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255

        # 키를 PERSON_H 로 맞추기
        scale = PERSON_H / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
        alpha = cv2.resize(alpha, (crop.shape[1], crop.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        found.append((x1, crop, alpha))

    # 사진 왼쪽 사람부터 순서대로
    found.sort(key=lambda p: p[0])
    return [(crop, alpha) for _, crop, alpha in found]


# ============================================================
# 2. 그리기 도우미 함수
# ============================================================
def tint_body(sprite, alpha):
    """머리(위쪽 20%)를 뺀 몸통을 TINT_COLOR 쪽으로 물들이기"""
    out = sprite.copy()
    h = out.shape[0]
    body = np.zeros(alpha.shape, dtype=bool)
    top = int(h * 0.2)
    body[top:] = alpha[top:] > 0
    color = np.array(TINT_COLOR, dtype=np.float32)
    out[body] = (out[body] * (1 - TINT_STRENGTH)
                 + color * TINT_STRENGTH).astype(np.uint8)
    return out


def paste(frame, sprite, alpha, x, y):
    """사람 그림을 (x, y) 위치에 붙이기. 화면 밖으로 나간 부분은 잘라 냄"""
    sh, sw = sprite.shape[:2]
    fx1, fy1 = max(int(x), 0), max(int(y), 0)
    fx2 = min(int(x) + sw, frame.shape[1])
    fy2 = min(int(y) + sh, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return
    sx1, sy1 = fx1 - int(x), fy1 - int(y)
    sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)
    region = frame[fy1:fy2, fx1:fx2]
    m = alpha[sy1:sy2, sx1:sx2] > 0
    region[m] = sprite[sy1:sy2, sx1:sx2][m]


def make_background():
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        if y < FLOOR_Y:
            c = 215 - y * 50 // FLOOR_Y          # 벽: 위쪽이 밝게
            bg[y] = (c, c, c + 5)
        else:
            bg[y] = (125, 135, 145)              # 바닥
    cv2.line(bg, (0, FLOOR_Y), (W, FLOOR_Y), (100, 100, 110), 2)
    return bg


def x_at(t, keys):
    """시간 t 에서의 x 위치 (keys 사이를 일정한 속도로 이동)"""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, x0), (t1, x1) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            return x0 + (x1 - x0) * (t - t0) / (t1 - t0)
    return keys[-1][1]


# ============================================================
# 3. 영상 만들기
# ============================================================
def build_video(people):
    if len(people) < 2:
        print("사람이 2명 이상 필요합니다. 찾은 사람 수:", len(people))
        return

    a_img, a_alpha = people[0]
    b_img, b_alpha = people[1]
    if TINT_B:
        b_img = tint_body(b_img, b_alpha)

    wa = a_img.shape[1]
    wb = b_img.shape[1]

    # 기둥은 A, B가 완전히 숨을 수 있을 만큼 넓게
    pillar_w = max(wa, wb) + 40
    px1 = PILLAR_CX - pillar_w // 2
    px2 = PILLAR_CX + pillar_w // 2

    hide_a = PILLAR_CX - wa / 2     # 기둥 뒤에 완전히 숨는 위치
    hide_b = PILLAR_CX - wb / 2
    start_x = 40

    # (시간, x위치) 목록
    keys_a = [(0.0, start_x), (1.5, hide_a), (3.5, hide_a),      # 장면 1
              (5.0, start_x), (5.5, start_x),
              (7.0, hide_a), (9.0, hide_a),                       # 장면 2
              (10.5, start_x), (DURATION, start_x)]
    keys_b = [(0.0, hide_b), (8.2, hide_b),                       # 기둥 뒤에 계속 숨어 있다가
              (10.2, -wb - 20), (DURATION, -wb - 20)]             # 왼쪽으로 걸어 나감

    bg = make_background()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

    total = int(DURATION * FPS)
    for n in range(total):
        t = n / FPS
        frame = bg.copy()

        # 구경꾼 C (있으면): 오른쪽에 서 있기
        if len(people) >= 3:
            c_img, c_alpha = people[2]
            paste(frame, c_img, c_alpha,
                  W - c_img.shape[1] - 50, FLOOR_Y - c_img.shape[0])

        # B, A 그리기 (기둥보다 먼저 그려야 기둥 뒤에 숨음)
        paste(frame, b_img, b_alpha, x_at(t, keys_b), FLOOR_Y - b_img.shape[0])
        paste(frame, a_img, a_alpha, x_at(t, keys_a), FLOOR_Y - a_img.shape[0])

        # 기둥
        cv2.rectangle(frame, (px1, 0), (px2, FLOOR_Y + 8), (95, 95, 105), -1)
        cv2.rectangle(frame, (px1, 0), (px2, FLOOR_Y + 8), (70, 70, 80), 3)

        # 장면 안내 글자 (프로그램의 CMD 줄에 가리지 않게 조금 아래에)
        scene = "Scene 1: A hides, then comes back" if t < 5.5 \
            else "Scene 2: B (different color) comes out first, then A"
        cv2.putText(frame, scene, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (40, 40, 40), 2)
        cv2.putText(frame, f"t = {t:4.1f}s", (W - 130, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

        writer.write(frame)

    writer.release()
    print("완성:", OUT_PATH, f"({DURATION:.0f}초, {total}장)")
    print("A = 사진 맨 왼쪽 사람, B = 두 번째 사람(몸통 파랗게), C = 세 번째 사람")


if __name__ == "__main__":
    people = cut_people(SRC_IMG)
    print("오려 낸 사람 수:", len(people))
    build_video(people)
