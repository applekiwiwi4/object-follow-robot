import cv2
import numpy as np
from ultralytics import YOLO

img = cv2.imread("./image/3people.jpg")
if img is None:
    print("사진을 찾을 수 없습니다.")
    exit()

# 세그멘테이션 모델 (처음 실행 때 자동 다운로드)
model = YOLO("yolov8n-seg.pt")
results = model(img, classes=[0], conf=0.4, iou=0.5)
r = results[0]

if r.masks is None:
    print("사람을 찾지 못했습니다.")
    exit()

# 1. 반투명 색칠용 복사본에 사람 모양을 채워 칠하기
overlay = img.copy()
for poly in r.masks.xy:
    if len(poly) == 0:
        continue
    pts = poly.astype(np.int32)
    cv2.fillPoly(overlay, [pts], (0, 255, 0))

# 2. 원본과 섞어서 반투명하게 만들기
result = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

# 3. 윤곽선과 번호 그리기
for i, poly in enumerate(r.masks.xy):
    if len(poly) == 0:
        continue
    pts = poly.astype(np.int32)
    cv2.polylines(result, [pts], True, (0, 255, 0), 2)
    x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[i].tolist()]
    cv2.putText(result, f"ID {i + 1}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print(f"ID {i + 1}: 윤곽 점 {len(poly)}개")

cv2.imshow("seg test", result)
cv2.waitKey(0)
cv2.destroyAllWindows()