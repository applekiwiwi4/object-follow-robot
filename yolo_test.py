import cv2

# 1. YOLO 불러오기 (torch 문제가 있으면 여기서 실패)
try:
    from ultralytics import YOLO
except Exception as e:
    print("YOLO를 불러오지 못했습니다:", e)
    print("-> torch 설치 문제입니다. torch를 다시 설치한 뒤 실행하세요.")
    exit()

# 2. 사진 불러오기
IMG_PATH = "./image/3people.jpg"
img = cv2.imread(IMG_PATH)
if img is None:
    print("사진을 찾을 수 없습니다. 경로를 확인하세요:", IMG_PATH)
    exit()

# 3. YOLO 모델 불러오기 (처음 실행 때 yolov8n.pt 자동 다운로드)
model = YOLO("yolov8n.pt")

# 4. 사람(0번)만, 확신도 40% 이상만 찾기
results = model(img, classes=[0], conf=0.4, iou=0.5)

# 5. 결과를 우리가 쓰던 boxes 모양 (x1, y1, x2, y2)으로 꺼내기
boxes = []
for b in results[0].boxes:
    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
    conf = float(b.conf[0])
    boxes.append((x1, y1, x2, y2))
    print(f"ID {len(boxes)}: {(x1, y1, x2, y2)}  확신도 {conf:.2f}")

print("찾은 사람 수:", len(boxes))

# 6. 화면에 그리기
for i, (x1, y1, x2, y2) in enumerate(boxes):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f"ID {i + 1}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imshow("yolo test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
