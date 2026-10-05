# 설정값 모음 (자주 바꾸는 숫자는 여기서만 고치기)

# 영상
VIDEO_PATH = "./video/test_reid.mp4"   # 웹캠을 쓰려면 0
MAX_WIDTH = 960                        # 이보다 큰 영상은 줄여서 처리

# YOLO
MODEL_PATH = "yolov8n-seg.pt"
CONF = 0.4          # 확신도 기준
NMS_IOU = 0.5       # 겹친 네모 정리 기준
MIN_H = 80          # 키가 이보다 작은 사람은 무시

# 로봇
FOLLOW_DIST = 10    # 사람과 유지할 거리 (픽셀)
ROBOT_SPEED = 2     # 한 화면마다 움직이는 최대 거리
CENTER_TOL = 60     # 가운데서 이만큼 벗어나면 회전
NEAR_RATIO = 0.8    # 키가 화면의 80% 이상이면 너무 가까움
FAR_RATIO = 0.4     # 키가 화면의 40% 이하이면 너무 멂

# 다시 잡기
REACQ_FRAMES = 90   # 놓친 뒤 몇 화면 동안 찾을지
REACQ_DIST = 1.5    # (사람 폭 x 이 값) 안에 나타나야 같은 사람
COLOR_MIN = 0.5     # 옷 색 유사도가 이보다 낮으면 다른 사람

# 화면
FILL_ALPHA = 0     # 색칠 진하기 (0 = 선만, 0.4 = 반투명)