# Object Follow Robot

YOLO로 영상 속 사람을 추적해서, 선택한 사람이 가려지거나 사라졌다 나타나도
위치와 옷 색깔로 다시 찾아내 로봇이 계속 따라가게 하는 프로젝트입니다.

## 주요 기능

- **사람 감지 + 추적**: YOLOv8 세그멘테이션 모델로 사람을 찾고, 사람마다 추적 번호를 붙임
- **따라갈 사람 선택**: 클릭하거나, 대충 드래그하면 가장 많이 겹치는(IoU) 사람을 선택
- **가상 로봇 따라가기**: 선택한 사람의 발 위치를 일정 거리 두고 따라가며 전진/후진/회전/정지 명령 판단
- **다시 잡기**: 가려지거나 놓쳐도 위치·키·시간·옷 색깔로 같은 사람을 다시 찾음
- **원래 번호 유지**: 추적기가 새 번호를 붙여도 화면에는 처음 번호 그대로 표시

## 동작 흐름

```
화면 읽기 → 사람 감지·추적 (Detector)
         → 선택한 사람 찾기 (Selector)
         → 놓쳤으면 다시 잡기 (ReIdentifier)
         → 로봇 이동 + 명령 판단 (Robot)
         → 화면 그리기 (drawing)
```

## 폴더 구조

```
object-follow-robot/
├─ main.py              실행 파일
├─ config.py            설정값 모음
├─ requirements.txt     필요한 패키지 목록
├─ follow/              핵심 부품
│   ├─ detector.py      YOLO 감지·추적
│   ├─ selector.py      클릭·드래그 선택
│   ├─ reid.py          놓친 사람 다시 잡기 (옷 색깔 비교 포함)
│   ├─ robot.py         가상 로봇 이동, 명령 판단
│   └─ drawing.py       화면 그리기
├─ tools/
│   └─ make_test_video.py   테스트 영상 생성기
├─ learning/            단계별 연습 파일 (참고용)
├─ image/               테스트 사진
└─ video/               테스트 영상 (Git에 올리지 않음)
```

## 설치

Anaconda를 기준으로 설명합니다.

**1. 환경 만들기**

```
conda create -n follow python=3.10
conda activate follow
```

**2. PyTorch 설치 (CPU 버전)**

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

> conda로 설치한 PyTorch는 Windows에서 `DLL load failed` 에러가 날 수 있어서, pip로 공식 CPU 버전을 설치합니다.

**3. 나머지 패키지 설치**

```
pip install -r requirements.txt
```

YOLO 모델 파일(`yolov8n-seg.pt`)은 처음 실행할 때 자동으로 다운로드됩니다.

## 실행

**1. 테스트 영상 만들기** (처음 한 번)

```
python tools/make_test_video.py
```

`image/3people.jpg`의 사람들을 오려 내서 `video/test_reid.mp4`를 만듭니다.

**2. 프로그램 실행**

```
python main.py
```

> 모든 명령은 **프로젝트 맨 위 폴더**에서 실행하세요. 코드 안의 `./image`, `./video` 경로가 실행한 위치를 기준으로 합니다.

## 조작법

| 키 / 마우스 | 동작 |
|---|---|
| 클릭 | 그 사람 선택 |
| 드래그 | 드래그 영역과 가장 많이 겹치는 사람 선택 |
| SPACE | 일시정지 / 재생 |
| C | 선택 해제 |
| ESC | 종료 |

> 키가 안 먹히면 영상 창을 한 번 클릭하고, 키보드가 영문 상태인지 확인하세요.

## 주요 설정값 (`config.py`)

| 이름 | 뜻 |
|---|---|
| `VIDEO_PATH` | 영상 경로 (웹캠은 `0`) |
| `CONF` | YOLO 확신도 기준. 높이면 사람이 덜 잡힘 |
| `MIN_H` | 이보다 작게 보이는(멀리 있는) 사람은 무시 |
| `FOLLOW_DIST`, `ROBOT_SPEED` | 로봇이 유지할 거리, 움직이는 속도 |
| `REACQ_FRAMES` | 놓친 뒤 다시 찾아볼 화면 수 |
| `REACQ_DIST` | 마지막 위치에서 이 범위 안에 나타나야 같은 사람으로 봄 |
| `COLOR_MIN` | 옷 색 유사도 기준. 낮추면 더 잘 잡지만 엉뚱한 사람도 잡을 수 있음 |
| `FILL_ALPHA` | 몸 색칠 진하기 (`0`이면 윤곽선만) |

## 팀 작업 규칙

1. 작업 시작 전 `git switch main` → `git pull`로 최신 상태 맞추기
2. `git switch -c 기능이름`으로 브랜치를 만들어 작업
3. 커밋 후 `git push -u origin 기능이름`
4. GitHub에서 Pull Request를 만들고, **Files changed**에서 바뀐 파일을 확인한 뒤 Merge
5. Merge한 브랜치는 삭제

## 알려진 한계

- CPU에서 세그멘테이션까지 돌리면 처리 속도가 느림
- 옷 색이 비슷한 사람이 같은 자리에 나타나면 다시 잡기가 헷갈릴 수 있음
- 로봇 명령은 아직 화면 속 시뮬레이션이며, 실제 하드웨어와는 연결되지 않음
- 테스트 영상은 합성 영상이라 실제 걷는 모습과 다름

## 문제 해결

| 증상 | 해결 |
|---|---|
| `DLL load failed while importing _C` | conda로 설치한 torch를 지우고(`conda remove --force pytorch torchvision`) 위 설치 2단계대로 pip로 다시 설치 |
| `영상을 열 수 없습니다` | `config.py`의 `VIDEO_PATH`와 실행 위치(프로젝트 맨 위 폴더) 확인 |
| `ModuleNotFoundError: follow` | `follow` 폴더 안에 `__init__.py`가 있는지 확인 |
| 추적 번호가 1번이 아닌 큰 숫자부터 시작 | Jupyter 대신 터미널이나 VS Code ▶ 버튼으로 실행 |
| 오른쪽 클릭하면 메뉴가 뜸 | OpenCV 창 기본 기능이라 정상. 선택은 왼쪽 버튼만 사용 |
