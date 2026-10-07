# 목표 소지품을 찾아서 앞까지 다가가기
#
# 상태
#   SEARCH   : 목표가 안 보임 -> 제자리에서 천천히 돌면서 찾기
#   APPROACH : 목표가 보임 -> 가운데로 맞추면서 다가가기
#   ARRIVED  : 충분히 가까움 -> 멈춤

# ============================================================
# 설정값
# ============================================================
SEARCH_TURN = 0.4      # 찾을 때 도는 속도 (0 ~ 1)
TURN_GAIN = 1.5        # 가운데에서 벗어난 만큼 얼마나 세게 돌지
MAX_MOVE = 0.6         # 다가갈 때 최대 속도 (0 ~ 1)
ARRIVE_SIZE = 0.35     # 물건 높이가 화면의 35% 이상이면 도착
ARRIVE_BOTTOM = 0.95   # 물건 아래쪽이 화면 맨 아래(95%)에 닿으면 도착
LOST_FRAMES = 10       # 다가가다가 이만큼 연속으로 안 보이면 다시 찾기


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Approacher:
    def __init__(self, target):
        self.target = target
        self.state = "SEARCH"
        self.lost = 0

    def restart(self, target=None):
        """처음부터 다시 찾기 (목표를 바꿀 수도 있음)"""
        if target is not None:
            self.target = target
        self.state = "SEARCH"
        self.lost = 0

    def find_target(self, items):
        """보이는 소지품 중 목표와 같은 이름에서 가장 큰 것"""
        best = None
        best_area = 0
        for it in items:
            if it["name"] != self.target:
                continue
            x1, y1, x2, y2 = it["box"]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best = it["box"]
        return best

    def update(self, items, frame_w, frame_h):
        """
        매 화면마다 부른다.
        반환: Unity로 보낼 명령 {"move": -1~1, "turn": -1~1, "state": 글자}
        """
        if self.state == "ARRIVED":
            return {"move": 0.0, "turn": 0.0, "state": "ARRIVED"}

        box = self.find_target(items)

        # 목표가 안 보일 때
        if box is None:
            self.lost += 1
            if self.state == "APPROACH" and self.lost <= LOST_FRAMES:
                # 잠깐 안 보인 것일 수 있으니 멈춰서 기다림
                return {"move": 0.0, "turn": 0.0, "state": "APPROACH (lost)"}
            self.state = "SEARCH"
            return {"move": 0.0, "turn": SEARCH_TURN, "state": "SEARCH"}

        # 목표가 보일 때
        self.lost = 0
        self.state = "APPROACH"
        x1, y1, x2, y2 = box

        # 가운데에서 얼마나 벗어났나: -1(왼쪽 끝) ~ 0(가운데) ~ 1(오른쪽 끝)
        offset = ((x1 + x2) / 2 - frame_w / 2) / (frame_w / 2)
        size = (y2 - y1) / frame_h       # 물건 높이가 화면에서 차지하는 비율
        bottom = y2 / frame_h            # 물건 아래쪽 위치

        # 도착 판단
        if size >= ARRIVE_SIZE or bottom >= ARRIVE_BOTTOM:
            self.state = "ARRIVED"
            return {"move": 0.0, "turn": 0.0, "state": "ARRIVED"}

        # 많이 벗어날수록 세게 돌기 (비례 제어)
        turn = clamp(offset * TURN_GAIN, -1.0, 1.0)

        # 방향이 많이 틀어져 있으면 천천히, 가까워질수록 천천히
        move = MAX_MOVE * (1 - min(abs(offset), 1.0))
        move *= clamp(1 - size / ARRIVE_SIZE, 0.3, 1.0)

        return {"move": round(move, 3), "turn": round(turn, 3), "state": "APPROACH"}
