# 목표(소지품 또는 어르신)를 찾아서 앞까지 다가가기
#
# 상태
#   SEARCH   : 목표가 안 보임 -> 제자리에서 천천히 돌면서 찾기
#   APPROACH : 목표가 보임 -> 가운데로 맞추면서 다가가기
#   ARRIVED  : 충분히 가까움 -> 멈춤

# ============================================================
# 기본 설정값
# ============================================================
SEARCH_TURN = 0.4      # 찾을 때 도는 속도 (0 ~ 1)
TURN_GAIN = 1.5        # 가운데에서 벗어난 만큼 얼마나 세게 돌지
MAX_MOVE = 0.6         # 다가갈 때 최대 속도 (0 ~ 1)
ARRIVE_SIZE = 0.35     # 목표 높이가 화면의 35% 이상이면 가까움
ARRIVE_BOTTOM = 0.95   # 목표 아래쪽이 화면 맨 아래(95%)에 닿으면 가까움
ARRIVE_CENTER = 0.25   # 가운데에서 이만큼 안쪽에 있을 때만 도착 인정 (-1 ~ 1 기준)
ARRIVE_CONFIRM = 3     # 도착 조건이 이만큼 연속으로 맞아야 진짜 도착
LOST_FRAMES = 10       # 다가가다가 이만큼 연속으로 안 보이면 다시 찾기


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Approacher:
    def __init__(self, target, arrive_size=ARRIVE_SIZE, arrive_bottom=ARRIVE_BOTTOM):
        self.target = target
        self.arrive_size = arrive_size        # 목표마다 다르게 정할 수 있음 (사람은 크니까 더 크게)
        self.arrive_bottom = arrive_bottom
        self.state = "SEARCH"
        self.lost = 0
        self.arrive_count = 0
        # 판단에 쓴 숫자 (화면 표시용)
        self.offset = 0.0
        self.size = 0.0
        self.bottom = 0.0

    def restart(self, target=None):
        """처음부터 다시 찾기 (목표를 바꿀 수도 있음)"""
        if target is not None:
            self.target = target
        self.state = "SEARCH"
        self.lost = 0
        self.arrive_count = 0

    def find_target(self, items):
        """보이는 것 중 목표와 같은 이름에서 가장 큰 것"""
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
        반환: {"move": -1~1, "turn": -1~1, "state": 글자}
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
        size = (y2 - y1) / frame_h       # 목표 높이가 화면에서 차지하는 비율
        bottom = y2 / frame_h            # 목표 아래쪽 위치
        self.offset, self.size, self.bottom = offset, size, bottom

        # 도착 판단: 가까움 + 가운데에 있음 + 몇 장 연속
        close = size >= self.arrive_size or bottom >= self.arrive_bottom
        centered = abs(offset) <= ARRIVE_CENTER
        if close and centered:
            self.arrive_count += 1
            if self.arrive_count >= ARRIVE_CONFIRM:
                self.state = "ARRIVED"
                return {"move": 0.0, "turn": 0.0, "state": "ARRIVED"}
            return {"move": 0.0, "turn": 0.0, "state": "APPROACH (check)"}
        self.arrive_count = 0

        # 가깝지만 옆에 있으면: 앞으로 가지 말고 돌기만
        if close:
            turn = clamp(offset * TURN_GAIN, -1.0, 1.0)
            return {"move": 0.0, "turn": round(turn, 3), "state": "APPROACH (turn)"}

        # 많이 벗어날수록 세게 돌기 (비례 제어)
        turn = clamp(offset * TURN_GAIN, -1.0, 1.0)

        # 방향이 많이 틀어져 있으면 천천히, 가까워질수록 천천히
        move = MAX_MOVE * (1 - min(abs(offset), 1.0))
        move *= clamp(1 - size / self.arrive_size, 0.3, 1.0)

        return {"move": round(move, 3), "turn": round(turn, 3), "state": "APPROACH"}