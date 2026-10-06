import config


def decide_command(box, frame_w, frame_h):
    """사람 위치와 크기를 보고 로봇 명령을 정한다."""
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    offset = center_x - frame_w / 2      # 음수면 왼쪽, 양수면 오른쪽
    h_ratio = (y2 - y1) / frame_h        # 사람 키가 화면에서 차지하는 비율

    if offset < -config.CENTER_TOL:
        return "TURN LEFT"
    if offset > config.CENTER_TOL:
        return "TURN RIGHT"
    if h_ratio > config.NEAR_RATIO:
        return "BACKWARD"
    if h_ratio < config.FAR_RATIO:
        return "FORWARD"
    return "STOP (good distance)"


class Robot:
    """선택한 사람의 발 위치를 일정 거리 두고 따라가는 가상 로봇"""

    def __init__(self):
        self.pos = None           # 로봇 위치 [x, y] (아직 없으면 None)
        self.target = None        # 지금 향하는 지점 (그리기용)
        self.following_id = None  # 누구를 따라가는 중인지

    def keep_following(self, new_id):
        """다시 잡기로 번호만 바뀐 경우: 위치는 그대로, 따라갈 번호만 바꿈"""
        self.following_id = new_id

    def update(self, selected, label, target_box, frame_w, frame_h,
               new_frame, searching):
        """
        매 화면마다 부른다. 로봇을 움직이고 명령 글자를 돌려준다.
          selected   : 선택한 사람의 추적 번호 (없으면 None)
          label      : 화면에 보여줄 번호 (다시 잡은 사람은 원래 번호)
          target_box : 이번 화면에서 그 사람의 네모 (안 보이면 None)
          new_frame  : 새 화면을 읽었는지 (일시정지면 False)
          searching  : 놓쳤을 때 아직 다시 찾는 중인지
        """
        # 다른 사람을 선택하면 처음 위치에서 다시 출발
        if selected != self.following_id:
            self.pos = None
            self.following_id = selected

        self.target = None

        if selected is None:
            return "WAITING (select a person)"

        if target_box is None:
            if searching:
                return f"STOP (ID {label} lost, searching)"
            return f"STOP (ID {label} lost, gave up)"

        x1, y1, x2, y2 = target_box
        self.target = ((x1 + x2) / 2, y2)          # 발 위치를 따라감
        if self.pos is None:
            self.pos = [frame_w / 2, frame_h - 40.0]   # 화면 아래 가운데에서 출발
        if new_frame:
            self._move()
        return decide_command(target_box, frame_w, frame_h)

    def _move(self):
        dx = self.target[0] - self.pos[0]
        dy = self.target[1] - self.pos[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist > config.FOLLOW_DIST:
            step = min(config.ROBOT_SPEED, dist - config.FOLLOW_DIST)
            self.pos[0] += dx / dist * step
            self.pos[1] += dy / dist * step