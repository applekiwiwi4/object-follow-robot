# 심부름 관리자: "컵 가져와" 한 번에 전체 순서를 진행한다.
#
#   IDLE      : 요청 기다리는 중
#   GO_ITEM   : 소지품 찾아서 다가가기
#   PICK      : 집기 (Unity에 "pick:cup" 명령)
#   GO_ELDER  : 어르신 찾아서 다가가기
#   DROP      : 내려놓기 (Unity에 "drop" 명령)
#   DONE      : 전달 완료

from approach import Approacher

# ============================================================
# 설정값
# ============================================================
ELDER_NAME = "elder"        # Unity에서 어르신에게 붙인 이름표
ELDER_ARRIVE_SIZE = 0.9     # 어르신은 키가 크니까 화면 높이의 90%쯤 차면 도착
ELDER_ARRIVE_BOTTOM = 0.98
PICK_WAIT = 20              # 집기 명령 후 이만큼 기다려도 안 집히면 다시 다가가기
DROP_WAIT = 20

STOP = {"move": 0.0, "turn": 0.0}


class FetchTask:
    def __init__(self):
        self.state = "IDLE"
        self.item = None
        self.approacher = None
        self.wait = 0
        self.message = "1~5 키로 요청하세요"

    def request(self, item):
        """심부름 시작"""
        self.item = item
        self.state = "GO_ITEM"
        self.approacher = Approacher(item)
        self.wait = 0
        self.message = f"'{item}' 가지러 갑니다"
        print(">>", self.message)

    def cancel(self):
        self.state = "IDLE"
        self.approacher = None
        self.message = "취소됨. 1~5 키로 요청하세요"
        print(">>", self.message)

    def _go(self, new_state, message):
        self.state = new_state
        self.wait = 0
        self.message = message
        print(">>", message)

    def update(self, items, holding, frame_w, frame_h):
        """
        매 화면마다 부른다.
          items   : 보이는 것들 [{"name", "box"}, ...]
          holding : 로봇이 지금 들고 있는 것 이름 ("" 이면 빈손)
        반환: Unity로 보낼 명령 {"move", "turn", "state", "action"}
        """
        action = ""

        if self.state in ("IDLE", "DONE"):
            cmd = dict(STOP)
            sub = ""

        elif self.state == "GO_ITEM":
            if holding == self.item:          # 이미 들고 있으면 바로 어르신에게
                self.approacher = Approacher(ELDER_NAME, ELDER_ARRIVE_SIZE, ELDER_ARRIVE_BOTTOM)
                self._go("GO_ELDER", f"이미 '{self.item}'을(를) 들고 있어요. 어르신께 갑니다")
                cmd, sub = dict(STOP), ""
            else:
                cmd = self.approacher.update(items, frame_w, frame_h)
                sub = cmd["state"]
                if self.approacher.state == "ARRIVED":
                    self._go("PICK", f"'{self.item}' 앞 도착. 집습니다")

        elif self.state == "PICK":
            cmd, sub = dict(STOP), ""
            if holding == self.item:
                self.approacher = Approacher(ELDER_NAME, ELDER_ARRIVE_SIZE, ELDER_ARRIVE_BOTTOM)
                self._go("GO_ELDER", f"'{self.item}' 집음! 어르신께 갑니다")
            else:
                action = f"pick:{self.item}"
                self.wait += 1
                if self.wait > PICK_WAIT:     # 너무 멀어서 못 집었으면 다시 다가가기
                    self.approacher.restart()
                    self._go("GO_ITEM", f"'{self.item}'이(가) 손에 안 닿아요. 다시 다가갑니다")

        elif self.state == "GO_ELDER":
            cmd = self.approacher.update(items, frame_w, frame_h)
            sub = cmd["state"]
            if self.approacher.state == "ARRIVED":
                self._go("DROP", "어르신 앞 도착. 내려놓습니다")

        elif self.state == "DROP":
            cmd, sub = dict(STOP), ""
            if holding == "":
                self._go("DONE", f"'{self.item}' 전달 완료! 1~5 키로 다음 요청")
            else:
                action = "drop"
                self.wait += 1
                if self.wait > DROP_WAIT:
                    self._go("DONE", "내려놓기 실패. Unity Console을 확인하세요")

        else:
            cmd, sub = dict(STOP), ""

        state_text = self.state if not sub else f"{self.state} / {sub}"
        return {"move": cmd["move"], "turn": cmd["turn"],
                "state": state_text, "action": action}