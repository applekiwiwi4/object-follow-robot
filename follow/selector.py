import cv2


def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union == 0:
        return 0
    return inter / union


class Selector:
    """마우스 클릭·드래그로 따라갈 사람을 고른다."""

    def __init__(self, id_map):
        self.id_map = id_map      # 다시 잡은 사람을 원래 번호로 보여주기용
        self.boxes = []           # 지금 화면의 사람 목록 (main이 매번 넣어 줌)
        self.selected = None      # 선택한 사람의 추적 번호
        self.scores = {}          # 드래그 선택 때 사람별 IoU 점수
        self.dragging = False     # 드래그 중인지
        self.drag_start = None    # 드래그 시작 위치
        self.drag_cur = None      # 드래그 현재 위치

    def show_id(self, tid):
        return self.id_map.get(tid, tid)

    def clear(self):
        self.selected = None
        self.scores = {}

    def select_by_point(self, x, y):
        self.scores = {}
        self.selected = None
        for tid, box in self.boxes:
            if point_in_box(x, y, box):
                self.selected = tid
                break
        if self.selected is None:
            print("선택 해제")
        else:
            print("선택: ID", self.show_id(self.selected))

    def select_by_drag(self, drag):
        self.scores = {}
        best_id = None
        best_score = 0
        for tid, box in self.boxes:
            score = iou(drag, box)
            self.scores[tid] = score
            if score > best_score:
                best_score = score
                best_id = tid
        if best_score < 0.1:
            self.selected = None
            print("선택 해제 (많이 겹치는 사람 없음)")
        else:
            self.selected = best_id
            print("선택: ID", self.show_id(best_id), "점수:", round(best_score, 2))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_cur = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_cur = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x0, y0 = self.drag_start
            drag = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            if drag[2] - drag[0] > 3 and drag[3] - drag[1] > 3:
                self.select_by_drag(drag)
            else:
                self.select_by_point(x, y)