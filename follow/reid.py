import cv2
import numpy as np

import config


def get_color_hist(img, poly):
    """사람 몸(윤곽선 안쪽)의 옷 색깔 분포를 구한다."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


class ReIdentifier:
    """놓친 사람을 위치·키·시간·옷 색깔로 다시 찾는다."""

    def __init__(self):
        self.id_map = {}          # {새 번호: 원래 번호}
        self.last_box = None      # 선택한 사람의 마지막 위치
        self.lost_count = 0       # 놓친 뒤 지난 화면 수
        self.ids_at_lost = set()  # 놓친 순간 화면에 있던 번호들
        self.target_hist = None   # 선택한 사람의 옷 색깔 기억
        self.hist_owner = None    # 그 색깔이 누구 것인지

    def show_id(self, tid):
        """화면에 보여줄 번호 (다시 잡은 사람은 원래 번호)"""
        return self.id_map.get(tid, tid)

    def is_searching(self):
        """아직 다시 찾는 중인지 (포기하지 않았는지)"""
        return self.lost_count <= config.REACQ_FRAMES

    def update(self, selected, target_box, boxes, polys, frame, new_frame):
        """
        매 화면마다 부른다.
        반환: 다시 잡았으면 그 사람의 새 추적번호, 아니면 None
        """
        # 선택한 사람이 바뀌면 기억한 색깔도 버림
        if selected != self.hist_owner:
            self.target_hist = None
            self.hist_owner = selected

        # 아무도 선택 안 함
        if selected is None:
            self.last_box = None
            self.lost_count = 0
            return None

        # 보이는 중: 위치와 옷 색깔을 기억
        if target_box is not None:
            self.last_box = target_box
            self.lost_count = 0
            if new_frame and selected in polys:
                hist = get_color_hist(frame, polys[selected])
                if self.target_hist is None:
                    self.target_hist = hist
                else:
                    self.target_hist = self.target_hist * 0.9 + hist * 0.1
            return None

        # 안 보이는 중: 다시 찾기
        if self.last_box is None or not new_frame:
            return None

        if self.lost_count == 0:
            self.ids_at_lost = set()
            for tid, box in boxes:
                self.ids_at_lost.add(tid)
        self.lost_count += 1

        if not self.is_searching():
            return None

        new_id, sim = self._find(boxes, polys, frame)
        if new_id is None:
            return None

        print("다시 잡음: ID", self.show_id(selected),
              "(내부 번호", selected, "->", new_id,
              ", 색 유사도", round(sim, 2), ")")
        self.id_map[new_id] = self.show_id(selected)
        self.hist_owner = new_id      # 같은 사람이니 색깔 기억 유지
        self.lost_count = 0
        return new_id

    def _find(self, boxes, polys, frame):
        """마지막 위치 근처에서 같은 사람으로 보이는 새 번호를 찾는다."""
        lx1, ly1, lx2, ly2 = self.last_box
        lcx = (lx1 + lx2) / 2
        lcy = (ly1 + ly2) / 2
        lw = lx2 - lx1
        lh = ly2 - ly1

        best_id = None
        best_sim = 0
        best_dist = lw * config.REACQ_DIST

        for tid, (x1, y1, x2, y2) in boxes:
            if tid in self.ids_at_lost:
                continue
            h = y2 - y1
            if h < lh * 0.7 or h > lh * 1.3:
                continue

            sim = 1.0
            if self.target_hist is not None and tid in polys:
                cand_hist = get_color_hist(frame, polys[tid])
                sim = cv2.compareHist(self.target_hist, cand_hist,
                                      cv2.HISTCMP_CORREL)
                if sim < config.COLOR_MIN:
                    continue

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid
                best_sim = sim

        return best_id, best_sim