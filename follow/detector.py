import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.basetrack import BaseTrack

import config


class Detector:
    """YOLO 세그멘테이션으로 사람을 찾고 추적 번호를 붙인다."""

    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)
        BaseTrack.reset_id()          # 추적 번호를 1번부터 시작

    def detect(self, frame):
        """
        화면 한 장에서 사람을 찾는다.
        반환값
          boxes : [(추적번호, (x1, y1, x2, y2)), ...]
          polys : {추적번호: 몸 윤곽선 점들}
        """
        results = self.model.track(frame, persist=True, classes=[0],
                                   conf=config.CONF, iou=config.NMS_IOU,
                                   verbose=False)
        boxes = []
        polys = {}
        r = results[0]
        for i, b in enumerate(r.boxes):
            if b.id is None:
                continue
            tid = int(b.id[0])
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if (y2 - y1) < config.MIN_H:
                continue
            boxes.append((tid, (x1, y1, x2, y2)))
            if r.masks is not None and len(r.masks.xy[i]) > 0:
                polys[tid] = r.masks.xy[i].astype(np.int32)

        return boxes, polys