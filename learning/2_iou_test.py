def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    # 1. 겹친 부분의 네모 좌표
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    # 2. 겹친 넓이 (안 겹치면 0)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    # 3. 각 네모 넓이
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    # 4. 합친 전체 넓이
    union = area_a + area_b - inter
    if union == 0:
        return 0

    # 5. IoU
    return inter / union


print("예제:", iou((0, 0, 10, 10), (5, 0, 15, 10)))
print("똑같은 네모:", iou((0, 0, 10, 10), (0, 0, 10, 10)))
print("안 겹침:", iou((0, 0, 10, 10), (20, 20, 30, 30)))
print("안에 쏙 들어감:", iou((0, 0, 10, 10), (0, 0, 5, 5)))