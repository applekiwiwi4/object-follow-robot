import queue

# 모든 요청이 모이는 우편함 (여러 스레드가 같이 써도 안전)
#
# 요청 종류 (type)
#   fetch  : 물건 가져오기   {"type": "fetch", "item": "cup", "source": "voice"}
#   cancel : 심부름 취소     (따라다니기는 그대로)
#   follow : 어르신 따라다니기 시작
#   stop   : 모두 멈춤       (심부름 취소 + 따라다니기 끄기)
_box = queue.Queue()


def put_fetch(item, source):
    """물건 가져오기 요청 넣기. source: 누가 넣었는지 (keyboard, voice, gesture)"""
    _box.put({"type": "fetch", "item": item, "source": source})


def put_cancel(source):
    """심부름 취소 요청 넣기"""
    _box.put({"type": "cancel", "source": source})


def put_follow(source):
    """따라다니기 시작 요청 넣기"""
    _box.put({"type": "follow", "source": source})


def put_stop(source):
    """모두 멈춤 요청 넣기 (심부름 취소 + 따라다니기 끄기)"""
    _box.put({"type": "stop", "source": source})


def get_all():
    """우편함에 쌓인 요청을 전부 꺼내기 (없으면 빈 리스트)"""
    out = []
    while True:
        try:
            out.append(_box.get_nowait())
        except queue.Empty:
            return out