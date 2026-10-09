import queue

# 모든 요청이 모이는 우편함 (여러 스레드가 같이 써도 안전)
_box = queue.Queue()


def put_fetch(item, source):
    """물건 가져오기 요청 넣기. source: 누가 넣었는지 (keyboard, voice, gesture)"""
    _box.put({"type": "fetch", "item": item, "source": source})


def put_cancel(source):
    """취소 요청 넣기"""
    _box.put({"type": "cancel", "source": source})


def get_all():
    """지금까지 쌓인 요청을 전부 꺼내서 목록으로 돌려주기"""
    commands = []
    while True:
        try:
            commands.append(_box.get_nowait())
        except queue.Empty:
            return commands