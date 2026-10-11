# 음성 요청: 마이크로 말을 듣고 -> 글자로 바꾸고 -> 물건 이름을 찾아 명령 상자에 넣기
#
# 사용법: Python 창에서 V 키 -> RECORD_SEC 초 동안 말하기
#   "컵 가져다줘"    -> 물건 심부름
#   "따라와"         -> 따라다니기 시작
#   "그만", "기다려"  -> 모두 멈춤
#   "취소"           -> 심부름만 취소
#
# 필요한 설치 (cctv 환경에서 한 번만):
#   pip install faster-whisper sounddevice

import threading

import command_box

# ============================================================
# 설정값
# ============================================================
SAMPLE_RATE = 16000     # 녹음 품질 (음성 인식 모델이 이 값을 씀)
RECORD_SEC = 3          # V 를 누르면 몇 초 동안 들을지
MODEL_SIZE = "base"     # tiny / base / small : 클수록 정확하지만 느리고 무거움

# 들은 말 속에 이 단어가 있으면 그 물건으로 판단
# (물건 이름은 unity_detect.py 의 ITEM_NAMES 와 같아야 함)
KEYWORDS = {
    "remote": ["리모컨", "리모콘"],
    "phone": ["휴대폰", "핸드폰", "스마트폰", "전화기", "폰"],
    "bottle": ["물병", "약병", "병"],
    "cup": ["물컵", "컵"],
    "book": ["책"],
}
# 모두 멈춤 (심부름 취소 + 따라다니기 끄기)
#   "따라오지마", "그만 따라와" 처럼 '따라' 가 들어가도 멈춤이 먼저 되도록 맨 먼저 검사
STOP_WORDS = ["따라오지마", "따라오지말", "그만", "멈춰", "기다려", "거기있어", "스톱"]
# 따라다니기 시작
FOLLOW_WORDS = ["따라와", "따라오", "이리와", "같이가", "같이와"]
# 심부름만 취소 (따라다니기는 그대로)
CANCEL_WORDS = ["취소"]


def parse_command(text):
    """
    들은 말에서 명령 찾기. (띄어쓰기는 지우고 비교)
    반환: ("fetch", 물건이름) / ("cancel", None) / ("follow", None) / ("stop", None) / (None, None)
    """
    t = text.replace(" ", "")
    for w in STOP_WORDS:
        if w in t:
            return "stop", None
    for w in FOLLOW_WORDS:
        if w in t:
            return "follow", None
    for w in CANCEL_WORDS:
        if w in t:
            return "cancel", None
    for item, words in KEYWORDS.items():
        for w in words:
            if w in t:
                return "fetch", item
    return None, None


class VoiceInput:
    """V 키를 누르면 정해진 시간 동안 듣고, 알아들은 명령을 명령 상자에 넣는다."""

    def __init__(self):
        self.status = "voice: loading"
        self.model = None
        self.busy = False
        # 모델 불러오기는 몇 초 걸려서, 화면이 멈추지 않게 따로 일꾼(스레드)에게 맡김
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            self.status = "voice: ready (V)"
            print("[voice] 음성 인식 준비 완료. Python 창에서 V 를 누르고 말하세요")
        except Exception as e:
            self.status = "voice: error"
            print("[voice] 음성 인식을 불러오지 못했습니다:", e)

    def start_listening(self):
        """V 키를 눌렀을 때 부르기"""
        if self.model is None:
            print("[voice] 아직 준비 중이에요. 잠시 후 다시 눌러 주세요")
            return
        if self.busy:
            return
        self.busy = True
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        try:
            import sounddevice as sd

            self.status = "voice: listening..."
            print(f"[voice] 듣는 중... ({RECORD_SEC}초 동안 말하세요)")
            audio = sd.rec(int(RECORD_SEC * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32")
            sd.wait()

            self.status = "voice: thinking..."
            segments, _ = self.model.transcribe(audio.flatten(), language="ko",
                                                beam_size=1)
            text = "".join(s.text for s in segments).strip()
            print(f"[voice] 들은 말: '{text}'")

            kind, item = parse_command(text)
            if kind == "fetch":
                command_box.put_fetch(item, "voice")
            elif kind == "cancel":
                command_box.put_cancel("voice")
            elif kind == "follow":
                command_box.put_follow("voice")
            elif kind == "stop":
                command_box.put_stop("voice")
            else:
                print("[voice] 알아들은 명령이 없어요. V 를 누르고 다시 말해 주세요")
        except Exception as e:
            print("[voice] 오류:", e)
        finally:
            self.busy = False
            if self.model is not None:
                self.status = "voice: ready (V)"