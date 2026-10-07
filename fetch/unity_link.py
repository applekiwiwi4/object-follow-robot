# Unity와 주고받는 통신 도구 모음
#
# Unity -> Python: [정보 크기 4바이트] + [정보(JSON)] + [사진 크기 4바이트] + [사진(JPG)]
# Python -> Unity: [답 크기 4바이트] + [답(JSON)]

import json
import socket
import struct

import cv2
import numpy as np

HOST = "127.0.0.1"   # 같은 컴퓨터
PORT = 5005          # Unity와 약속한 번호


def open_server():
    """Unity의 연결을 기다릴 서버 만들기"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    return server


def recv_exact(conn, n):
    """정확히 n바이트를 다 받을 때까지 읽기. 연결이 끊기면 None"""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_block(conn):
    """[크기 4바이트] + [내용] 하나 받기"""
    head = recv_exact(conn, 4)
    if head is None:
        return None
    size = struct.unpack(">I", head)[0]
    return recv_exact(conn, size)


def recv_packet(conn):
    """
    Unity가 보낸 정보와 사진 한 묶음 받기.
    반환: (frame, info)  연결이 끊기면 (None, None)
      frame : OpenCV 이미지
      info  : {"width": 640, "height": 480,
               "items": [{"name": "cup", "box": [x1, y1, x2, y2]}, ...]}
    """
    info_bytes = recv_block(conn)
    if info_bytes is None:
        return None, None
    jpg = recv_block(conn)
    if jpg is None:
        return None, None
    # 정보가 망가져서 왔으면 이번 장은 "소지품 없음"으로 처리 (프로그램이 꺼지지 않게)
    try:
        info = json.loads(info_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("[경고] Unity 정보를 읽지 못해 이번 장은 건너뜀:", info_bytes[:30])
        info = {"items": []}

    # 사진이 망가져서 왔으면 검은 화면으로 대신
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        print("[경고] Unity 사진을 읽지 못해 검은 화면으로 대신함")
        frame = np.zeros((480, 640, 3), np.uint8)
    return frame, info


def send_reply(conn, message):
    """Unity에 답 보내기 (딕셔너리 -> JSON)"""
    data = json.dumps(message).encode("utf-8")
    conn.sendall(struct.pack(">I", len(data)) + data)