# Unity 로봇 눈 화면 받아서 보기 (+ Unity가 알려 준 소지품 위치 표시)
#
# 실행 순서: 이 프로그램을 먼저 실행 -> 그다음 Unity에서 ▶
# 종료: 영상 창을 클릭하고 ESC

import cv2

from unity_link import PORT, open_server, recv_packet, send_reply


def main():
    server = open_server()

    while True:
        print(f"Unity 연결 기다리는 중... (포트 {PORT}) 이제 Unity에서 ▶ 를 누르세요")
        conn, addr = server.accept()
        print("Unity 연결됨:", addr)

        count = 0
        quit_all = False
        while True:
            frame, info = recv_packet(conn)
            if frame is None:
                print("Unity 연결 끊김 (▶ 를 멈춘 경우 정상)")
                break

            count += 1
            for item in info["items"]:
                x1, y1, x2, y2 = item["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
                cv2.putText(frame, item["name"], (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            cv2.putText(frame, f"frame {count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Unity Robot Eye", frame)

            send_reply(conn, {"frame": count})

            if cv2.waitKey(1) & 0xFF == 27:
                quit_all = True
                break

        conn.close()
        if quit_all:
            break

    server.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()