# ============================================
# 라즈베리파이: 웹캠 프레임을 주기적으로 서버에 전송
# 실행: python3 pi_streamer.py
# 필요: pip install opencv-python requests
# ============================================
import time
import cv2
import requests

SERVER_URL = "https://여기에-railway-주소.up.railway.app/api/frame"
API_KEY = "change-this-key"   # 서버 환경변수 API_KEY와 동일하게
LOCATION = "gym"
INTERVAL = 10                 # 전송 간격 (초) — Railway 크레딧 아끼려면 10초 이상 권장
SEND_WIDTH = 640              # 전송 전 리사이즈 폭 (px) — 작을수록 전송/추론 빠름
JPEG_QUALITY = 70             # JPEG 압축 품질 (1~100)

cap = cv2.VideoCapture(0)     # 웹캠 (0번 장치)
if not cap.isOpened():
    raise RuntimeError("웹캠을 열 수 없음. USB 연결 확인.")

print(f"전송 시작: {INTERVAL}초마다 {SERVER_URL}")

while True:
    # 버퍼에 쌓인 오래된 프레임을 버리고 최신 프레임 확보
    for _ in range(3):
        cap.grab()
    ok, frame = cap.read()

    if not ok:
        print("프레임 캡처 실패, 재시도...")
        time.sleep(2)
        continue

    # 리사이즈 (가로 SEND_WIDTH 기준, 비율 유지)
    h, w = frame.shape[:2]
    if w > SEND_WIDTH:
        frame = cv2.resize(frame, (SEND_WIDTH, int(h * SEND_WIDTH / w)))

    # JPEG 인코딩
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        print("JPEG 인코딩 실패")
        time.sleep(INTERVAL)
        continue

    # 서버 전송
    try:
        res = requests.post(
            SERVER_URL,
            data={"key": API_KEY, "location": LOCATION},
            files={"frame": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
            timeout=30,
        )
        print(f"전송 {len(jpg)//1024}KB → {res.status_code} {res.text[:80]}")
    except Exception as e:
        print("전송 실패:", e)

    time.sleep(INTERVAL)
