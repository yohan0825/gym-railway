# ============================================
# 체육관 모니터링 서버 (최종본)
# - 라즈베리파이가 보낸 프레임을 AI 모델로 분석 (종목 + 인원수)
# - 이미지는 분석 즉시 폐기, 결과 숫자만 SQLite에 기록
# - 실시간 현황판 + 시간대별 기록 API 제공
# 실행: python server.py  (학교 서버에서)
# ============================================
import os
import time
import sqlite3
import threading

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="public", static_url_path="")

API_KEY = os.environ.get("API_KEY", "change-this-key")
ADMIN_KEY = os.environ.get("ADMIN_KEY", "5050")   # 수동 입력 전용 비밀번호 (API_KEY와 별개)
STALE_SEC = 120                 # 2분간 업데이트 없으면 오프라인 표시
DB_PATH = os.environ.get("DB_PATH", "records.db")
RECORD_INTERVAL = 60            # DB 기록 최소 간격(초) — 같은 장소는 1분에 1번만 기록
VALID_SPORTS = {"배드민턴", "농구", "배드민턴&농구"}

# ---------- 실시간 상태 (메모리) ----------
state = {
    "gym": {"name": "체육관", "count": None, "sport": None, "zones": {}, "updatedAt": None, "manual": False, "onlineOverride": True},
}
lock = threading.Lock()
_last_record = {}  # 장소별 마지막 DB 기록 시각

# ---------- SQLite (결과 숫자만 저장, 이미지는 저장 안 함) ----------
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,          -- unix time (초)
            location TEXT NOT NULL,
            count INTEGER NOT NULL,
            sport TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_records_ts ON records(location, ts)")
    return conn

def record_result(location, count, sport):
    """분석 결과를 DB에 기록 (RECORD_INTERVAL 간격으로 샘플링)"""
    now = time.time()
    if now - _last_record.get(location, 0) < RECORD_INTERVAL:
        return
    _last_record[location] = now
    conn = db()
    conn.execute(
        "INSERT INTO records (ts, location, count, sport) VALUES (?, ?, ?, ?)",
        (int(now), location, int(count), sport),
    )
    conn.commit()
    conn.close()

def manual_recorder_loop():
    """수동 모드일 때는 값이 안 바뀌어도 RECORD_INTERVAL마다 현재값을 계속 기록해서
    그래프가 항상 '기록 없음'으로 멈춰있지 않도록 함"""
    while True:
        time.sleep(RECORD_INTERVAL)
        with lock:
            snapshot = [(loc_id, loc["count"], loc["sport"])
                        for loc_id, loc in state.items()
                        if loc.get("manual") and loc["count"] is not None]
        for loc_id, count, sport in snapshot:
            record_result(loc_id, count, sport)

threading.Thread(target=manual_recorder_loop, daemon=True).start()

# ---------- AI 모델 ----------
# 친구 모델 파일(best.pt)을 서버 폴더에 두면 자동 로드. 없으면 더미 모드.
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")
model = None

def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"[모델] {MODEL_PATH} 없음 → 더미 모드로 작동 (파이프라인 테스트 가능)")
        return
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        print(f"[모델] {MODEL_PATH} 로드 완료. 클래스: {model.names}")
    except Exception as e:
        print(f"[모델] 로드 실패 → 더미 모드: {e}")

load_model()


def analyze_frame(img):
    """
    프레임 1장 분석 → (인원수, 종목, 구역별 인원) 반환.
    ★ 친구 모델의 클래스 구성에 맞게 이 함수만 수정하면 됨 ★
    """
    if model is None:
        return 0, None, {}   # 더미 모드

    results = model(img, verbose=False)[0]
    names = results.names                      # {class_id: 이름}

    # 사람 수: COCO 기반이면 class 0 = person
    person_boxes = [b for b in results.boxes if int(b.cls) == 0]
    count = len(person_boxes)

    # 종목 판별: 친구 모델이 탐지하는 클래스 이름에 맞게 수정
    detected = {names[int(b.cls)] for b in results.boxes}
    sport = None
    if "basketball" in detected:
        sport = "농구"
    elif "badminton_racket" in detected or "shuttlecock" in detected:
        sport = "배드민턴"

    zones = {}  # 구역별 카운트 필요 시 바운딩박스 좌표로 분리
    return count, sport, zones


# ---------- API ----------

@app.route("/api/frame", methods=["POST"])
def receive_frame():
    """라즈베리파이 → 서버: 프레임 업로드 → 분석 → 즉시 폐기"""
    if request.form.get("key") != API_KEY:
        return jsonify(ok=False, error="invalid key"), 401

    location = request.form.get("location", "gym")
    if location not in state:
        return jsonify(ok=False, error="unknown location"), 400

    if state[location].get("manual"):
        # 관리자가 수동 입력 모드로 전환해둔 상태 → AI 분석 결과로 덮어쓰지 않음
        return jsonify(ok=True, skipped="manual mode active")

    file = request.files.get("frame")
    if file is None:
        return jsonify(ok=False, error="no frame"), 400

    buf = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="bad image"), 400

    t0 = time.time()
    count, sport, zones = analyze_frame(img)
    elapsed = time.time() - t0
    # img는 여기서 스코프 종료와 함께 폐기됨 (디스크 저장 없음)

    with lock:
        state[location].update(count=count, sport=sport, zones=zones,
                               updatedAt=int(time.time() * 1000))
    record_result(location, count, sport)

    print(f"[frame] {location}: {count}명, 종목={sport}, 추론 {elapsed:.2f}s")
    return jsonify(ok=True, count=count, sport=sport, inference_sec=round(elapsed, 2))


@app.route("/api/manual", methods=["POST"])
def manual_input():
    """관리자 수동 입력: 인원수/종목을 직접 지정 (AI 분석 대신 표시)"""
    data = request.get_json(silent=True) or {}
    if data.get("key") != ADMIN_KEY:
        return jsonify(ok=False, error="invalid key"), 401

    location = data.get("location", "gym")
    if location not in state:
        return jsonify(ok=False, error="unknown location"), 400

    mode = data.get("mode", "manual")
    if mode == "auto":
        # 수동 모드 해제 → 다음 프레임부터 다시 AI 분석 결과를 반영
        with lock:
            state[location]["manual"] = False
        return jsonify(ok=True, manual=False)

    sport = data.get("sport")
    if sport not in VALID_SPORTS and sport is not None:
        return jsonify(ok=False, error="invalid sport"), 400

    try:
        count = int(data.get("count", 0))
    except (TypeError, ValueError):
        return jsonify(ok=False, error="invalid count"), 400

    online = bool(data.get("online", True))

    with lock:
        state[location].update(count=count, sport=sport, manual=True, onlineOverride=online,
                               updatedAt=int(time.time() * 1000))
    record_result(location, count, sport)

    return jsonify(ok=True, manual=True, count=count, sport=sport, online=online)


@app.route("/api/reset-history", methods=["POST"])
def reset_history():
    """장난/오입력 등으로 남은 기록 그래프 초기화 (해당 location의 records 전체 삭제)"""
    data = request.get_json(silent=True) or {}
    if data.get("key") != ADMIN_KEY:
        return jsonify(ok=False, error="invalid key"), 401

    location = data.get("location", "gym")
    if location not in state:
        return jsonify(ok=False, error="unknown location"), 400

    conn = db()
    conn.execute("DELETE FROM records WHERE location=?", (location,))
    conn.commit()
    conn.close()
    _last_record.pop(location, None)

    return jsonify(ok=True)


@app.route("/api/status")
def status():
    """실시간 현황 조회 (웹페이지 폴링용)"""
    now = int(time.time() * 1000)
    out = {}
    with lock:
        for loc_id, loc in state.items():
            if loc.get("manual"):
                online = loc.get("onlineOverride", True)
            else:
                stale = loc["updatedAt"] is None or now - loc["updatedAt"] > STALE_SEC * 1000
                online = not stale
            out[loc_id] = {**loc, "online": online}
    return jsonify(out)


@app.route("/api/history")
def history():
    """시간대별 기록 조회: /api/history?location=gym&hours=6"""
    location = request.args.get("location", "gym")
    hours = min(int(request.args.get("hours", 6)), 168)  # 최대 7일
    since = int(time.time()) - hours * 3600

    conn = db()
    rows = conn.execute(
        "SELECT ts, count, sport FROM records WHERE location=? AND ts>=? ORDER BY ts",
        (location, since),
    ).fetchall()
    conn.close()

    return jsonify([{"ts": ts, "count": c, "sport": s} for ts, c, s in rows])


@app.route("/")
def index():
    return send_from_directory("public", "index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    print(f"서버 시작: http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
