# 운동장 현황판 — Railway 시범 배포용

이미지(프레임)를 받아서 인원수를 분석하고 웹 현황판에 표시하는 서버.
시범 버전이라 AI 모델 없이 **더미 모드**로 작동함 (파이프라인 테스트용).
나중에 학교 서버로 옮길 때 ultralytics 설치 + best.pt 추가하면 실제 추론 작동.

## 파일 구성
- `server.py` : 메인 서버 (프레임 수신 → 분석 → SQLite 기록 → 현황판)
- `public/index.html` : 웹 현황판 (실시간 인원 + 종목 + 6시간 그래프)
- `pi_streamer.py` : 라즈베리파이용 프레임 전송 코드
- `requirements.txt`, `Procfile` : Railway 배포 설정

## Railway 배포 순서

1. 이 폴더를 GitHub 리포지토리로 올린다
   (GitHub에서 New repository 만들고, 이 폴더 내용물을 업로드)

2. railway.app 접속 → New Project → **Deploy from GitHub repo** → 리포 선택

3. 배포되면 Variables 탭에서 환경변수 추가:
   - `API_KEY` = 아무 비밀 문자열 (예: gym-test-2026)

4. Settings → Networking → **Generate Domain** 클릭
   → `https://뭐뭐.up.railway.app` 주소 생성됨

5. 그 주소로 접속하면 현황판이 뜸 (처음엔 둘 다 "오프라인" — 정상)

## 파이프라인 테스트 (노트북에서)

1. `pi_streamer.py` 열어서 수정:
   - `SERVER_URL` = "https://생성된주소.up.railway.app/api/frame"
   - `API_KEY` = Railway에 설정한 값과 동일하게

2. 노트북에 웹캠 연결하고 실행:
   ```
   pip install opencv-python requests
   python pi_streamer.py
   ```

3. 현황판 새로고침 → 체육관 카드가 "작동 중"으로 바뀌고
   숫자 0이 표시되면 성공 (더미 모드라 0명으로 나옴)

## 참고
- 이미지는 분석 즉시 폐기됨. DB(records.db)에는 숫자만 기록.
- Railway 무료 플랜은 재시작 시 DB가 초기화될 수 있음 (시범용이라 무관)
- 실전 배포(학교 서버)에서는 requirements.txt에 `ultralytics` 추가하고
  친구 모델 파일 `best.pt`를 server.py 옆에 두면 자동으로 실제 추론 모드로 전환됨
