# Pi 카메라 물체 인식 / 추적 클라이언트

Raspberry Pi(`picamera2_stream.py`)의 MJPEG 스트림을 다른 PC에서 받아 YOLO로 인식하고 ByteTrack으로 추적합니다.

## 설치

```bash
python -m venv venv
venv\Scripts\activate            # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

NVIDIA GPU가 있으면 먼저 CUDA용 PyTorch를 설치하세요 (https://pytorch.org). 모델(`yolo11n.pt`)은 첫 실행 때 자동으로 받습니다.

## 실행

```bash
python object_tracker.py                                   # 전체 클래스, 화면 표시
python object_tracker.py --classes person --conf 0.5       # 사람만
python object_tracker.py --model yolo11s.pt --device cuda:0
python object_tracker.py --no-show --log tracks.csv --save out.mp4
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--url` | `http://192.168.0.82:8000/video_feed` | 스트림 주소 |
| `--model` | `yolo11n.pt` | YOLO 모델 (n < s < m: 정확도↑ 속도↓) |
| `--tracker` | `bytetrack.yaml` | `botsort.yaml`도 가능 |
| `--classes` | 전체 | 이름 또는 번호 (COCO: person=0, car=2 …) |
| `--conf` | 0.4 | 검출 신뢰도 임계값 |
| `--log` | – | CSV: Pi 시각, 프레임 번호, track id, 박스, 중심, 속도(px/s) |
| `--save` | – | 결과 영상(mp4) 저장 |

화면에서 `q` 또는 `ESC`로 종료합니다.

## 구조

- `mjpeg_reader.py` — 저지연 스트림 리더. 백그라운드 스레드가 항상 **최신 프레임만** 보관하므로,
  추론이 30fps보다 느려도 지연이 쌓이지 않습니다 (중간 프레임은 건너뜀, 화면의 `skipped` 수치).
  각 프레임에 서버가 보낸 `frame_id`, `timestamp`(Pi 기준 시각)가 들어 있습니다.
- `object_tracker.py` — `model.track(..., persist=True)`로 프레임마다 추적,
  궤적/속도 화살표 표시. 속도는 Pi 타임스탬프 기준이라 PC 처리 지연과 무관합니다.

다른 코드에서 리더만 쓰려면:

```python
from mjpeg_reader import MJPEGStreamReader
reader = MJPEGStreamReader("http://192.168.0.82:8000/video_feed").start()
last = None
while True:
    f = reader.read(last_id=last)      # 새 프레임까지 대기
    if f is None:
        continue
    last = f.frame_id
    # f.image (BGR numpy), f.frame_id, f.timestamp
```

## 서버 엔드포인트

- `/video_feed` — MJPEG 스트림 (640x480 @ 30fps, 프레임 헤더 `X-Frame-Id`, `X-Timestamp`)
- `/snapshot.jpg` — 최신 프레임 1장
- `/status` — JSON (실제 fps, 프레임 크기, 인코더, Pi 온도)

## 참고 성능 (2026-10-06 측정)

- 서버: 640x480 30.1fps, 프레임 약 37KB(약 9Mbit/s), 하드웨어 인코더, Pi CPU 약 13%, 클라이언트 2대 동시 접속에서도 중복·누락 0
- 클라이언트(이 테스트 PC, CPU 전용, yolo11n): 약 18fps 처리, 프레임당 추론+그리기 50~80ms

## USB 카메라 (port 8001)

Pi에 연결된 USB 웹캠(Philips SPC 1300NC)은 `usb_camera_stream.py`가 **8001번 포트**로 스트리밍합니다.
엔드포인트/헤더가 같으므로 클라이언트는 URL만 바꾸면 됩니다.

```bash
python object_tracker.py --url http://192.168.0.82:8001/video_feed
python mjpeg_reader.py http://192.168.0.82:8001/video_feed
```

- `ROTATE=180`이면 회전 후 재인코딩 (카메라가 거꾸로 설치된 경우) — 29.3fps, 약 50KB/프레임, Pi CPU 약 40%
- `ROTATE=0`(기본)이면 카메라 MJPEG를 그대로 전달 — 29.3fps, 약 106KB/프레임(약 25Mbit/s), Pi CPU 약 15%
