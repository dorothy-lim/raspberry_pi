# -*- coding: utf-8 -*-
"""USB(UVC) 카메라 MJPEG 스트리밍 서버.

카메라가 직접 출력하는 MJPEG 프레임을 재인코딩 없이 그대로 전달한다 (Pi CPU 거의 사용 안 함).
ROTATE=180 등 회전이 필요하면 그때만 디코드→회전→JPEG 재인코딩한다.
엔드포인트/헤더는 picamera2_stream.py와 같으므로 client/ 코드를 그대로 쓸 수 있다.

  /video_feed    multipart MJPEG (X-Frame-Id, X-Timestamp 헤더)
  /snapshot.jpg  최신 프레임 1장
  /status        JSON 상태

환경변수
  USB_DEVICE=auto (첫 번째 UVC 장치) 또는 /dev/video1
  USB_WIDTH=640  USB_HEIGHT=480  USB_FPS=30
  ROTATE=0|90|180|270   JPEG_QUALITY=80 (회전 시)   ROTATE_WORKERS=3   PORT=8001
"""
import glob
import itertools
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify

DEVICE = os.getenv("USB_DEVICE", "auto")
WIDTH = int(os.getenv("USB_WIDTH", "640"))
HEIGHT = int(os.getenv("USB_HEIGHT", "480"))
FPS = int(os.getenv("USB_FPS", "30"))
ROTATE = int(os.getenv("ROTATE", "0"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
PORT = int(os.getenv("PORT", "8001"))
ROTATE_WORKERS = int(os.getenv("ROTATE_WORKERS", "3"))

ROTATE_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

stop_event = threading.Event()
app = Flask(__name__)


def find_uvc_device():
    """uvcvideo 드라이버를 쓰는 첫 번째 캡처 장치."""
    for dev in sorted(glob.glob("/dev/video*"), key=lambda d: int(d[10:] or 0)):
        name = os.path.basename(dev)
        try:
            driver = os.path.basename(os.readlink(f"/sys/class/video4linux/{name}/device/driver"))
            with open(f"/sys/class/video4linux/{name}/index") as f:
                index = int(f.read())
        except Exception:
            continue
        if driver == "uvcvideo" and index == 0:     # index 0 = 영상 노드 (1은 메타데이터)
            return dev
    return None


def read_cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except Exception:
        return -1.0


# -------------------------------------------------
# 최신 프레임 보관
# -------------------------------------------------
class FrameBuffer:
    def __init__(self):
        self.frame = None
        self.frame_id = 0
        self.frame_time = time.time()
        self.condition = threading.Condition()
        self.recent = deque(maxlen=60)

    def put(self, jpeg: bytes):
        now = time.time()
        with self.condition:
            self.frame = jpeg
            self.frame_id += 1
            self.frame_time = now
            self.recent.append((now, len(jpeg)))
            self.condition.notify_all()

    def stats(self):
        with self.condition:
            r = list(self.recent)
            fid, ft = self.frame_id, self.frame_time
        fps = (len(r) - 1) / (r[-1][0] - r[0][0]) if len(r) > 1 and r[-1][0] > r[0][0] else 0.0
        avg = sum(s for _, s in r) // len(r) if r else 0
        return {"frame_id": fid, "last_frame_time": ft, "fps": round(fps, 2), "avg_frame_bytes": avg}


buf = FrameBuffer()
device_path = None


def open_camera(dev):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)      # 디코드하지 않고 원본 MJPEG 바이트를 받음
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # 어두울 때 카메라가 스스로 fps를 낮추지 않도록 (지원하는 카메라만)
    subprocess.run(["v4l2-ctl", "-d", dev, "-c", "exposure_dynamic_framerate=0"],
                   capture_output=True)
    return cap


def capture_thread():
    global device_path
    while not stop_event.is_set():
        dev = DEVICE if DEVICE != "auto" else find_uvc_device()
        if not dev:
            print("[USB] no UVC camera found, retrying ...", flush=True)
            stop_event.wait(3)
            continue
        cap = open_camera(dev)
        if not cap.isOpened():
            print(f"[USB] cannot open {dev}, retrying ...", flush=True)
            stop_event.wait(3)
            continue
        device_path = dev
        print(f"[USB] {dev} {int(cap.get(3))}x{int(cap.get(4))} @ {cap.get(5):.0f}fps, "
              f"rotate={ROTATE}", flush=True)
        fails = 0
        while not stop_event.is_set():
            ok, raw = cap.read()
            if not ok or raw is None:
                fails += 1
                if fails > 30:                 # 카메라 분리 등 → 다시 열기
                    print("[USB] read failed, reopening ...", flush=True)
                    break
                continue
            fails = 0
            jpeg = raw.tobytes()
            if not jpeg.startswith(b"\xff\xd8"):
                continue
            if ROTATE in ROTATE_CODES:
                seq = next(seq_counter)
                try:
                    rotate_queue.put_nowait((seq, raw.reshape(-1).copy()))
                except queue.Full:
                    pass                       # 워커가 바쁘면 이 프레임은 버림 (지연 누적 방지)
            else:
                buf.put(jpeg)
        cap.release()
        device_path = None


# -------------------------------------------------
# 회전 워커: 디코드→회전→인코드를 여러 스레드로 병렬 처리
# (cv2.imdecode/imencode는 GIL을 풀어서 여러 코어를 사용)
# -------------------------------------------------
rotate_queue = queue.Queue(maxsize=ROTATE_WORKERS)
seq_counter = itertools.count(1)
publish_lock = threading.Lock()
last_published_seq = 0


def rotate_worker():
    global last_published_seq
    while not stop_event.is_set():
        try:
            seq, data = rotate_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.rotate(img, ROTATE_CODES[ROTATE])
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
        with publish_lock:
            if seq <= last_published_seq:      # 더 최신 프레임이 이미 나갔으면 버림 (순서 보장)
                continue
            last_published_seq = seq
            buf.put(enc.tobytes())


# -------------------------------------------------
# Flask
# -------------------------------------------------
def generate_mjpeg():
    last_id = 0
    while not stop_event.is_set():
        with buf.condition:
            buf.condition.wait_for(lambda: buf.frame_id != last_id or stop_event.is_set(), timeout=1.0)
            if buf.frame_id == last_id or buf.frame is None:
                continue
            frame, last_id, ts = buf.frame, buf.frame_id, buf.frame_time
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
               b"X-Frame-Id: " + str(last_id).encode() + b"\r\n"
               b"X-Timestamp: " + f"{ts:.6f}".encode() + b"\r\n\r\n" +
               frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store"}, direct_passthrough=True)


@app.route("/snapshot.jpg")
def snapshot():
    with buf.condition:
        frame, fid, ts = buf.frame, buf.frame_id, buf.frame_time
    if frame is None:
        return "No frame yet", 503
    return Response(frame, mimetype="image/jpeg", headers={
        "Cache-Control": "no-cache, no-store", "X-Frame-Id": str(fid), "X-Timestamp": f"{ts:.6f}"})


@app.route("/status")
def status():
    st = buf.stats()
    st.update({"device": device_path, "width": WIDTH, "height": HEIGHT, "target_fps": FPS,
               "rotate": ROTATE, "encoder": "passthrough" if ROTATE not in ROTATE_CODES else "opencv",
               "cpu_temp_c": read_cpu_temp_c(), "server_time": time.time()})
    return jsonify(st)


@app.route("/")
def index():
    return """
    <html><body>
      <h1>USB Camera Live Stream</h1>
      <img src="/video_feed">
      <p id="st"></p>
      <script>
      setInterval(() => fetch("/status").then(r => r.json()).then(s => {
        document.getElementById("st").textContent =
          `${s.device} ${s.width}x${s.height} ${s.fps} fps, ${(s.avg_frame_bytes/1000).toFixed(1)} KB/frame, ` +
          `${s.encoder}, ${s.cpu_temp_c.toFixed(1)}°C`;
      }), 1000);
      </script>
    </body></html>
    """


def signal_handler(sig, frame):
    stop_event.set()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    cv2.setNumThreads(1)                       # 병렬화는 워커 스레드로 함
    if ROTATE in ROTATE_CODES:
        for _ in range(ROTATE_WORKERS):
            threading.Thread(target=rotate_worker, daemon=True).start()
    threading.Thread(target=capture_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
