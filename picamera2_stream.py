# -*- coding: utf-8 -*-
"""Pi Camera MJPEG 스트리밍 서버 (물체 인식/추적 클라이언트용).

- 고정 해상도/FPS (기본 640x480 @ 30fps) — 자동 튜닝 없음, 카메라 재시작 없음
- 하드웨어 MJPEG 인코더(bcm2835-codec) 우선, 실패 시 소프트웨어 JpegEncoder로 대체
- 각 프레임에 X-Frame-Id / X-Timestamp 헤더 포함, 같은 프레임 중복 전송 안 함

엔드포인트
  /video_feed    multipart MJPEG 스트림
  /snapshot.jpg  최신 프레임 1장
  /status        JSON 상태 (해상도, 실제 fps, 인코더, 온도 등)

환경변수 (systemd: Environment=STREAM_FPS=20 등)
  STREAM_WIDTH=640  STREAM_HEIGHT=480  STREAM_FPS=30
  ENCODER=hw|sw     MJPEG_BITRATE=10000000 (hw)   JPEG_QUALITY=80 (sw)
  USE_HTTPS=0|1     PORT=8000
"""
import threading
import time
import signal
import sys
import os
import ssl
import socket
import subprocess
from collections import deque
from flask import Flask, Response, request, jsonify

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder, MJPEGEncoder
from picamera2.outputs import Output

# -------------------------------------------------
# 설정
# -------------------------------------------------
WIDTH = int(os.getenv("STREAM_WIDTH", "640"))
HEIGHT = int(os.getenv("STREAM_HEIGHT", "480"))
FPS = int(os.getenv("STREAM_FPS", "30"))
ENCODER = os.getenv("ENCODER", "hw").lower()
MJPEG_BITRATE = int(os.getenv("MJPEG_BITRATE", "10000000"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
PORT = int(os.getenv("PORT", "8000"))

# 기본은 HTTP. USE_HTTPS=1 환경변수로 HTTPS(자체 서명 인증서) 실행 가능
USE_HTTPS = os.getenv("USE_HTTPS", "0") == "1"
SSL_CERT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cert.pem")
SSL_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.pem")

# 프레임이 이 시간 이상 안 들어오면 watchdog 알림을 멈춰 systemd가 재시작하게 함
FRAME_STALL_SEC = 5.0

# -------------------------------------------------
# 종료 제어
# -------------------------------------------------
stop_event = threading.Event()

# -------------------------------------------------
# systemd notify (선택)
# -------------------------------------------------
try:
    from systemd.daemon import notify
    SYSTEMD_OK = True
except Exception:
    SYSTEMD_OK = False

app = Flask(__name__)

# -------------------------------------------------
# 유틸
# -------------------------------------------------
def sd_notify(msg: str):
    if SYSTEMD_OK:
        try:
            notify(msg)
        except Exception:
            pass

def read_cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except Exception:
        return -1.0

def get_local_ips():
    """이 장치의 IPv4 주소 목록 (hostname -I)."""
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5).stdout
        return [ip for ip in out.split() if ip.count(".") == 3]
    except Exception:
        return []

def build_san_entries():
    """인증서 SAN 항목: localhost, 호스트명, 모든 로컬 IP."""
    hostname = socket.gethostname()
    dns = ["localhost", hostname, f"{hostname}.local"]
    ips = ["127.0.0.1"] + get_local_ips()
    return [f"DNS:{d}" for d in dict.fromkeys(dns)] + [f"IP:{i}" for i in dict.fromkeys(ips)]

def cert_covers(cert_file, san_entries):
    """기존 인증서의 SAN에 현재 필요한 항목이 모두 들어 있는지 확인."""
    try:
        out = subprocess.run(
            ["openssl", "x509", "-in", cert_file, "-noout", "-ext", "subjectAltName"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return False
    # openssl 출력 형식: "DNS:localhost, IP Address:192.168.0.82"
    lines = out.replace("IP Address:", "IP:").splitlines()
    present = {e.strip() for line in lines[1:] for e in line.split(",")}
    return set(san_entries) <= present

def ensure_self_signed_cert(cert_file=SSL_CERT_FILE, key_file=SSL_KEY_FILE):
    """HTTPS용 자체 서명 인증서 준비.

    SAN(subjectAltName)이 없는 인증서는 Chrome/Edge가 거부하고 '이동' 링크도 막으므로,
    인증서가 없거나 SAN에 현재 IP/호스트명이 빠져 있으면 다시 생성한다.
    """
    san_entries = build_san_entries()
    if (os.path.isfile(cert_file) and os.path.isfile(key_file)
            and cert_covers(cert_file, san_entries)):
        return
    print(f"[SSL] Generating self-signed certificate: {cert_file}, {key_file}")
    print(f"[SSL] SAN = {', '.join(san_entries)}")
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-sha256",
            "-keyout", key_file, "-out", cert_file,
            "-days", "825", "-nodes",
            "-subj", f"/CN={socket.gethostname()}",
            "-addext", "subjectAltName=" + ",".join(san_entries),
            "-addext", "basicConstraints=critical,CA:FALSE",
            "-addext", "keyUsage=critical,digitalSignature,keyEncipherment",
            "-addext", "extendedKeyUsage=serverAuth",
        ],
        check=True,
    )
    os.chmod(key_file, 0o600)

# -------------------------------------------------
# MJPEG Output
# -------------------------------------------------
class MJPEGOutput(Output):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.frame_id = 0          # 프레임 번호 (1부터 증가, 빠진 프레임 확인용)
        self.frame_time = time.time()  # 프레임 수신 시각 (Pi 기준 Unix time), 시작 시각으로 초기화해 watchdog 유예
        self.condition = threading.Condition()
        self.recent = deque(maxlen=60)   # (time, size) — 실제 fps/크기 계산용

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=None):
        if stop_event.is_set():
            return
        now = time.time()
        with self.condition:
            self.frame = bytes(frame)
            self.frame_id += 1
            self.frame_time = now
            self.recent.append((now, len(frame)))
            self.condition.notify_all()

    def stats(self):
        with self.condition:
            r = list(self.recent)
            fid, ft = self.frame_id, self.frame_time
        fps = (len(r) - 1) / (r[-1][0] - r[0][0]) if len(r) > 1 and r[-1][0] > r[0][0] else 0.0
        avg = sum(s for _, s in r) // len(r) if r else 0
        return {"frame_id": fid, "last_frame_time": ft, "fps": round(fps, 2), "avg_frame_bytes": avg}

# -------------------------------------------------
# Camera
# -------------------------------------------------
picam2 = Picamera2()
mjpeg_output = MJPEGOutput()
encoder_name = None

def make_encoder(kind):
    if kind == "hw":
        return MJPEGEncoder(bitrate=MJPEG_BITRATE)
    return JpegEncoder(q=JPEG_QUALITY)

def start_camera():
    """고정 해상도/FPS로 카메라 시작. hw 인코더 실패 시 sw로 대체."""
    global encoder_name
    config = picam2.create_video_configuration(
        main={"size": (WIDTH, HEIGHT)},
        buffer_count=4,
        queue=False,        # 저지연: 항상 최신 프레임
    )
    picam2.configure(config)
    frame_us = int(1_000_000 / max(1, FPS))
    picam2.set_controls({
        "AwbEnable": True,
        "AeEnable": True,
        "FrameDurationLimits": (frame_us, frame_us),
    })

    kinds = ["hw", "sw"] if ENCODER == "hw" else ["sw"]
    for kind in kinds:
        try:
            picam2.start_recording(make_encoder(kind), mjpeg_output)
            encoder_name = "MJPEGEncoder(hw)" if kind == "hw" else "JpegEncoder(sw)"
            print(f"[CAM] {WIDTH}x{HEIGHT} @ {FPS}fps, encoder={encoder_name}", flush=True)
            return
        except Exception as e:
            print(f"[CAM] encoder {kind} failed: {e}", flush=True)
            try:
                picam2.stop_recording()
            except Exception:
                pass
    raise RuntimeError("No usable JPEG encoder")

def stop_everything(reason="unknown"):
    """중복 종료 로직 통합."""
    if stop_event.is_set():
        return

    stop_event.set()
    sd_notify(f"STOPPING=1\nSTATUS=Stopping ({reason})")
    with mjpeg_output.condition:
        mjpeg_output.condition.notify_all()

    try:
        picam2.stop_recording()
    except Exception:
        pass
    try:
        picam2.close()
    except Exception:
        pass

# -------------------------------------------------
# systemd watchdog 스레드
# -------------------------------------------------
def systemd_watchdog_thread():
    if not SYSTEMD_OK:
        return

    watchdog_usec = os.getenv("WATCHDOG_USEC")
    if watchdog_usec is None:
        return

    interval = int(watchdog_usec) / 1_000_000 / 2
    print(f"[WATCHDOG] enabled (interval={interval:.2f}s)", flush=True)

    while not stop_event.is_set():
        st = mjpeg_output.stats()
        # 카메라가 멈추면 watchdog 알림을 보내지 않음 → systemd가 서비스 재시작
        if time.time() - st["last_frame_time"] < FRAME_STALL_SEC:
            sd_notify(f"WATCHDOG=1\nSTATUS={WIDTH}x{HEIGHT} {st['fps']}fps {encoder_name}")
        else:
            print("[WATCHDOG] camera stalled, skipping watchdog ping", flush=True)
        time.sleep(interval)

# -------------------------------------------------
# Flask
# -------------------------------------------------
def generate_mjpeg():
    last_id = 0
    while not stop_event.is_set():
        with mjpeg_output.condition:
            # 새 프레임이 올 때만 전송 (같은 프레임 중복 전송 방지)
            mjpeg_output.condition.wait_for(
                lambda: mjpeg_output.frame_id != last_id or stop_event.is_set(), timeout=1.0)
            if mjpeg_output.frame_id == last_id or mjpeg_output.frame is None:
                continue
            frame = mjpeg_output.frame
            last_id = mjpeg_output.frame_id
            ts = mjpeg_output.frame_time

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
               b"X-Frame-Id: " + str(last_id).encode() + b"\r\n"
               b"X-Timestamp: " + f"{ts:.6f}".encode() + b"\r\n\r\n" +
               frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        direct_passthrough=True
    )

@app.route("/snapshot.jpg")
def snapshot():
    with mjpeg_output.condition:
        frame, fid, ts = mjpeg_output.frame, mjpeg_output.frame_id, mjpeg_output.frame_time
    if frame is None:
        return "No frame yet", 503
    return Response(frame, mimetype="image/jpeg", headers={
        "Cache-Control": "no-cache, no-store",
        "X-Frame-Id": str(fid),
        "X-Timestamp": f"{ts:.6f}",
    })

@app.route("/status")
def status():
    st = mjpeg_output.stats()
    st.update({
        "width": WIDTH, "height": HEIGHT, "target_fps": FPS,
        "encoder": encoder_name, "cpu_temp_c": read_cpu_temp_c(),
        "server_time": time.time(),
    })
    return jsonify(st)

@app.route("/shutdown")
def shutdown():
    q = request.args.get("q", "")
    if q.lower() == "q":
        stop_everything("web shutdown")
        func = request.environ.get("werkzeug.server.shutdown")
        if func:
            func()
        return "Shutting down..."
    return "Invalid", 400

@app.route("/")
def index():
    return f"""
    <html>
    <body>
        <h1>Pi Camera Live Stream</h1>
        <img src="/video_feed" width="{WIDTH}" height="{HEIGHT}">
        <p id="st"></p>
        <p>Press Q to shutdown</p>
        <script>
        setInterval(() => fetch("/status").then(r => r.json()).then(s => {{
            document.getElementById("st").textContent =
                `${{s.width}}x${{s.height}} ${{s.fps}} fps, ${{(s.avg_frame_bytes/1000).toFixed(1)}} KB/frame, ` +
                `${{s.encoder}}, ${{s.cpu_temp_c.toFixed(1)}}°C`;
        }}), 1000);
        document.addEventListener("keydown", e => {{
            if (e.key === "q" || e.key === "Q") {{
                fetch("/shutdown?q=Q");
                alert("Shutdown requested");
            }}
        }});
        </script>
    </body>
    </html>
    """

# -------------------------------------------------
# Signal
# -------------------------------------------------
def signal_handler(sig, frame):
    stop_everything(f"signal {sig}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    start_camera()

    ssl_context = None
    if USE_HTTPS:
        ensure_self_signed_cert()
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=SSL_CERT_FILE, keyfile=SSL_KEY_FILE)

    # systemd: 준비 완료 + 상태 메시지
    sd_notify(f"READY=1\nSTATUS=Running ({'HTTPS' if USE_HTTPS else 'HTTP'}) "
              f"{WIDTH}x{HEIGHT}@{FPS} {encoder_name}")

    threading.Thread(target=systemd_watchdog_thread, daemon=True).start()

    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True, ssl_context=ssl_context)
