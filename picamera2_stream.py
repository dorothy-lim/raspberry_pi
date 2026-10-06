# -*- coding: utf-8 -*-
import threading
import time
import signal
import sys
import os
import ssl
import socket
import subprocess
from collections import deque
from flask import Flask, Response, request

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import Output

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

# -------------------------------------------------
# psutil (선택)
# -------------------------------------------------
try:
    import psutil
    PSUTIL_OK = True
except Exception:
    psutil = None
    PSUTIL_OK = False

app = Flask(__name__)

# -------------------------------------------------
# 튜닝 파라미터
# -------------------------------------------------
RES_LEVELS = [(320, 240), (640, 480), (1280, 720)]
Q_MIN, Q_MAX = 55, 85
FPS_MIN, FPS_MAX = 6, 20

TEMP_HOT, TEMP_COOL = 70.0, 60.0
CPU_HOT, CPU_COOL = 75.0, 45.0
CPU_WINDOW = 5

TARGET_FRAME_BYTES = 80_000
TUNE_INTERVAL_SEC = 3.0

CCM_INDOOR = [
    1.30, -0.15, -0.15,
   -0.10,  1.20, -0.10,
   -0.05, -0.20,  1.25
]

# -------------------------------------------------
# HTTPS(자체 서명 인증서) 설정
# 기본은 HTTP. USE_HTTPS=1 환경변수로 HTTPS(자체 서명 인증서) 실행 가능
# -------------------------------------------------
USE_HTTPS = os.getenv("USE_HTTPS", "0") == "1"
SSL_CERT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cert.pem")
SSL_KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "key.pem")

# 초기 상태
state_lock = threading.Lock()
current_res_idx = 1
current_q = 80
current_fps = 15

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

def read_cpu_percent():
    if not PSUTIL_OK:
        return -1.0
    try:
        return float(psutil.cpu_percent(interval=0.2))
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
        self.condition = threading.Condition()
        self.last_sizes = deque(maxlen=30)

    def outputframe(self, frame, keyframe=True, timestamp=None, packet=None, audio=None):
        if stop_event.is_set():
            return
        with self.condition:
            self.frame = frame
            self.last_sizes.append(len(frame))
            self.condition.notify_all()

    def avg_frame_size(self):
        with self.condition:
            return sum(self.last_sizes) // len(self.last_sizes) if self.last_sizes else 0

# -------------------------------------------------
# Camera
# -------------------------------------------------
picam2 = Picamera2()
mjpeg_output = MJPEGOutput()
encoder = None

def apply_fps_limit(fps):
    frame_us = int(1_000_000 / max(1, fps))
    picam2.set_controls({"FrameDurationLimits": (frame_us, frame_us)})

def restart_camera(resolution, q, fps):
    """카메라 재시작(해상도/q/fps 반영). stop_event면 아무 것도 하지 않음."""
    global encoder
    if stop_event.is_set():
        return

    # recording 중이면 정지
    try:
        picam2.stop_recording()
    except Exception:
        pass

    config = picam2.create_video_configuration(
        main={"size": resolution},
        queue=False,        # 저지연
        buffer_count=2
    )
    picam2.configure(config)

    picam2.set_controls({
        "AwbEnable": True,
        "AeEnable": True,
        "ColourCorrectionMatrix": CCM_INDOOR
    })
    time.sleep(0.2)

    apply_fps_limit(fps)

    encoder = JpegEncoder(q=q)  # (현 구조에서 CPU 낮은 쪽 유지)
    picam2.start_recording(encoder, mjpeg_output)

def stop_everything(reason="unknown"):
    """중복 종료 로직 통합."""
    if stop_event.is_set():
        return

    stop_event.set()
    sd_notify(f"STOPPING=1\nSTATUS=Stopping ({reason})")

    try:
        picam2.stop_recording()
    except Exception:
        pass
    try:
        picam2.close()
    except Exception:
        pass

# -------------------------------------------------
# 자동 튜닝 스레드
# -------------------------------------------------
def tuning_thread():
    global current_res_idx, current_q, current_fps
    cpu_hist = deque(maxlen=CPU_WINDOW)

    while not stop_event.is_set():
        time.sleep(TUNE_INTERVAL_SEC)

        temp = read_cpu_temp_c()
        cpu = read_cpu_percent()
        if cpu >= 0:
            cpu_hist.append(cpu)
        cpu_avg = sum(cpu_hist) / len(cpu_hist) if cpu_hist else -1

        avg_size = mjpeg_output.avg_frame_size()

        with state_lock:
            res_idx, q, fps = current_res_idx, current_q, current_fps

        new_res, new_q, new_fps = res_idx, q, fps
        need_restart = False

        overload = (
            (temp >= 0 and temp >= TEMP_HOT) or
            (cpu_avg >= 0 and cpu_avg >= CPU_HOT) or
            (avg_size > 0 and avg_size > TARGET_FRAME_BYTES * 1.2)
        )

        if overload:
            if new_q > Q_MIN:
                new_q -= 10
            elif new_fps > FPS_MIN:
                new_fps -= 3
            elif new_res > 0:
                new_res -= 1
            need_restart = True
        else:
            cool = (
                (temp < 0 or temp <= TEMP_COOL) and
                (cpu_avg < 0 or cpu_avg <= CPU_COOL) and
                (avg_size == 0 or avg_size < TARGET_FRAME_BYTES * 0.8)
            )
            if cool:
                if new_fps < FPS_MAX:
                    new_fps += 2
                elif new_res < len(RES_LEVELS) - 1:
                    new_res += 1
                elif new_q < Q_MAX:
                    new_q += 3
                need_restart = True

        if need_restart and not stop_event.is_set():
            with state_lock:
                current_res_idx = max(0, min(new_res, len(RES_LEVELS) - 1))
                current_q = max(Q_MIN, min(new_q, Q_MAX))
                current_fps = max(FPS_MIN, min(new_fps, FPS_MAX))
                resolution = RES_LEVELS[current_res_idx]

            print(f"[TUNE] temp={temp:.1f}C cpu={cpu_avg:.1f}% size={avg_size}B -> "
                  f"res={resolution} q={current_q} fps={current_fps}")

            restart_camera(resolution, current_q, current_fps)

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
    print(f"[WATCHDOG] enabled (interval={interval:.2f}s)")
    sd_notify(f"STATUS=Watchdog enabled ({interval:.2f}s)")

    while not stop_event.is_set():
        sd_notify("WATCHDOG=1")
        time.sleep(interval)

# -------------------------------------------------
# Flask
# -------------------------------------------------
def generate_mjpeg():
    while not stop_event.is_set():
        with mjpeg_output.condition:
            mjpeg_output.condition.wait(timeout=0.5)
            frame = mjpeg_output.frame

        if not frame:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
               frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        direct_passthrough=True
    )

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
    return """
    <html>
    <body>
        <h1>Pi Camera Live Stream</h1>
        <img src="/video_feed" width="640" height="480">
        <p>Press Q to shutdown</p>
        <script>
        document.addEventListener("keydown", e => {
            if (e.key === "q" || e.key === "Q") {
                fetch("/shutdown?q=Q");
                alert("Shutdown requested");
            }
        });
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
    with state_lock:
        restart_camera(RES_LEVELS[current_res_idx], current_q, current_fps)

    ssl_context = None
    if USE_HTTPS:
        ensure_self_signed_cert()
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=SSL_CERT_FILE, keyfile=SSL_KEY_FILE)

    # systemd: 준비 완료 + 상태 메시지
    sd_notify(f"READY=1\nSTATUS=Running ({'HTTPS' if USE_HTTPS else 'HTTP'})")

    threading.Thread(target=tuning_thread, daemon=True).start()
    threading.Thread(target=systemd_watchdog_thread, daemon=True).start()

    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True, ssl_context=ssl_context)
