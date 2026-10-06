# -*- coding: utf-8 -*-
"""Pi 카메라 MJPEG 스트림 리더 (저지연).

cv2.VideoCapture(url)은 내부 버퍼 때문에 처리 속도가 느리면 지연이 계속 쌓인다.
이 리더는 별도 스레드에서 스트림을 계속 받아 '가장 최신 프레임'만 보관하므로,
추론이 30fps보다 느려도 지연이 쌓이지 않는다 (중간 프레임은 버려짐).

사용 예:
    reader = MJPEGStreamReader("http://192.168.0.82:8000/video_feed").start()
    frame = reader.read()          # 새 프레임 (numpy BGR) — frame.image, frame.frame_id, frame.timestamp
    reader.stop()
"""
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests


@dataclass
class Frame:
    image: np.ndarray      # BGR 이미지
    frame_id: int          # 서버 프레임 번호 (X-Frame-Id)
    timestamp: float       # Pi에서 프레임을 받은 시각 (X-Timestamp, Unix time), 없으면 수신 시각
    received: float        # 이 PC에서 받은 시각 (time.time())


class MJPEGStreamReader:
    def __init__(self, url, timeout=5.0, reconnect_delay=1.0):
        self.url = url
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self._latest = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self.connected = False
        self.frames_received = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def read(self, timeout=2.0, last_id=None):
        """last_id 이후의 새 프레임을 기다려 반환. timeout 동안 없으면 None."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._latest is not None
                and (last_id is None or self._latest.frame_id != last_id),
                timeout=timeout)
            return self._latest if ok else None

    # ---------------------------------------------
    def _run(self):
        while not self._stop.is_set():
            try:
                with requests.get(self.url, stream=True, timeout=self.timeout) as r:
                    r.raise_for_status()
                    self.connected = True
                    self._parse(r.raw)
            except Exception as e:
                if not self._stop.is_set():
                    print(f"[reader] {type(e).__name__}: {e} — reconnecting")
            self.connected = False
            self._stop.wait(self.reconnect_delay)

    def _parse(self, raw):
        """multipart 파트 헤더(Content-Length, X-Frame-Id, X-Timestamp)를 읽고 JPEG 본문을 디코드."""
        while not self._stop.is_set():
            line = raw.readline()
            if not line:
                return  # 연결 끊김
            if not line.startswith(b"--"):
                continue
            headers = {}
            while True:
                h = raw.readline()
                if not h:
                    return
                h = h.strip()
                if not h:
                    break
                k, _, v = h.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            n = int(headers.get("content-length", 0))
            if n <= 0:
                continue
            data = raw.read(n)
            if len(data) != n:
                return
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            now = time.time()
            frame = Frame(
                image=img,
                # 헤더가 없는 서버면 수신 순번을 대신 사용
                frame_id=int(headers.get("x-frame-id", self.frames_received + 1)),
                timestamp=float(headers.get("x-timestamp", now)),
                received=now,
            )
            with self._cond:
                self._latest = frame
                self.frames_received += 1
                self._cond.notify_all()
