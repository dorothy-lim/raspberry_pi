#!/bin/bash
# USB 카메라 스트림 수동 실행/종료 (systemd 서비스 없이 테스트할 때)
# 사용법: ./usb_stream_ctl.sh start|stop|restart
# 환경변수: ROTATE(기본 180 — 카메라가 거꾸로 설치됨), USB_FPS, PORT 등
DIR="$(cd "$(dirname "$0")" && pwd)"
export ROTATE="${ROTATE:-180}"
stop() { pkill -f "python3 $DIR/usb_camera_stream.py" && sleep 2; }
start() {
  setsid nohup python3 "$DIR/usb_camera_stream.py" > /tmp/usb_stream.log 2>&1 < /dev/null &
  sleep 5; grep -E "^\[USB\]|Error" /tmp/usb_stream.log
}
case "$1" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  *) echo "usage: $0 start|stop|restart" ;;
esac
