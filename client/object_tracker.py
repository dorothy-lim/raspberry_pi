# -*- coding: utf-8 -*-
"""Pi 카메라 스트림 물체 인식 + 추적 (YOLO + ByteTrack).

    python object_tracker.py                                  # 기본: http://192.168.0.82:8000, 사람/전체 클래스
    python object_tracker.py --classes person car --conf 0.5
    python object_tracker.py --model yolo11s.pt --device cuda:0 --save out.mp4
    python object_tracker.py --no-show --log tracks.csv       # 화면 없이 CSV 기록
    python object_tracker.py --url http://192.168.0.82:8001/video_feed --rotate 180   # USB 카메라(거꾸로 설치)

화면 키: q / ESC 종료
"""
import argparse
import csv
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from mjpeg_reader import MJPEGStreamReader

TRAIL_LEN = 40          # 궤적 점 개수
TRACK_TTL_SEC = 2.0     # 이 시간 동안 안 보이면 궤적 삭제


def color_for(track_id):
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def parse_args():
    p = argparse.ArgumentParser(description="Pi camera object detection & tracking")
    p.add_argument("--url", default="http://192.168.0.82:8000/video_feed")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="받은 영상을 시계 방향으로 회전 (거꾸로 달린 카메라는 180)")
    p.add_argument("--model", default="yolo11n.pt", help="YOLO 모델 (처음 실행 시 자동 다운로드)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml 또는 botsort.yaml")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None, help="cpu, cuda:0, mps 등 (기본 자동)")
    p.add_argument("--classes", nargs="*", default=None,
                   help="추적할 클래스 이름 또는 번호 (예: person car 0 2). 생략 시 전체")
    p.add_argument("--no-show", action="store_true", help="화면 표시 안 함")
    p.add_argument("--save", default=None, help="결과 영상 저장 경로 (mp4)")
    p.add_argument("--log", default=None, help="추적 결과 CSV 저장 경로")
    p.add_argument("--max-frames", type=int, default=0, help="0이면 무제한 (테스트용)")
    return p.parse_args()


def resolve_classes(model, classes):
    if not classes:
        return None
    name_to_id = {v: k for k, v in model.names.items()}
    ids = []
    for c in classes:
        if c.isdigit():
            ids.append(int(c))
        elif c in name_to_id:
            ids.append(name_to_id[c])
        else:
            raise SystemExit(f"알 수 없는 클래스: {c} (사용 가능: {', '.join(model.names.values())})")
    return ids


def main():
    args = parse_args()
    model = YOLO(args.model)
    class_ids = resolve_classes(model, args.classes)

    reader = MJPEGStreamReader(args.url, rotate=args.rotate).start()
    print(f"[tracker] connecting {args.url} ...")

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LEN))   # id -> [(t, cx, cy)]
    last_seen = {}
    writer = None
    log_f = log_w = None
    if args.log:
        log_f = open(args.log, "w", newline="", encoding="utf-8")
        log_w = csv.writer(log_f)
        log_w.writerow(["pi_timestamp", "frame_id", "track_id", "class", "conf",
                        "x1", "y1", "x2", "y2", "cx", "cy", "vx_px_s", "vy_px_s"])

    last_id = None
    processed = 0
    skipped = 0
    t_prev = time.time()
    proc_fps = 0.0

    try:
        while True:
            frame = reader.read(timeout=3.0, last_id=last_id)
            if frame is None:
                print("[tracker] waiting for frames ...")
                continue
            if last_id is not None and frame.frame_id > last_id + 1:
                skipped += frame.frame_id - last_id - 1   # 추론이 느려 건너뛴 프레임 (지연 누적 방지)
            last_id = frame.frame_id
            img = frame.image
            t = frame.timestamp          # Pi 촬영 시각 기준 → 속도 계산에 사용

            results = model.track(img, persist=True, tracker=args.tracker, conf=args.conf,
                                  imgsz=args.imgsz, classes=class_ids, device=args.device,
                                  verbose=False)
            r = results[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.int().cpu().tolist()
                clss = r.boxes.cls.int().cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()
                for (x1, y1, x2, y2), tid, c, cf in zip(boxes, ids, clss, confs):
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    trail = trails[tid]
                    trail.append((t, cx, cy))
                    last_seen[tid] = t

                    # 속도 (px/s): 궤적 처음~끝 기준
                    vx = vy = 0.0
                    if len(trail) >= 2 and trail[-1][0] > trail[0][0]:
                        dt = trail[-1][0] - trail[0][0]
                        vx = (trail[-1][1] - trail[0][1]) / dt
                        vy = (trail[-1][2] - trail[0][2]) / dt

                    if log_w:
                        log_w.writerow([f"{t:.6f}", frame.frame_id, tid, model.names[c], f"{cf:.3f}",
                                        int(x1), int(y1), int(x2), int(y2),
                                        f"{cx:.1f}", f"{cy:.1f}", f"{vx:.1f}", f"{vy:.1f}"])

                    col = color_for(tid)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                    label = f"#{tid} {model.names[c]} {cf:.2f}"
                    # 화면 위쪽(HUD 영역)이면 라벨을 박스 안쪽에 표시
                    ly = int(y1) - 6 if y1 > 45 else int(y1) + 18
                    cv2.putText(img, label, (int(x1) + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                    pts = np.array([(int(px), int(py)) for _, px, py in trail], np.int32)
                    if len(pts) >= 2:
                        cv2.polylines(img, [pts], False, col, 2)
                    cv2.arrowedLine(img, (int(cx), int(cy)),
                                    (int(cx + vx * 0.3), int(cy + vy * 0.3)), col, 2, tipLength=0.3)

            # 오래 안 보인 궤적 정리
            for tid in [k for k, ts in last_seen.items() if t - ts > TRACK_TTL_SEC]:
                trails.pop(tid, None)
                last_seen.pop(tid, None)

            # 처리 fps (지수 평균)
            now = time.time()
            proc_fps = 0.9 * proc_fps + 0.1 * (1.0 / max(1e-6, now - t_prev)) if processed else 0.0
            t_prev = now
            processed += 1
            latency_ms = (now - frame.received) * 1000   # 이 PC 수신 → 처리 완료
            hud = (f"proc {proc_fps:4.1f} fps | infer+draw {latency_ms:4.0f} ms | "
                   f"tracks {len(trails)} | skipped {skipped}")
            (tw, th), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            roi = img[0:th + 12, 0:tw + 12]
            roi[:] = (roi * 0.4).astype(np.uint8)       # 반투명 어두운 배경
            cv2.putText(img, hud, (6, th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

            if args.save:
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
                writer.write(img)

            if not args.no_show:
                cv2.imshow("Pi camera tracking", img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            elif processed % 30 == 0:
                print(hud)

            if args.max_frames and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        if writer:
            writer.release()
        if log_f:
            log_f.close()
        cv2.destroyAllWindows()
        print(f"[tracker] processed {processed} frames, skipped {skipped}")


if __name__ == "__main__":
    main()
