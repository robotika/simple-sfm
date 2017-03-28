
import numpy as np
import cv2
from time import clock

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

def draw_str(dst, target, s):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)

def main():
    track_len = 10
    detect_interval = 5
    tracks = []
    cam = cv2.VideoCapture(0)
    frame_idx = 0
    step_time, detect_time, last_time = 0.0, 0.0, clock()-33/1000
    fps = 0
    while True:
        now = clock()
        fps = (15*fps + 1/(now-last_time))/16
        last_time = now
        ret, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()

        if len(tracks) > 0:
            img0, img1 = prev_gray, frame_gray
            p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
            before = clock()
            p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
            p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
            after = clock()
            step_time = (5*step_time + (after-before)*1000)/6
            d = abs(p0-p0r).reshape(-1, 2).max(-1)
            good = d < 1
            new_tracks = []
            for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
                if not good_flag:
                    continue
                tr.append((x, y))
                if len(tr) > track_len:
                    del tr[0]
                new_tracks.append(tr)
                cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
            tracks = new_tracks
            cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))
            draw_str(vis, (20, 20), 'track count: %d' % len(tracks))
            draw_str(vis, (20, 40), 'step time: %dms' % (step_time))
            draw_str(vis, (20, 60), 'detect time: %dms' % (detect_time))
            draw_str(vis, (20, 80), 'FPS: %d' % (fps))

        if frame_idx % detect_interval == 0:
            before = clock()
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            for x, y in [np.int32(tr[-1]) for tr in tracks]:
                cv2.circle(mask, (x, y), 5, 0, -1)
            p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    tracks.append([(x, y)])
            after = clock()
            detect_time = (5*detect_time + (after - before)*1000)/6


        frame_idx += 1
        prev_gray = frame_gray
        cv2.imshow('odometry', vis)

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
