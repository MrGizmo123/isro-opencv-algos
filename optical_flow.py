import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=20,  # Reduced maxCorners to limit points
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Random colors for drawing
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners
ret, old_frame = cap.read()
if not ret:
    print("Failed to capture frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Mask for drawing
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % len(color)].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i % len(color)].tolist(), -1)

            p0 = good_new.reshape(-1, 1, 2)
            print(p0)
            
    # If points are lost, detect new ones
    if p0 is None or len(p0) < 1:  # Lower threshold for detecting new points
        new_points = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        if new_points is not None:
            if p0 is None:
                p0 = new_points
            else:
                p0 = np.vstack((p0, new_points))[:5]  # Limit total points to 5

    img = cv2.add(frame, mask)
    cv2.imshow('Optical Flow', img)

    # Exit condition
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # ESC key
        break

    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()
