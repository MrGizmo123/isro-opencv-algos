import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=2,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Random colors for drawing
color = np.random.randint(0, 255, (100, 3))

# Take first frame
ret, old_frame = cap.read()
if not ret:
    print("Failed to capture frame.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

height, width = old_frame.shape[:2]

# Define the central region (square)
square_size = min(width, height) // 2  # Keeping it proportional
x1, y1 = (width - square_size) // 2, (height - square_size) // 2
x2, y2 = x1 + square_size, y1 + square_size

# Convert to grayscale
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Create a mask for feature detection in the central region
mask_region = np.zeros_like(old_gray)
mask_region[y1:y2, x1:x2] = 255  # Only allow detection in the central square

# Find initial tracking points within the central region
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_region, **feature_params)

# Mask for drawing optical flow
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frames grabbed!")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if p0 is not None and len(p0) > 0:
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Filter points that stay within the central region
            filtered_new = []
            filtered_old = []
            
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                if x1 < a < x2 and y1 < b < y2:  # Only keep points within the square
                    filtered_new.append(new)
                    filtered_old.append(old)

            if filtered_new:
                filtered_new = np.array(filtered_new).reshape(-1, 1, 2)
                filtered_old = np.array(filtered_old).reshape(-1, 1, 2)

                # Draw the tracking lines
                for i, (new, old) in enumerate(zip(filtered_new, filtered_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i % len(color)].tolist(), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, color[i % len(color)].tolist(), -1)

                p0 = filtered_new

    # If points are lost, detect new ones inside the central region
    if p0 is None or len(p0) < 5:  # Adjust threshold as needed
        new_points = cv2.goodFeaturesToTrack(frame_gray, mask=mask_region, **feature_params)
        if new_points is not None:
            if p0 is None:
                p0 = new_points
            else:
                p0 = np.vstack((p0, new_points))[:5]  # Keep within limit

    # Draw the central region boundary
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img = cv2.add(frame, mask)
    cv2.imshow('Optical Flow in Central Region', img)

    # Exit condition
    k = cv2.waitKey(30) & 0xFF
    if k == 27:  # ESC key
        break

    old_gray = frame_gray.copy()

cv2.destroyAllWindows()
cap.release()
