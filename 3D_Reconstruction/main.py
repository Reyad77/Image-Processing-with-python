import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    _has_skimage = True
except Exception:
    _has_skimage = False


def _ssim_fallback(imgA, imgB):
    # Convert to grayscale float
    A = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY).astype(np.float64)
    B = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY).astype(np.float64)

    # Gaussian kernel params similar to skimage default
    ksize = (11, 11)
    sigma = 1.5

    muA = cv2.GaussianBlur(A, ksize, sigma)
    muB = cv2.GaussianBlur(B, ksize, sigma)

    muA2 = muA * muA
    muB2 = muB * muB
    muAB = muA * muB

    sigmaA2 = cv2.GaussianBlur(A * A, ksize, sigma) - muA2
    sigmaB2 = cv2.GaussianBlur(B * B, ksize, sigma) - muB2
    sigmaAB = cv2.GaussianBlur(A * B, ksize, sigma) - muAB

    L = 255.0
    K1 = 0.01
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    num = (2 * muAB + C1) * (2 * sigmaAB + C2)
    den = (muA2 + muB2 + C1) * (sigmaA2 + sigmaB2 + C2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


def _ssim_skimage(imgA, imgB):
    # Compute a safe odd win_size for structural_similarity
    h, w = imgA.shape[:2]
    min_dim = min(h, w)
    # default skimage win_size is 7; ensure it's <= min_dim and odd
    win = 7
    if win > min_dim:
        win = min_dim if (min_dim % 2 == 1) else (min_dim - 1)

    if win < 3:
        raise ValueError("images too small for skimage SSIM window")

    # Try channel_axis param first (newer skimage), fallback to multichannel
    try:
        return float(structural_similarity(imgA, imgB, data_range=255, channel_axis=2, win_size=win))
    except TypeError:
        return float(structural_similarity(imgA, imgB, data_range=255, multichannel=True, win_size=win))

# =====================================
# STEP 1: LOAD IMAGES
# =====================================
IMAGE_FOLDER = "images"
image_files = sorted(os.listdir(IMAGE_FOLDER))

images = []
for name in image_files:
    img = cv2.imread(os.path.join(IMAGE_FOLDER, name))
    if img is not None:
        images.append(img)

print(f"Images loaded: {len(images)}")

if len(images) < 2:
    raise RuntimeError("At least two images are required.")

# =====================================
# STEP 2: SHOW ORIGINAL IMAGE
# =====================================
plt.figure(figsize=(6, 4))
plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
plt.title("Step 1: Original Input Image")
plt.axis("off")
plt.show()

# =====================================
# STEP 3: FEATURE DETECTION (SIFT)
# =====================================
sift = cv2.SIFT_create()
keypoints = []
descriptors = []

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    keypoints.append(kp)
    descriptors.append(des)

# Show keypoints on first image
kp_img = cv2.drawKeypoints(
    images[0],
    keypoints[0],
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(6, 4))
plt.imshow(cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB))
plt.title("Step 2: Detected Feature Points (SIFT)")
plt.axis("off")
plt.show()

# =====================================
# STEP 4: FEATURE MATCHING (ONE PAIR FOR DISPLAY)
# =====================================
bf = cv2.BFMatcher()
raw_matches = bf.knnMatch(descriptors[0], descriptors[1], k=2)

good_matches = []
for m, n in raw_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

match_img = cv2.drawMatches(
    images[0], keypoints[0],
    images[1], keypoints[1],
    good_matches[:40], None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(10, 4))
plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
plt.title("Step 3: Feature Matching Between Two Views")
plt.axis("off")
plt.show()

# =====================================
# STEP 4.5: QUALITY EVALUATION (PSNR & SSIM)
# =====================================

# Compute and display PSNR / SSIM between the first two input images (if available)
if len(images) >= 2:
    img0_rgb = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(images[1], cv2.COLOR_BGR2RGB)

    try:
        if _has_skimage:
            try:
                psnr_val = peak_signal_noise_ratio(img0_rgb, img1_rgb, data_range=255)
            except Exception:
                psnr_val = cv2.PSNR(img0_rgb, img1_rgb)

            try:
                ssim_val = _ssim_skimage(img0_rgb, img1_rgb)
            except Exception as e:
                print("skimage SSIM failed, falling back to custom SSIM:", e)
                ssim_val = _ssim_fallback(img0_rgb, img1_rgb)
        else:
            psnr_val = cv2.PSNR(img0_rgb, img1_rgb)
            ssim_val = _ssim_fallback(img0_rgb, img1_rgb)

        print(f"PSNR (image0 vs image1): {psnr_val:.2f} dB")
        print(f"SSIM (image0 vs image1): {ssim_val:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img0_rgb)
        axes[0].set_title("Image 0")
        axes[0].axis("off")

        axes[1].imshow(img1_rgb)
        axes[1].set_title(f"Image 1\nPSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        axes[1].axis("off")

        plt.suptitle("Step 4.5: PSNR and SSIM Evaluation")
        plt.show()

        # Also show a compact text-only output window for metrics
        fig_txt = plt.figure(figsize=(5, 2))
        fig_txt.suptitle("Step 4.5: PSNR & SSIM Evaluation")
        plt.axis('off')
        txt = f"PSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.4f}"
        plt.text(0.5, 0.5, txt, ha='center', va='center', fontsize=12)
        plt.show()
    except Exception as e:
        print("PSNR/SSIM evaluation skipped due to error:", e)

# =====================================
# STEP 5: CAMERA INTRINSICS (ASSUMED)
# =====================================
h, w = images[0].shape[:2]
f = 0.85 * w

K = np.array([
    [f, 0, w / 2],
    [0, f, h / 2],
    [0, 0, 1]
])

# =====================================
# STEP 6: TRIANGULATION (ALL IMAGES)
# =====================================
points_3d = []
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

for i in range(len(images) - 1):
    matches = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        continue

    pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in good])
    pts2 = np.float32([keypoints[i + 1][m.trainIdx].pt for m in good])

    E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
    if E is None:
        continue

    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    P2 = np.hstack((R, t))

    pts4d = cv2.triangulatePoints(K @ P1, K @ P2, pts1.T, pts2.T)
    pts4d /= pts4d[3]

    points_3d.extend(pts4d[:3].T)

points_3d = np.array(points_3d)

print(f"3D points generated: {points_3d.shape[0]}")

# =====================================
# STEP 7: RAW 3D POINT DISTRIBUTION
# =====================================
plt.figure(figsize=(6, 5))
plt.scatter(points_3d[:, 0], points_3d[:, 2], s=1)
plt.title("Step 4: Raw 3D Point Projection (XZ View)")
plt.xlabel("X")
plt.ylabel("Z")
plt.show()

# =====================================
# STEP 8: NORMALIZATION
# =====================================
points_3d -= points_3d.mean(axis=0)
scale = np.percentile(np.linalg.norm(points_3d, axis=1), 95)
points_3d /= scale

X, Y, Z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]

# =====================================
# STEP 9: FINAL 3D CHART OUTPUT
# =====================================
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    X, Y, Z,
    c=Z,
    cmap="plasma",
    s=4,
    alpha=0.85
)

ax.set_title("Final Output: 3D Reconstruction from Multiple 2D Images", fontsize=14)
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_zlabel("Z (Depth)")

plt.colorbar(scatter, ax=ax, label="Depth (Z value)")

# =====================================
# STEP 10: EXPLANATION BELOW FINAL OUTPUT
# =====================================
explanation = (
    "Pipeline Explanation:\n\n"
    "1. Input images are captured from multiple viewpoints.\n"
    "2. Distinct feature points are detected using SIFT.\n"
    "3. Corresponding features are matched between adjacent images.\n"
    "4. Matched points are triangulated to estimate 3D positions.\n"
    "5. The final chart visualizes the reconstructed 3D structure.\n\n"
    "Color represents depth variation along the Z-axis."
)

plt.figtext(
    0.5, 0.02,
    explanation,
    ha="center",
    fontsize=10,
    wrap=True
)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.show()
