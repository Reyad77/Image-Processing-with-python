
import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('D:\Study\sem 5\Image processing\projects\Experiment 3\Experiment 3\leaf.jpg')
if image is None:
    print("Error: Could not read 'leaf.jpg'. Please check the file path.")
    exit()


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Color Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
plt.show()


print(f"Image shape: {image.shape}")
print(f"Grayscale shape: {gray.shape}")
print(f"Image dtype: {gray.dtype}")
print(f"Min pixel value: {gray.min()}, Max pixel value: {gray.max()}")