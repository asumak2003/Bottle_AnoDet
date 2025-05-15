import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def pixel_above_thresh(img, grauwerte_threshold=225):

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create a mask for pixels exceeding the threshold
    mask = grayscale > grauwerte_threshold
    
    # Create an output image to visualize differences
    diff_image = np.zeros_like(img)  # Initialize blank image
    diff_image[mask] = 255  # Highlight pixels exceeding threshold

    masked_img = cv2.bitwise_and(img, diff_image)

    return diff_image, masked_img

# Load the two images
file_name = "C:/Users/Arian/OneDrive - Hochschule Heilbronn/data_seminararbeit/wrong_lid/cropped/nd_0020.jpg"
img = cv2.imread(file_name)
diff_image, masked_img = pixel_above_thresh(img, grauwerte_threshold=225)





# Display the result
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.title("Normales Bild")
plt.imshow(img, cmap='gray')
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Pixel Unterschiede (Threshold: 225)")
plt.imshow(diff_image, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()