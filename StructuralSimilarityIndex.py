import glob

from skimage.metrics import structural_similarity
import cv2
import numpy as np


def search_similarity():
    # first = cv2.imread('Banana1 (1).jpg')
    # second = cv2.imread('Banana1 (3).jpg')
    # first = cv2.resize(first, (1200, 800))
    # second = cv2.resize(second, (1200, 800))
    image_names = list(glob.glob(r'folder_for_frames/*.jpg'))
    # image_names = list(r'folder_for_frames/*.jpg')
    #     # Convert images to grayscale
    for i in range(len(image_names)-1):
        first = cv2.imread(image_names[i])
        second = cv2.imread(image_names[i+1])
        first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two images
        score, diff = structural_similarity(first_gray, second_gray, full=True)
        print("Similarity Score: {:.3f}%".format(score * 100))
        if score<0.9:
            continue

        # The diff image contains the actual image differences between the two images
        # and is represented as a floating point data type so we must convert the array
        # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
        diff = (diff * 255).astype("uint8")

        # Threshold the difference image, followed by finding contours to
        # obtain the regions that differ between the two images
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # Highlight differences
        mask = np.zeros(first.shape, dtype='uint8')
        filled = second.copy()

        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(first, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.rectangle(second, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
                cv2.drawContours(filled, [c], 0, (0, 255, 0), -1)

        # cv2.imshow('first', first)
        # cv2.imshow('second', second)
        cv2.imshow('diff', diff)
        cv2.imshow('mask', mask)
        cv2.imshow('filled', filled)
        # cv2.waitKey()
        if cv2.waitKey() == 27:
            break




if __name__ == '__main__':
    search_similarity()
