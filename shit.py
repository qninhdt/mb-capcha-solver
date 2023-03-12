from core import CapchaManager
import cv2
import numpy as np

cm = CapchaManager()

img = cv2.imread('mb8.png')
digits = cm.extract_digits(img)

while True:
    if cv2.waitKey(0) & 0xFF == 27:
        break
cv2.destroyAllWindows()
