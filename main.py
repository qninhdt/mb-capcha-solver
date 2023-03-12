from sklearn.cluster import KMeans
import cv2
import os
from uuid import uuid4
import numpy as np
from core import CapchaManager, CapchaTester
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# for i in range(43):
#     os.makedirs('data/output/%d' % i)

capcha_manager = CapchaManager()
# capcha_manager.dowload_capcha(2000)
# capcha_manager.preprocess_capcha()
capcha_manager.load_digits()

# while True:
#     if cv2.waitKey(0) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()

init_digits = []

for x in os.listdir('data/init_digits'):
    digit = cv2.imread('data/init_digits/{}'.format(x), 0)
    init_digits.append(digit)

init_digits = np.array([digit.reshape((32 * 32)) for digit in init_digits])

model = KMeans(n_clusters=43, init=init_digits, n_init=1, max_iter=3000)

X = np.array([digit.reshape((32 * 32)) for digit in capcha_manager.digits])

print('Training . . .')
model.fit(X)
print('Done')

for i, x in enumerate(model.cluster_centers_):
    x = x.reshape((32, 32))
    cv2.imwrite('data/unique_digits/{}.png'.format(i), x)

# digits_ = [[] for _ in range(43)]
# for i in range(X.shape[0]):
#     k = model.labels_[i]
#     digits_[k].append(X[i].reshape((32, 32)))

#     if len(digits_[k]) == 8*8:
#         cv2.imwrite('data/output/%d/%s.png' % (k, str(uuid4())),
#                     capcha_manager.pack_images(digits_[k], 8))
#         digits_[k] = []

# print(h)

symbols = [
    '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'A', 'b_or_h', 'B', 'c', 'C', 'd', 'D', 'e', 'E',
    'g', 'G', 'b_or_h', 'H', 'k', 'K', 'm', 'M', 'n', 'N',
    'p_or_P', 'p_or_P', 'q', 'Q', 't', 'u', 'U', 'v_or_V', 'v_or_V',
    'y_or_Y', 'y_or_Y', 'z', 'Z', '4', '4'
]


def get_symbol(digit, n, height, width, offset):
    s = symbols[n]

    if s == 'p_or_P':
        if offset > 230:
            s = 'p'
        else:
            s = 'P'

    if s == 'v_or_V':
        if width < 170:
            s = 'v'
        else:
            s = 'V'

    if s == 'y_or_Y':
        if offset > 230:
            s = 'y'
        else:
            s = 'Y'

    if s == 'b_or_h':
        s = 'h'

        for i in range(10):
            if digit[31-i, 20] == 1:
                s = 'b'
                break

    return s


def predict(img):
    digits, heights, widths, offsets = capcha_manager.extract_digits(img)

    result = ''

    if len(digits) != 6:
        return '000000'

    for i in range(6):
        k = model.predict(np.array([digits[i].reshape((32*32))]))[0]
        s = get_symbol(digits[i], k, heights[i], widths[i], offsets[i])

        result += s

    return result


capcha_tester = CapchaTester(predict)

capcha_tester.test_capcha(100)

# print(predict(cv2.imread('mb11.png')))
