import pyppeteer
import asyncio
import base64
import os
import cv2
import numpy as np
from uuid import uuid4
from functools import cmp_to_key
from progress.bar import IncrementalBar

MB_LOGIN_URL = 'https://online.mbbank.com.vn/pl/login'
CAPCHA_REFRESH_DELAY = 20  # ms
CAPCHA_FOLDER = 'data/capcha'


class CapchaManager:

    def __init__(self):
        self.digits = []
        self.height = []
        self.width = []
        self.offset = []

    def dowload_capcha(self, num):
        asyncio.run(self.async_dowload_capcha(num))

    async def async_dowload_capcha(self, num):
        print('Starting browser')

        # open browser
        browser = await pyppeteer.launch()
        page = await browser.newPage()
        print('Loading %s' % MB_LOGIN_URL)
        await page.goto(MB_LOGIN_URL)

        bar = IncrementalBar('Dowloading capcha', max=num,
                             suffix='%(percent)d%% (%(index)d/%(max)d) [ %(elapsed_td)s ]')

        old_capcha_data = None

        await page.evaluate("""
            var capcha = document.querySelector('mbb-word-captcha img');
            var refresh = document.getElementById('refresh-captcha');                 
        """)

        for i in range(num):

            # click refresh button to update capcha
            await page.evaluate("""
                refresh.dispatchEvent(
                    new MouseEvent('click', {
                        bubbles: true,
                        cancelable: true,
                        view: window
                    })
                );
            """)

            capcha_data = old_capcha_data
            while old_capcha_data == capcha_data:
                await page.waitFor(CAPCHA_REFRESH_DELAY)

                # get capcha image data in base64
                capcha_data = await page.evaluate("""
                    capcha.src.substring(21);
                """)

            old_capcha_data = capcha_data

            capcha_id = str(uuid4())

            decodeit = open('data/capcha/{}.png'.format(capcha_id), 'wb')
            decodeit.write(base64.b64decode((capcha_data)))
            decodeit.close()

            bar.next()

        print()

        # close browser
        print('Close browser')
        await browser.close()

    @staticmethod
    def show_image(title, img, scale=1):
        img = cv2.resize(
            img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(title, img)

    @staticmethod
    def pack_images(images, n=32):
        return cv2.vconcat([
            cv2.hconcat([images[j + i*n] for j in range(n)]) for i in range(n)
        ])

    @staticmethod
    def unpack_images(image, n=32):
        images = []

        s = image.shape[0] // n

        for i in range(n):
            for j in range(n):
                images.append(image[i*s:(i+1)*s, j*s:(j+1)*s])

        return images

    def preprocess_capcha(self):
        folder = os.listdir('data/capcha')

        bar = IncrementalBar('Preprocess capcha', max=len(folder),
                             suffix='%(percent)d%% (%(index)d/%(max)d) [ %(elapsed_td)s ]')

        digits_ = []

        for i, file in enumerate(folder):
            # if i == 3000:
            #     break
            img = cv2.imread('data/capcha/{}'.format(file))
            # img = cv2.imread('mb7.png')

            digits, height, width, offset = self.extract_digits(img)

            self.height += height
            self.width += width
            self.digits += digits
            self.offset += offset

            for digit in digits:
                digits_.append(digit)

                if len(digits_) == 32*32:
                    digit_id = str(uuid4())
                    cv2.imwrite('data/digits/%s.png' %
                                digit_id, self.pack_images(digits_))
                    digits_ = []

            # break
            bar.next()
        print()

    def extract_digits(self, img):
        img = cv2.resize(
            img, (img.shape[1] * 8, img.shape[0] * 8), interpolation=cv2.INTER_NEAREST)
        pad = 10
        img = img[pad:img.shape[0]-pad, pad:img.shape[1]-pad]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)

        contours_, hierarchy = cv2.findContours(
            gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = []

        for i in range(hierarchy.shape[1]):
            if hierarchy[0, i, 3] == -1:
                contours.append(contours_[i])

        def contourCmp(cnt1, cnt2):
            bb1 = cv2.boundingRect(cnt1)
            bb2 = cv2.boundingRect(cnt2)

            if bb1[0] > bb2[0]:
                return 1
            else:
                return -1

        contours = sorted(contours, key=cmp_to_key(contourCmp))

        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

        # gray = cv2.dilate(gray, np.ones((5, 5), dtype='uint8'), iterations=2)

        results = []
        height = []
        width = []
        offset = []

        for i, cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(img, [box], -1, (0, 0, 255), 3)

            bb = cv2.boundingRect(cnt)

            mask = np.zeros(img.shape[:2], dtype='uint8')
            cv2.drawContours(mask, contours, i, 255, -1)

            temp = np.zeros(img.shape[:2], dtype='uint8')
            temp = cv2.bitwise_and(gray, gray, mask=mask)

            digit_ = temp[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]

            h, w = digit_.shape[:2]
            s = 256

            if h >= s or w >= s:
                return [], [], [], []

            digit = np.zeros((s, s), dtype='uint8')

            digit[(s-h)//2:(s+h)//2, (s-w)//2:(s+w)//2] = digit_

            center = (s//2, s//2)

            if rect[2] < 45:
                M = cv2.getRotationMatrix2D(center, rect[2], 1.0)
                digit = cv2.warpAffine(
                    digit, M, (s, s), flags=cv2.INTER_NEAREST)
            else:
                M = cv2.getRotationMatrix2D(center, rect[2]-90, 1.0)
                digit = cv2.warpAffine(
                    digit, M, (s, s), flags=cv2.INTER_NEAREST)

            digit = cv2.resize(
                digit, (32, 32), interpolation=cv2.INTER_NEAREST)

            # digit = cv2.morphologyEx(
            #     digit, cv2.MORPH_CLOSE, np.ones((2, 2), dtype='uint8'), iterations=2)

            results.append(digit)
            height.append(rect[1][1])
            width.append(rect[1][0])
            offset.append(bb[1]+bb[3]//2)

        #     self.show_image('Digit %d: %d deg' % (i, rect[2]), digit, 8)

        # self.show_image('Capcha', img)
        # self.show_image('Gray', gray)

        return results, height, width, offset

    def load_digits(self):
        folder = os.listdir('data/digits')

        bar = IncrementalBar('Load digits', max=len(folder),
                             suffix='%(percent)d%% (%(index)d/%(max)d) [ %(elapsed_td)s ]')

        for i, file in enumerate(folder):
            digit = cv2.imread('data/digits/{}'.format(file), 0)
            digits_ = self.unpack_images(digit)

            self.digits += digits_
            bar.next()
        print()
