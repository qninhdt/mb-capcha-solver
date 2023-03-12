import pyppeteer
import asyncio
import base64
import cv2
from uuid import uuid4
from progress.bar import IncrementalBar

MB_LOGIN_URL = 'https://online.mbbank.com.vn/pl/login'
CAPCHA_REFRESH_DELAY = 20  # ms
CAPCHA_FOLDER = 'data/capcha'


class CapchaTester:

    def __init__(self, predict):
        self.predict = predict

    def test_capcha(self, num):
        asyncio.run(self.async_test_capcha(num))

    async def async_test_capcha(self, num):
        print('Starting browser')

        # open browser
        browser = await pyppeteer.launch()
        page = await browser.newPage()
        print('Loading %s' % MB_LOGIN_URL)
        await page.goto(MB_LOGIN_URL)

        bar = IncrementalBar('Testing capcha', max=num,
                             suffix='%(percent)d%% (%(index)d/%(max)d) [ %(elapsed_td)s ]')

        # await page.waitFor(500)
        await page.evaluate("""
            var capcha = document.querySelector('mbb-word-captcha img');
            var userId = document.getElementById('user-id');
            var password = document.getElementById('new-password');
            var capchaInput = document.querySelector('mbb-word-captcha input');
            var loginBtn = document.getElementById('login-btn');
            var refresh = document.getElementById('refresh-captcha');   
        """)

        await page.type('#user-id', 'testtest')
        await page.type('#new-password', 'testtest')

        correct = 0
        total = 0
        for i in range(num):
            capcha_data = await page.evaluate("""
                capcha.src.substring(21);
            """)

            decodeit = open('temp.png', 'wb')
            decodeit.write(base64.b64decode((capcha_data)))
            decodeit.close()

            img = cv2.imread('temp.png')
            result = self.predict(img)

            if result != '000000':
                total += 1

            await page.evaluate("""
                capchaInput.value = '';        
            """)

            await page.type('mbb-word-captcha input', result)
            await page.screenshot({'path': 'screenshot.png'})

            # await page.waitFor(100)
            # GW283 GW21
            await page.evaluate("""
                loginBtn.click();                    
            """)

            await page.waitFor(1000)

            error_code = await page.evaluate("""
                document.querySelector('span.fc-header').textContent                               
            """)

            if error_code == 'GW21':
                correct += 1
                cv2.imwrite('data/results/%s.png' % str(result), img)
            else:
                await page.evaluate("""
                    refresh.click();
                """)
                await page.waitFor(100)

            # await page.evaluate("""
            #     document.querySelector('mbb-word-captcha input').click();
            # """)
            await page.mouse.click(100, 100)

            await page.waitFor(600)

            bar.next()

        print()
        print('Accuracy: %d/%d' % (correct, total))

        # close browser
        print('Close browser')
        await browser.close()
