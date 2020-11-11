import re
import cv2
import pytesseract
import requests
import numpy as np

url = 'http://localhost:5000/huaxi.jpg';
response = requests.get(url, stream=True).content
img = np.frombuffer(response, np.uint8)
img = cv2.imdecode(img, -1)
cv2.imshow('captcha', img)

'''
img = cv2.medianBlur(img, 5)
cv2.imshow('medianBlur', img)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('morphologyEx', img)
'''


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', img)

img = cv2.medianBlur(img, 5)
cv2.imshow('medianBlur', img)
#ret, cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret, img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
print('threshold: %d', ret)
cv2.imshow('threshold', img)
cv2.imwrite('digit.jpg', img)

cv2.waitKey(0)
cv2.destroyAllWindows()

exit()

img = img.convert("L")  # 转灰度图
pixdata = img.load()
w, h = img.size
threshold = 210
# 遍历所有像素，大于阈值的为黑色
for y in range(h):
    for x in range(w):
        if pixdata[x, y] < threshold:
            pixdata[x, y] = 0
        else:
            pixdata[x, y] = 255
img.show('captcha')


#result = pytesseract.image_to_string(img)
# 可能存在异常符号，用正则提取其中的数字
#regex = '\d+'
#result = ''.join(re.findall(regex, result))
#print(result)
