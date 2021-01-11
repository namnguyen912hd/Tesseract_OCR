import cv2
import numpy as np
import pytesseract
import os
import re

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

per = 25
imgForm = cv2.imread('form.png')
h, w, c = imgForm.shape

orb = cv2.ORB_create(nfeatures=1000)
result1, des1 = orb.detectAndCompute(imgForm, None)

roi = [[(102, 977), (682, 1079), 'text', 'Name'],
       [(742, 979), (1319, 1069), 'text', 'Phone']]

path = 'UserForms'
myImgs = os.listdir(path)

for j, y in enumerate(myImgs):
    img = cv2.imread(path + "/" + y)
    result2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x : x.distance)
    good = matches[:int(len(matches)*(per/100))]
    imgMatche = cv2.drawMatches(img, result2, imgForm, result1, good, None, flags=2)
    srcPts = np.float32([result2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPts = np.float32([result1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPts, dstPts, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w, h))
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []
    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        if r[2] == 'text':
            tempStr = pytesseract.image_to_string(imgCrop)
            resultStr = re.sub(r'[^\w.]', '', tempStr)
            print('{}: {}'.format(r[3], resultStr))
            myData.append(resultStr)
    with open('output.csv', 'a+') as f:
        for data in myData:
            f.write(str(data) + ',')
        f.write('\n')
    imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    cv2.imshow(y + "2", imgShow)

cv2.waitKey(0)








