import cv2
import numpy as np
import pytesseract    # version: 0.3.4
import os

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

per = 25
pixelThreshold = 500

pathImgForm = 'form.png'
imgForm = cv2.imread(pathImgForm)
h, w, c = imgForm.shape

orb = cv2.ORB_create(nfeatures=1000)
result1, des1 = orb.detectAndCompute(imgForm, None)

roi = [[(102, 977), (682, 1079), 'text', 'Name'],
        [(742, 979), (1319, 1069), 'text', 'Phone'],
        [(99, 1152), (144, 1199), 'box', 'Check1'],
        [(742, 1149), (789, 1197), 'box', 'Check2']]

path = 'UserForms'
myImgs = os.listdir(path)

for j, y in enumerate(myImgs):    #j: index ; y: value
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
            print('{}: {}'.format(r[3], tempStr))
            myData.append(tempStr)
        if r[2] == 'box':
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)
            if totalPixels > pixelThreshold:
                totalPixels = 1
            else:
                totalPixels = 0
            print('{}: {}'.format(r[3], totalPixels))
            myData.append(totalPixels)

    #save data
    with open('output.csv', 'a+') as f:
        for data in myData:
            f.write(str(data) + ',')
        f.write('\n')
    imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    cv2.imshow(y + "2", imgShow)

cv2.waitKey(0)








