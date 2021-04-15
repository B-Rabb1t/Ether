import cv2
import numpy as np
from matplotlib import pyplot as plt

import imutils

def detect_one(img_gray, template):
    # template = cv2.Canny(template, 50, 250)
    # img_gray = cv2.Canny(img_gray, 50, 250)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)

    min_Val, max_Val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_Val, min_loc)

    width, height = template.shape
    stanga_sus = min_loc
    dreapta_jos = (stanga_sus[0] + width, stanga_sus[1] + height)
    cv2.rectangle(img_gray, stanga_sus, dreapta_jos, (255, 0, 0), 2)

    cv2.imshow("gasit", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_multe(img_gray, template, threshold = 0.33):
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where(res > threshold)
    width, height = template.shape

    for avion in zip(*loc[::-1]):
        cv2.rectangle(img_gray, avion, (avion[0] + width, avion[1] + height), (255, 0 ,0), 2)
    
    cv2.imshow("gasit", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_toate(img_gray, template, threshold = 0.66):

    for avion in range(10, 250):
        resized = cv2.resize(template, (avion + 1, avion))
        res = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)

        loc = np.where(res > threshold)
        width, height = resized.shape
        
        for altceva in zip(*loc[::-1]):
            cv2.rectangle(img_gray, altceva, (altceva[0] + width, altceva[1] + height), (255, 0 ,0), 2)

    cv2.imshow("Toate", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scaler(img_gray, template, threshold = 0.533, MinScale = 0.5, MaxScale = 3.5):

    """
    Peform template matching with a touch of rescaling
    
    Threshold = between 0 and 1, just play with it
   
    MinScale  = the minimum value of the rescaling percentage. Make it smaller than 1 if you want to detect objects bigger than the one in the template.
   
    MaxScale  = the maximum value of the rescaling percentage. Make it bigger than 1 if you want to detect objects smaller than the one in the template.  
    """
    # img_gray = cv2.Canny(img_gray, 50, 200)
    # template = cv2.Canny(template, 50, 200)
    width, height = template.shape
    for scale in np.linspace(MinScale, MaxScale)[::-1]:
        resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale))
        r = img_gray.shape[1] / float(resized.shape[1])

        if resized.shape[0] < height or resized.shape[1] < width:
            break

        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

        if maxVal > threshold:
            gasit = (maxVal, maxLoc, r)
            _, maxLoc, r = gasit
            (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
            (endX, endY) = (int((maxLoc[0] + width) * r), int((maxLoc[1] + height) * r))
            cv2.rectangle(img_gray, (startX, startY), (endX, endY), (0, 0, 255), 2)

    cv2.imshow("CFR", img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_rgb = cv2.imread('SourceIMG.jpeg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('TemplateIMG.jpeg', 0)

    # detect_one(img_gray, template)
    # detect_multe(img_gray, template, 0.5)
    # detect_toate(img_gray, template, .67)
    scaler(img_gray, template)