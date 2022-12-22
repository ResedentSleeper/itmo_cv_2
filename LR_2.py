import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from itertools import cycle


methods = [
    'cv.TM_CCOEFF', 
    'cv.TM_CCOEFF_NORMED', 
    'cv.TM_CCORR',
    'cv.TM_CCORR_NORMED', 
    'cv.TM_SQDIFF', 
    'cv.TM_SQDIFF_NORMED'
]
cycle_methods = cycle(methods)


def next_method():
    return next(cycle_methods)


def get_tmpl_res(img_name, method_name):
    file1 = 'imgs/photo_' + img_name
    file2 = 'imgs/template_' + img_name

    img = cv.imread(file1, 0)
    template = cv.imread(file2, 0)

    w, h = template.shape[::-1]

    method = eval(method_name)
    res = cv.matchTemplate(img,template, method)
    
    _, _, min_loc, max_loc = cv.minMaxLoc(res)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(img,top_left, bottom_right, 255, 2)

    return res, img


def drawMatches(img1, kp1, img2, kp2, matches):

    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1])
    out[:rows2,cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv.circle(out, (int(x1),int(y1)), 4, (255, 0, 0, 1), 1)   
        cv.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0, 1), 1)
        cv.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0, 1), 1)

    return out


def get_sift_res(img):
    file1 = 'imgs/photo_' + img
    file2 = 'imgs/template_' + img
    img1 = cv.imread(file1)
    img2 = cv.imread(file2)

    sift = cv.SIFT_create()

    gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    bf = cv.BFMatcher()
    matches = bf.match(des1,des2)
    matches = sorted(matches, key=lambda val: val.distance)
    img3 = drawMatches(img1,kp1,img2,kp2,matches[:25])
    
    return img3


imgs = {
    '1.jpg': 'Image fragment',
    '2.jpg': 'Image Stretch',
    '3.jpg': 'Glare',
    '4.jpg': 'Angle rotation',
    '5.jpg': 'Noise',
    '6.jpg': 'Negative',
    '7.jpg': 'Decrease contrast and brightness',
    '8.jpg': 'Image Distortion',
    '9.jpg': 'Perspective change (horizontal)',
    '10.jpg': 'Perspective change (vertical)'
}

for img, desc in imgs.items():
    meth_name = next_method()
    
    sift_res = get_sift_res(img)
    tmpl_res, image = get_tmpl_res(img, meth_name)

    fig = plt.figure(figsize=(20, 5))
   
    plt.subplot(132)
    plt.imshow(image,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(sift_res)
    plt.title('SIFT Result'), plt.xticks([]), plt.yticks([])
    
    plt.suptitle(f'Method: {meth_name}\nEffect:{desc}')
    plt.show()