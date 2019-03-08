# USAGE
# python final.py --image images_detect/lebron_james.jpg --east frozen_east_text_detection.pb
import cv2
import numpy as np
import sys
import os.path
from imutils.object_detection import non_max_suppression
import argparse
import time

from matplotlib import pyplot 
from keras.models import load_model
from keras.models import model_from_json
chars= ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    help="path to input image")
ap.add_argument("-east", "--east", type=str,
    help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
    help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
    help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
    help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

DEBUG = 1
def ii(xx, yy):
    global img, img_y, img_x
    if yy >= img_y or xx >= img_x:
        return 0
    pixel = img[yy][xx]
    return 0.30 * pixel[2] + 0.59 * pixel[1] + 0.11 * pixel[0]

def connected(contour):
    first = contour[0][0]
    last = contour[len(contour) - 1][0]
    return abs(first[0] - last[0]) <= 1 and abs(first[1] - last[1]) <= 1

def c(index):
    global contours
    return contours[index]

def count_children(index, h_, contour):
    if h_[index][2] < 0:
        return 0
    else:
        if keep(c(h_[index][2])):
            count = 1
        else:
            count = 0
        count += count_siblings(h_[index][2], h_, contour, True)
        return count


def is_child(index, h_):
    return get_parent(index, h_) > 0

def get_parent(index, h_):
    parent = h_[index][3]
    while not keep(c(parent)) and parent > 0:
        parent = h_[parent][3]

    return parent

def count_siblings(index, h_, contour, inc_children=False):
    if inc_children:
        count = count_children(index, h_, contour)
    else:
        count = 0

    
    p_ = h_[index][0]
    while p_ > 0:
        if keep(c(p_)):
            count += 1
        if inc_children:
            count += count_children(p_, h_, contour)
        p_ = h_[p_][0]

    
    n = h_[index][1]
    while n > 0:
        if keep(c(n)):
            count += 1
        if inc_children:
            count += count_children(n, h_, contour)
        n = h_[n][1]
    return count

def keep(contour):
    return keep_box(contour) and connected(contour)

def keep_box(contour):
    xx, yy, w_, h_ = cv2.boundingRect(contour)
    w_ *= 1.0
    h_ *= 1.0
    if w_ / h_ < 0.1 or w_ / h_ > 10:
        return False
    if ((w_ * h_) > ((img_x * img_y) / 5)) or ((w_ * h_) < 15):
        return False
    return True


def include_box(index, h_, contour):
    if is_child(index, h_) and count_children(get_parent(index, h_), h_, contour) <= 2:
        return False

    if count_children(index, h_, contour) > 2:
        return False
    return True

orig_img = cv2.imread(args["image"])
cv2.imwrite("original_image.jpg",orig_img)
img = cv2.copyMakeBorder(orig_img, 50, 50, 50, 50, cv2.BORDER_CONSTANT)

img_y = len(img)
img_x = len(img[0])


blue, green, red = cv2.split(img)

blue_edges = cv2.Canny(blue, 200, 250)
green_edges = cv2.Canny(green, 200, 250)
red_edges = cv2.Canny(red, 200, 250)

edges = blue_edges | green_edges | red_edges

contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

hierarchy = hierarchy[0]

if DEBUG:
    processed = edges.copy()
    rejected = edges.copy()

keepers = []

for index_, contour_ in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour_)
    if keep(contour_) and include_box(index_, hierarchy, contour_):
        # It's a winner!
        keepers.append([contour_, [x, y, w, h]])
        if DEBUG:
            cv2.rectangle(processed, (x, y), (x + w, y + h), (100, 100, 100), 1)
            cv2.putText(processed, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
    else:
        if DEBUG:
            cv2.rectangle(rejected, (x, y), (x + w, y + h), (100, 100, 100), 1)
            cv2.putText(rejected, str(index_), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

new_image = edges.copy()
new_image.fill(255)
boxes = []

for index_, (contour_, box) in enumerate(keepers):
    fg_int = 0.0
    for p in contour_:
        fg_int += ii(p[0][0], p[0][1])

    fg_int /= len(contour_)
    x_, y_, width, height = box
    bg_int = \
        [
            # bottom left corner 3 pixels
            ii(x_ - 1, y_ - 1),
            ii(x_ - 1, y_),
            ii(x_, y_ - 1),

            # bottom right corner 3 pixels
            ii(x_ + width + 1, y_ - 1),
            ii(x_ + width, y_ - 1),
            ii(x_ + width + 1, y_),

            # top left corner 3 pixels
            ii(x_ - 1, y_ + height + 1),
            ii(x_ - 1, y_ + height),
            ii(x_, y_ + height + 1),

            # top right corner 3 pixels
            ii(x_ + width + 1, y_ + height + 1),
            ii(x_ + width, y_ + height + 1),
            ii(x_ + width + 1, y_ + height)
        ]

    bg_int = np.median(bg_int)

    if fg_int >= bg_int:
        fg = 255
        bg = 0
    else:
        fg = 0
        bg = 255
    for x in range(x_, x_ + width):
        for y in range(y_, y_ + height):
            if y >= img_y or x >= img_x:
                continue
            if ii(x, y) > fg_int:
                new_image[y][x] = bg
            else:
                new_image[y][x] = fg

new_image = cv2.blur(new_image, (2, 2))
cv2.imshow("preprocessed", new_image)
cv2.imwrite("pre_processing.jpg",new_image)

cv2.waitKey(0)

m = list()
def loading():
    j_file = open('model.json', 'r') #change in case of model_small.json
    j_model = j_file.read()
    model_without_weights = model_from_json(j_model)
    model_without_weights.load_weights('model.h5') #change in case of model_small.h5
    final_model = model_without_weights
    j_file.close()
    return final_model

model = loading()

def preprocessing(imag):
    hei, wi, d = imag.shape
    wi=wi*4
    hei=hei*4
    imag = cv2.resize(imag, dsize=(wi,hei), interpolation=cv2.INTER_AREA)
    gray_convert = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)
    #imagem = cv2.bitwise_not(gray_convert)
    imagem=gray_convert
    kernel = np.ones((5,5), np.uint8)
    retrn,threshold = cv2.threshold(imagem,100,255,cv2.THRESH_BINARY_INV)
    dilation = cv2.dilate(threshold, kernel, iterations=1)
    gaussianblur=cv2.GaussianBlur(dilation,(5,5),0)
    return imag,gaussianblur;

def detection(imag,gsblur):
    ctrs, hier = cv2.findContours(gsblur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    dp = imag.copy()
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(dp,(x-10,y-10),( x + w + 10, y + h + 10 ),(90,0,255),9)
    #pyplot.imshow(dp)
    #pyplot.show()
    return sorted_ctrs;

def recognition(imag,sorted_ctrs):
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = imag[y-10:y+h+10, x-10:x+w+10]
        roi = cv2.resize(roi, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        roi = np.array(roi)
        t = np.copy(roi)
        t = t / 255.0
        t = 1-t
        t = t.reshape(1,784)
        m.append(roi)
        pred = model.predict_classes(t)
        ocr.append(pred)
    return ocr,imag;

image = cv2.imread("pre_processing.jpg")




orig = image.copy()
orig2=orig.copy()
(H, W) = image.shape[:2]

(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

layerNames = [
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"]

print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    (123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
    scoresData = scores[0, 0, y]
    xData0 = geometry[0, 0, y]
    xData1 = geometry[0, 1, y]
    xData2 = geometry[0, 2, y]
    xData3 = geometry[0, 3, y]
    anglesData = geometry[0, 4, y]

    for x in range(0, numCols):
        # if our score does not have sufficient probability, ignore it
        if scoresData[x] < args["min_confidence"]:
            continue

        (offsetX, offsetY) = (x * 4.0, y * 4.0)

        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

boxes = non_max_suppression(np.array(rects), probs=confidences)
i=0

for (startX, startY, endX, endY) in boxes:
    i=i+1
    ocr = list()
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    crop_img = orig[(startY-10):(endY+10), (startX-5):(endX+5)]
    cv2.rectangle(orig2,(startX-10, startY-10), (endX+10, endY+10), (0, 255, 0), 2)
    a=str(i)
    stri=a+".jpg"
    cv2.waitKey(0)
    crop_img = cv2.copyMakeBorder(crop_img, 50, 50, 50, 50,cv2.BORDER_CONSTANT,value=(255,255,255))
    #cv2.imshow("cropped_before",crop_img)
    crop_img,gsblur=preprocessing(crop_img)
    sorted_ctrs=detection(crop_img,gsblur)
    ocr,crop_img=recognition(crop_img,sorted_ctrs)
    answer_charactrs = list()
    fig, axs = pyplot.subplots(nrows=len(sorted_ctrs), sharex=True, figsize=(1,len(sorted_ctrs)),squeeze=False)
    interp = 'bilinear'
    for j in range(len(ocr)):
        answer_charactrs.append(chars[ocr[j][0]]) 
        '''
        for a in axs[j]:
            a.set_title('-------> predicted letter: '+chars[ocr[j][0]], x=2.5,y=0.24)
            a.imshow(m[j], interpolation=interp)
        '''
    '''
    pyplot.show()
    '''
    
    predstring = ''.join(answer_charactrs)
    print("image number=",i)
    print('Predicted String: '+predstring)
    cv2.putText(orig2,predstring,(startX,startY),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,155),2)
    
   


# show the output image
cv2.imshow("Text Detection", orig2)
cv2.imwrite("result_image.jpg",orig2)
cv2.waitKey(0)