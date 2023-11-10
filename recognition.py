import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

import warnings
warnings.filterwarnings(action='ignore')

IMG_PATH = './images/osmo8.jpg'

img = cv2.imread(IMG_PATH)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_blur_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=2
)

contours, _ = cv2.findContours(
    img_blur_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)


contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })
    
    

MIN_AREA = (img.shape[0] * img.shape[0] * 0.00164)

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    
    if area > MIN_AREA:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)


img_origin = cv2.imread(IMG_PATH)

img_list = []

for i in range(len(possible_contours)):
    img_list.append(img_origin[possible_contours[i]['y'] : possible_contours[i]['y'] + possible_contours[i]['h'], \
        possible_contours[i]['x'] : possible_contours[i]['x'] + possible_contours[i]['w']])
    
    
size_list = []
for i in range(len(img_list)):
    size_list.append(img_list[i].size)

max_index = size_list.index(max(size_list))



def getimgindex(possible_contours, center, img_origin):
    size_list2 = []
    for idx, val in enumerate(possible_contours):
        if (abs(val['cy'] - center[0]) < (img_origin.shape[0] / 30)) & (abs(val['cx'] - center[1]) < (img_origin.shape[1] / 30)):
            size_list2.append([img_list[idx].size, idx])
    arrow_idx = min(size_list2)[1]
    return arrow_idx


if (possible_contours[max_index]['w'] / possible_contours[max_index]['h'] > 2) & (possible_contours[max_index]['w'] / possible_contours[max_index]['h'] < 3):
    # 1개짜리
    
    # 높이 나누기
    heightperblock = possible_contours[max_index]['h']
    middleofheight = heightperblock / 2
    
    # 화살표 좌표 구하기
    arrowwidthratio = 0.66
    middlewidtharrow = possible_contours[max_index]['w'] * arrowwidthratio
    arrow_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidtharrow)
    
    # 화살표 이미지 인덱스 구하기
    arrow_1_idx = getimgindex(possible_contours, arrow_1_center, img_origin)
    
    # 숫자 좌표 구하기
    numberwidthratio = 0.87
    middlewidthnumber = possible_contours[max_index]['w'] * numberwidthratio
    number_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidthnumber)
    
    # 숫자 이미지 인덱스 구하기
    number_1_idx = getimgindex(possible_contours, number_1_center, img_origin)
    
    # 색상 픽셀 구하기
    behavpixelratio = 0.1
    widthpixelbehav = possible_contours[max_index]['w'] * behavpixelratio
    behav_1_pixel = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + widthpixelbehav)
    
    arrow_idx_list = [arrow_1_idx]
    number_idx_list = [number_1_idx]
    action_pixel_list = [behav_1_pixel]
    
elif (possible_contours[max_index]['w'] / possible_contours[max_index]['h'] > 1.2) & (possible_contours[max_index]['w'] / possible_contours[max_index]['h'] < 1.4):
    # 2개짜리
    multiple = 2
    
    # 높이 나누기
    heightperblock = possible_contours[max_index]['h'] / multiple
    middleofheight = heightperblock / 2
    
    # 화살표 좌표 구하기
    arrowwidthratio = 0.66
    middlewidtharrow = possible_contours[max_index]['w'] * arrowwidthratio
    arrow_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidtharrow)
    arrow_2_center = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + middlewidtharrow)
    
    # 화살표 이미지 인덱스 구하기
    arrow_1_idx = getimgindex(possible_contours, arrow_1_center, img_origin)
    arrow_2_idx = getimgindex(possible_contours, arrow_2_center, img_origin)
    
    # 숫자 좌표 구하기
    numberwidthratio = 0.87
    middlewidthnumber = possible_contours[max_index]['w'] * numberwidthratio
    number_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidthnumber)
    number_2_center = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + middlewidthnumber)
    
    # 숫자 이미지 인덱스 구하기
    number_1_idx = getimgindex(possible_contours, number_1_center, img_origin)
    number_2_idx = getimgindex(possible_contours, number_2_center, img_origin)
    
    # 색상 픽셀 구하기
    behavpixelratio = 0.2
    widthpixelbehav = possible_contours[max_index]['w'] * behavpixelratio
    behav_1_pixel = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + widthpixelbehav)
    behav_2_pixel = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + widthpixelbehav)
    
    arrow_idx_list = [arrow_1_idx, arrow_2_idx]
    number_idx_list = [number_1_idx, number_2_idx]
    action_pixel_list = [behav_1_pixel, behav_2_pixel]
    
elif (possible_contours[max_index]['w'] / possible_contours[max_index]['h'] > 0.8) & (possible_contours[max_index]['w'] / possible_contours[max_index]['h'] < 1.1):
    # 3개짜리
    multiple = 3
    
    # 높이 나누기
    heightperblock = possible_contours[max_index]['h'] / multiple
    middleofheight = heightperblock / 2
    
    # 화살표 좌표 구하기
    arrowwidthratio = 0.66
    middlewidtharrow = possible_contours[max_index]['w'] * arrowwidthratio
    arrow_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidtharrow)
    arrow_2_center = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + middlewidtharrow)
    arrow_3_center = int(possible_contours[max_index]['y'] + middleofheight*5), int(possible_contours[max_index]['x'] + middlewidtharrow)
    
    # 화살표 이미지 인덱스 구하기
    arrow_1_idx = getimgindex(possible_contours, arrow_1_center, img_origin)
    arrow_2_idx = getimgindex(possible_contours, arrow_2_center, img_origin)
    arrow_3_idx = getimgindex(possible_contours, arrow_3_center, img_origin)
    
    # 숫자 좌표 구하기
    numberwidthratio = 0.87
    middlewidthnumber = possible_contours[max_index]['w'] * numberwidthratio
    number_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidthnumber)
    number_2_center = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + middlewidthnumber)
    number_3_center = int(possible_contours[max_index]['y'] + middleofheight*5), int(possible_contours[max_index]['x'] + middlewidthnumber)
    
    # 숫자 이미지 인덱스 구하기
    number_1_idx = getimgindex(possible_contours, number_1_center, img_origin)
    number_2_idx = getimgindex(possible_contours, number_2_center, img_origin)
    number_3_idx = getimgindex(possible_contours, number_3_center, img_origin)
    
    # 색상 픽셀 구하기
    behavpixelratio = 0.2
    widthpixelbehav = possible_contours[max_index]['w'] * behavpixelratio
    behav_1_pixel = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + widthpixelbehav)
    behav_2_pixel = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + widthpixelbehav)
    behav_3_pixel = int(possible_contours[max_index]['y'] + middleofheight*5), int(possible_contours[max_index]['x'] + widthpixelbehav)
    
    arrow_idx_list = [arrow_1_idx, arrow_2_idx, arrow_3_idx]
    number_idx_list = [number_1_idx, number_2_idx, number_3_idx]
    action_pixel_list = [behav_1_pixel, behav_2_pixel, behav_3_pixel]
    
    
def predict_arrow(arrow_idx_list:list, img_list):
    # 이미지 전처리
    arrow_img_list = []
    for i in arrow_idx_list:
        out = img_list[i].copy()
        out = 255 - out

        # 36 x 36
        output_img_36 = cv2.resize(out, (36, 36), interpolation = cv2.INTER_AREA)

        # 그레이 스케일 및 이진화
        gray_img = cv2.cvtColor(output_img_36, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
        
        arrow_img_list.append(th1)

    # shape / 예측
    X_test = np.array(arrow_img_list)
    input_shape = 1296
    X_test = X_test / 255
    X_test = X_test.reshape(-1, input_shape)
    loaded_model = load_model('./model/osmo_arrow_model_01.h5')

    test_pred = loaded_model.predict(X_test)

    for idx, pred in enumerate(test_pred):
        for i, a in enumerate(pred):
            if a == pred.max():
                if i == 0:
                    print('←')
                elif i == 1:
                    print('↑')
                elif i == 2:
                    print('→')
                elif i == 3:
                    print('↓')
                    
                    
def predict_number(number_idx_list:list, img_list):
    # 이미지 전처리
    number_img_list = []
    for i in number_idx_list:
        out = img_list[i].copy()
        out = 255 - out
        
        # 36 x 36
        output_img_36 = cv2.resize(out, (36, 36), interpolation = cv2.INTER_AREA)

        # 그레이 스케일 및 이진화
        gray_img = cv2.cvtColor(output_img_36, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray_img, 150, 255, cv2.THRESH_BINARY)
        
        number_img_list.append(th1)

    # shape / 예측
    X_test = np.array(number_img_list)
    input_shape = 1296
    X_test = X_test / 255
    X_test = X_test.reshape(-1, input_shape)
    loaded_model = load_model('./model/osmo_number_model_01.h5')

    test_pred = loaded_model.predict(X_test)

    for idx, pred in enumerate(test_pred):
        for i, a in enumerate(pred):
            if a == pred.max():
                if i == 0:
                    print('2')
                elif i == 1:
                    print('3')
                elif i == 2:
                    print('4')
                elif i == 3:
                    print('5')
                    
                    
                    
def predict_action(action_pixel_list, img_origin):
    img_hsv = cv2.cvtColor(img_origin, cv2.COLOR_RGB2HSV)
    for i in action_pixel_list:
        if img_hsv[i][0] < 50:
            print('1칸')
        elif img_hsv[i][0] < 115:
            print('손')
        else:
            print('2칸')
            
            
predict_arrow(arrow_idx_list, img_list)
predict_number(number_idx_list, img_list)
predict_action(action_pixel_list, img_origin)
                
plt.imshow(img_origin)