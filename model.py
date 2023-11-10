import cv2
import numpy as np
from keras.models import load_model


class Predict():
    def __init__(self):
        pass
    

    def predict_arrow(self, arrow_idx_list:list, img_list):
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
                        
                        
    def predict_number(self, number_idx_list:list, img_list):
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
                        
                        
                        
    def predict_action(self, action_pixel_list, img_origin):
        img_hsv = cv2.cvtColor(img_origin, cv2.COLOR_RGB2HSV)
        for i in action_pixel_list:
            if img_hsv[i][0] < 50:
                print('1칸')
            elif img_hsv[i][0] < 115:
                print('손')
            else:
                print('2칸')