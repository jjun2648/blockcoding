import cv2

from preprocessing import LoadToContours


class GetIndex():
    
    def __init__(self, IMG_PATH):
        self.img_origin = cv2.imread(IMG_PATH)
        self.possible_contours = LoadToContours(IMG_PATH).get_contours()
        
    
    def _get_max_index(self):
        # IMG_PATH = './images/osmo17.jpg'

        # img_origin = cv2.imread(self.IMG_PATH)

        img_list = []

        for i in range(len(self.possible_contours)):
            img_list.append(self.img_origin[self.possible_contours[i]['y'] : self.possible_contours[i]['y'] + self.possible_contours[i]['h'], self.possible_contours[i]['x'] : self.possible_contours[i]['x'] + self.possible_contours[i]['w']])
            
            
        size_list = []
        for i in range(len(img_list)):
            size_list.append(img_list[i].size)

        max_index = size_list.index(max(size_list))
        
        return img_list, max_index
    
    
    def _get_img_index(self, center):
        img_list, _ = self._get_max_index()
        size_list2 = []
        
        for idx, val in enumerate(self.possible_contours):
            if (abs(val['cy'] - center[0]) < (self.img_origin.shape[0] / 30)) & (abs(val['cx'] - center[1]) < (self.img_origin.shape[1] / 30)):
                size_list2.append([img_list[idx].size, idx])
        img_idx = min(size_list2)[1]
        
        return img_idx
    
    
    def get_idx_list(self):
        possible_contours = self.possible_contours
        _, max_index = self._get_max_index()
        
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
            arrow_1_idx = self._get_img_index(arrow_1_center)
            
            # 숫자 좌표 구하기
            numberwidthratio = 0.87
            middlewidthnumber = possible_contours[max_index]['w'] * numberwidthratio
            number_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidthnumber)
            
            # 숫자 이미지 인덱스 구하기
            number_1_idx = self._get_img_index(number_1_center)
            
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
            arrow_1_idx = self._get_img_index(arrow_1_center)
            arrow_2_idx = self._get_img_index(arrow_2_center)
            
            # 숫자 좌표 구하기
            numberwidthratio = 0.87
            middlewidthnumber = possible_contours[max_index]['w'] * numberwidthratio
            number_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidthnumber)
            number_2_center = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + middlewidthnumber)
            
            # 숫자 이미지 인덱스 구하기
            number_1_idx = self._get_img_index(number_1_center)
            number_2_idx = self._get_img_index(number_2_center)
            
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
            arrow_1_idx = self._get_img_index(arrow_1_center)
            arrow_2_idx = self._get_img_index(arrow_2_center)
            arrow_3_idx = self._get_img_index(arrow_3_center)
            
            # 숫자 좌표 구하기
            numberwidthratio = 0.87
            middlewidthnumber = possible_contours[max_index]['w'] * numberwidthratio
            number_1_center = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + middlewidthnumber)
            number_2_center = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + middlewidthnumber)
            number_3_center = int(possible_contours[max_index]['y'] + middleofheight*5), int(possible_contours[max_index]['x'] + middlewidthnumber)
            
            # 숫자 이미지 인덱스 구하기
            number_1_idx = self._get_img_index(number_1_center)
            number_2_idx = self._get_img_index(number_2_center)
            number_3_idx = self._get_img_index(number_3_center)
            
            # 색상 픽셀 구하기
            behavpixelratio = 0.2
            widthpixelbehav = possible_contours[max_index]['w'] * behavpixelratio
            behav_1_pixel = int(possible_contours[max_index]['y'] + middleofheight), int(possible_contours[max_index]['x'] + widthpixelbehav)
            behav_2_pixel = int(possible_contours[max_index]['y'] + middleofheight*3), int(possible_contours[max_index]['x'] + widthpixelbehav)
            behav_3_pixel = int(possible_contours[max_index]['y'] + middleofheight*5), int(possible_contours[max_index]['x'] + widthpixelbehav)
            
            arrow_idx_list = [arrow_1_idx, arrow_2_idx, arrow_3_idx]
            number_idx_list = [number_1_idx, number_2_idx, number_3_idx]
            action_pixel_list = [behav_1_pixel, behav_2_pixel, behav_3_pixel]
            
        return arrow_idx_list, number_idx_list, action_pixel_list