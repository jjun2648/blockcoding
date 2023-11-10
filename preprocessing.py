import cv2


class LoadToContours():
    
    def __init__(self, IMG_PATH):
        self.IMG_PATH = IMG_PATH
    
    
    def _load_img(self):
        # IMG_PATH = './images/osmo17.jpg'

        img = cv2.imread(self.IMG_PATH)
        
        return img
    
    
    def _gray_blur(self):
        img = self._load_img()
        
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
        
        return img_blur_thresh
    
    def get_contours(self):
        img_blur_thresh = self._gray_blur()
        img = self._load_img()
        
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
            
            

        MIN_AREA = (img.shape[0] * img.shape[1] * 0.00164)

        possible_contours = []

        cnt = 0
        for d in contours_dict:
            area = d['w'] * d['h']
            
            if area > MIN_AREA:
                d['idx'] = cnt
                cnt += 1
                possible_contours.append(d)
                
        return possible_contours