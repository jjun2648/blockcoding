import cv2
import click

from getindex import GetIndex
from model import Predict



@click.command()
@click.option('-i', '--img-name', type = click.STRING, default = 'osmo_test_3', help = '이미지 파일 이름')
def startmain(img_name):
    
    IMG_PATH = './images/' + img_name + '.jpg'
    
    print(IMG_PATH)
    
    img = cv2.imread(IMG_PATH)
    
    arrow_idx_list, number_idx_list, action_pixel_list = GetIndex(IMG_PATH).get_idx_list()
    
    img_list, _ = GetIndex(IMG_PATH)._get_max_index()
    
    Predict().predict_arrow(arrow_idx_list, img_list)
    Predict().predict_number(number_idx_list, img_list)
    Predict().predict_action(action_pixel_list, img)



if __name__ == '__main__':
    startmain()