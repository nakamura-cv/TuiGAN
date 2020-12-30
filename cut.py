import cv2
import numpy as np
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',default='Input/moving-gif-processed/moving-gif_longimg/test')
    parser.add_argument('--input_name', required=True)
    opt = parser.parse_args()


    img = cv2.imread('%s/%s' % (opt.input_dir, opt.input_name))

    dir = 'Input/moving-gif_%s' % opt.input_name[:-4]
    try:
        os.makedirs(dir)
    except OSError:
        pass

    for i in range(9):
        cut_image = img[:,i*128:i*128+128,:]
        cut_image = cv2.resize(cut_image, (250,250))
        cv2.imwrite('%s/moving-gif_%s_%d.png' %(dir, opt.input_name[:-4], i) ,cut_image)
        print(i)    