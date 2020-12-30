import cv2
import numpy as np
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Movies')
    parser.add_argument('--input_name', help='input image name', required=True)
    opt = parser.parse_args()


    # Gifファイルを読み込む
    gif = cv2.VideoCapture('%s/%s' % (opt.input_dir, opt.input_name))
    # スクリーンキャプチャを保存するディレクトリを生成
    dir = 'Input/%s' % opt.input_name[:-4]
    try:
        os.makedirs(dir)
    except OSError:
        pass

    i = 0
    j = 0
    while True:
        is_success, frame = gif.read()
        # ファイルが読み込めなくなったら終了
        if not is_success:
            break
        cv2.imwrite('%s/%s_%d.png' % (dir, opt.input_name[:-4], i), frame)
        # if (i % 3 == 0):
        #     # 画像ファイルに書き出す
        #     cv2.imwrite('%s/%s_%d.png' % (dir, opt.input_name[:-4], j), frame)
        #     j += 1
        i += 1