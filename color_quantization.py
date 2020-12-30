import cv2
import numpy as np
import os
import argparse


def Quantization(img, i, opt):
    # 画像をそのままk-meansにかけることはできないので、shapeを(ピクセル数, 3(BGR))に変換
    Z = img.reshape((-1, 3))
    # np.float32型に変換
    Z = np.float32(Z)

    # k-meansの終了条件
    # デフォルト値を使用
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # 分割後のグループの数
    K = i
    # k-means処理
    _, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # np.uint8型に変換
    center = np.uint8(center)
    # グループごとにグループ内平均値を割り当て
    res = center[label.flatten()]
    # 元の画像サイズにもどす
    res2 = res.reshape((img.shape))

    # 画像の保存
    cv2.imwrite("%s/%s_quantization_%d.png" % (opt.dir, opt.input_name[:-4], i), res2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='datas/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    opt = parser.parse_args()

    img = cv2.imread('%s/%s' % (opt.input_dir, opt.input_name))

    opt.dir = 'Quantization/%s' % opt.input_name[:-4]
    try:
        os.makedirs(opt.dir)
    except OSError:
        pass

    for i in range (2, 11):
        Quantization(img, i, opt)
   