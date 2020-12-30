import cv2
import numpy as np
import os
import argparse

def LSC(input_image, region_size, min_element_size, ruler, num_iterations, dir):


    # スーパーピクセルセグメンテーションの生成
    height, width, channels= input_image.shape[:3]


    converted = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV_FULL)
    slic = cv2.ximgproc.createSuperpixelLSC(converted,region_size,float(ruler))

    # 画像のスーパーピクセルセグメンテーションを計算
    # 入力画像は,HSVまたはL*a*b*
    slic.iterate(num_iterations)
    slic.enforceLabelConnectivity(min_element_size)
    labels = slic.getLabels()

    # スーパーピクセルセグメンテーションの境界を取得
    contour_mask = slic.getLabelContourMask(False)
    result = input_image.copy()
    result[0 < contour_mask] = (0, 0, 255)

    # セグメンテーション数の取得
    nseg = slic.getNumberOfSuperpixels()

    lbavgimg = np.zeros((height, width, channels), dtype=np.uint8)
    meanimg = np.zeros((nseg, channels+1), dtype=np.float32)

    # セグメンテーション毎のBGR平均値を取得
    for m in range(0, nseg):
        lb = np.zeros((height, width), dtype=np.uint8)
        lb[labels == m] = 255
        bgr = cv2.mean(input_image, lb)
        meanimg[m, :3] = bgr[:3]

    # セグメンテーションBGR平均値を反映。
    for y in range(height):
        for x in range(width):
            lbno = labels[y,x]
            lbavgimg[y,x] = meanimg[lbno, :3].astype(dtype=np.uint8)

    # 画像表示
    # cv2.imshow('LSC result', result)
    # cv2.imshow('LSC contour_mask', contour_mask)
    # cv2.imshow('LSC labels means image', lbavgimg)
    # cv2.waitKey(0)

    cv2.imwrite('%s/region_size=%d,min_size=%d,ruler=0.75,niter=%d.png' % (dir,region_size,min_element_size,num_iterations), lbavgimg)


def SEEDS(input_image, num_superpixels, num_levels, prior, num_iterations, dir):
 
    # スーパーピクセルセグメンテーションの生成
    height, width, channels= input_image.shape[:3]
    # Trueの場合，精度を高めるために各ブロックレベルをダブルで行う
    double_step = False
    # ヒストグラムビンの数
    num_histogram_bins = 5
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels,
            num_levels, prior, num_histogram_bins, double_step)

    # 画像のスーパーピクセルセグメンテーションを計算
    # 入力画像は,HSVまたはL*a*b*
    converted = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV_FULL)
    seeds.iterate(converted, num_iterations)

    # ラベル取得
    labels = seeds.getLabels()

    # スーパーピクセルセグメンテーションの境界を取得
    contour_mask = seeds.getLabelContourMask(False)
    result = input_image.copy()
    result[0 < contour_mask] = (255, 0, 0)

    # セグメンテーション数の取得
    nseg = seeds.getNumberOfSuperpixels()

    lbavgimg = np.zeros((height, width, channels), dtype=np.uint8)
    meanimg = np.zeros((nseg, channels+1), dtype=np.float32)

    # セグメンテーション毎のBGR平均値を取得
    for m in range(0, nseg):
        lb = np.zeros((height, width), dtype=np.uint8)
        lb[labels == m] = 255
        bgr = cv2.mean(input_image, lb)
        #print('m ', m, ' bgr ', bgr)
        meanimg[m, :3] = bgr[:3]

    # セグメンテーションBGR平均値を反映。
    for y in range(height):
        for x in range(width):
            lbno = labels[y,x]
            lbavgimg[y,x] = meanimg[lbno, :3].astype(dtype=np.uint8)

    # 画像表示
    # cv2.imshow('result', result)
    # cv2.imshow('contour_mask', contour_mask)
    # cv2.imshow('labels means image', lbavgimg)
    # cv2.waitKey(0)

    cv2.imwrite('%s/superpixels=%d,levels=%d,prior=%d,niter=%d.png' % (dir,num_superpixels,num_levels,prior,num_iterations), lbavgimg)


# SuperpixelSLIC
def SLIC(input_image, region_size, min_element_size, ruler, num_iterations, dir):

    # スーパーピクセルセグメンテーションのパラメータ
    height, width, channels= input_image.shape[:3]

    algorithms = [
        ('SLIC', cv2.ximgproc.SLIC), 
        ('SLICO', cv2.ximgproc.SLICO), 
        ('MSLIC', cv2.ximgproc.MSLIC) ]

    converted = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV_FULL)
    for alg in algorithms:
        slic = cv2.ximgproc.createSuperpixelSLIC(converted,alg[1],region_size,float(ruler))

        # 画像のスーパーピクセルセグメンテーションを計算
        # 入力画像は,HSVまたはL*a*b*
        slic.iterate(num_iterations)
        slic.enforceLabelConnectivity(min_element_size)

        labels = slic.getLabels()

        # スーパーピクセルセグメンテーションの境界を取得
        contour_mask = slic.getLabelContourMask(False)
        result = input_image.copy()
        result[0 < contour_mask] = (0, 0, 255)

        # セグメンテーション数の取得
        nseg = slic.getNumberOfSuperpixels()

        lbavgimg = np.zeros((height, width, channels), dtype=np.uint8)
        meanimg = np.zeros((nseg, channels+1), dtype=np.float32)

        # セグメンテーション毎のBGR平均値を取得
        for m in range(0, nseg):
            lb = np.zeros((height, width), dtype=np.uint8)
            lb[labels == m] = 255
            bgr = cv2.mean(input_image, lb)
            #print('m ', m, ' bgr ', bgr)
            meanimg[m, :3] = bgr[:3]

        # セグメンテーションBGR平均値を反映。
        for y in range(height):
            for x in range(width):
                lbno = labels[y,x]
                lbavgimg[y,x] = meanimg[lbno, :3].astype(dtype=np.uint8)

        # 画像表示
        # cv2.imshow(alg[0]+' result', result)
        # cv2.imshow(alg[0]+' contour_mask', contour_mask)
        # cv2.imshow(alg[0]+' labels means image', lbavgimg)
        # cv2.waitKey(0)

        cv2.imwrite('%s/%s_region_size=%d,min_size=%d,ruler=%d,niter=%d.png' % (dir,alg[0],region_size,min_element_size,ruler,num_iterations), lbavgimg)

# Mosaic 
def Mosaic(input_image, i, opt):
    ratio = i*0.1
    small = cv2.resize(input_image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    mosaic_image = cv2.resize(small, input_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    cv2.imwrite('%s/%s_mosaic_%d.png' % (opt.dir, opt.input_name[:-4], i), mosaic_image)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='datas/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='super pixel function', required=True)

    opt = parser.parse_args()

    # print(opt.input_name)
    # print(opt.mode)
    input_image = cv2.imread('%s/%s' % (opt.input_dir, opt.input_name))
    
    if opt.mode == 'LSC':
        dir = 'Superpixel/%s/%s' % (opt.input_name[:-4], opt.mode)
        try:
            os.makedirs(dir)
        except OSError:
            pass

        # スーパーピクセルの平均サイズ
        for region_size in range(10, 41, 10):
            # スーパーピクセル最小サイズ
            for min_element_size in range(5, 11, 5):
                # 繰り返し回数
                for num_iterations in range(3, 6, 1):
                    LSC(input_image, region_size, min_element_size, 0.75, num_iterations, dir)

    elif opt.mode == 'SEEDS':
        dir = 'Superpixel/%s/%s' % (opt.input_name[:-4], opt.mode)
        try:
            os.makedirs(dir)
        except OSError:
            pass

        # スーパーピクセルセグメンテーション最大数
        for num_superpixels in range(100, 401, 100):
            # ブロックレベル数（大きいほどセグメンテーションが正確
            for num_levels in range(3, 6, 1):
                # 平滑係数（大きいほど滑らか）
                for prior in range(1, 6, 1):
                    # 繰り返し回数
                    for num_iterations in range(3, 6, 1):
                        SEEDS(input_image, num_superpixels, num_levels, prior, num_iterations, dir)

    elif opt.mode == 'SLIC':
        dir = 'Superpixel/%s/%s' % (opt.input_name[:-4], opt.mode)
        try:
            os.makedirs(dir)
        except OSError:
            pass

        # スーパーピクセルの平均サイズ
        for region_size in range(10, 41, 10):
            # スーパーピクセル最小サイズ
            for min_element_size in range(5, 11, 5):
                # スーパーピクセルの平滑係数
                for ruler in range(10, 51, 10):
                    # 繰り返し回数
                    for num_iterations in range(3, 6, 1):
                        SLIC(input_image, region_size, min_element_size, ruler, num_iterations, dir)
    
    elif opt.mode == 'mosaic':
        opt.dir = 'Mosaic/%s' % opt.input_name[:-4]
        try:
            os.makedirs(opt.dir)
        except OSError:
            pass

        for i in range (1, 10):
            Mosaic(input_image, i, opt)