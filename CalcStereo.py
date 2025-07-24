import cv2
import numpy as np 
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib import mathtext
import os
import math
import pandas as pd
import tqdm
import imgio


''' 設定 '''
side = 2.5     # 中心から左右のカメラまでの距離
dist = 100     # MMAPs-カメラ間距離
focal_length = 5    # カメラの焦点距離
diagonal = 3.6      # horizontal value（カメラの水平方向センサーサイズ）

ikichi = 44    # 二値化処理の閾値（適当に変える）
kernel = 5     # フィルターのカーネルサイズ（適当に変える）

heights = range(40, 61, 5)    # 飛び出し距離の範囲（グラフの横軸になります）

# フォルダ
''' ファイル名の規則
folderBase + 'L（またはR）/{:04}'.format(int(height*10)) + extension
'''
folderBase = "./Sample_Render/render/"
folderAns = "./Sample_Render/result/"

# 拡張子
extension = ".png"


''' コマンドライン引数メモ
引数0つ
・上記のフォルダ名がデフォルトで指定される
引数2つ
・第1引数：入力画像フォルダ（フォルダ名の最後に「/」を忘れずに）
・第2引数：出力画像フォルダ（フォルダ名の最後に「/」を忘れずに）
引数3つ
・第3引数：二値化処理時の閾値
引数4つ
・第4引数：カーネルサイズ（整数値1つ）
'''


def mkdir(directoryPath):
    if not os.path.exists(directoryPath):
        os.makedirs(directoryPath)
    return directoryPath


def imread(filename, verbose=True):
    print(filename)
    image, maxval = imgio.imread(filename, verbose=verbose)
    
    # チャネル数チェック。必要に応じて輝度データに変換
    channelNum = image.shape[2]
    if channelNum==3:
        image = np.dot(image, [0.2125, 0.7154, 0.0721])
    elif channelNum==4:
        image = np.dot(image, [0.2125, 0.7154, 0.0721, 0])
    
    # image = image / maxval
    # image = normalize(image)
    return image

def mathTextConfig(default=False):
    # Configuration of text in matplotlib.pyplot
    #mathtext.FontConstantsBase = mathtext.ComputerModernFontConstants
    fontfamily = 'sans-serif' if default else 'Times New Roman'
    plt.rcParams.update({'mathtext.default': 'default',
                        'mathtext.fontset': 'stix',
                        'font.family': fontfamily,
                        'font.size': 12,
                        'figure.figsize': (3, 3)})

def halfImg(img):
    # 画像サイズ半分にする
    height, width = img.shape[:2]
    size = (int(width * 0.5), int(height * 0.5))
    img = cv2.resize(img, size)
    return img

# return : 16 point position of midair ( type ... np.array )
def getMidairPositions(img, side, dist, height, lr, sr, ikichi=60):
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ブラーを掛ける
    #gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    gray = cv2.medianBlur(gray, kernel)

    # monorize
    mono = cv2.threshold(gray, ikichi, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY)[1]
    
    # if sr=='R':
    #     # フィルタリングや閾値調整など
    #     filterSize = 7  # 中央値フィルタのカーネルサイズ
        
    #     mono = mono = cv2.threshold(gray, ikichi, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY)[1]
    #     mono = cv2.medianBlur(mono, filterSize)
           
    ##########################################
    
    # cv2.imwrite("./tstyomImgs/" + sr + lr + str(height) + ".png", img)

    # transformationn form mono image to color image to write result of labeling
    # print(mono.shape)
    color1 = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    color2 = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    # labeling
    label = cv2.connectedComponentsWithStats(mono)
    
    '''
    # [変更点]画像を回転させてからラベリングさせる。その後データだけ逆回転補正
    monoRot = cv2.rotate(mono, cv2.ROTATE_90_COUNTERCLOCKWISE)
    label = cv2.connectedComponentsWithStats(monoRot)
    for l in label[3]:
        tmp = l[0]
        l[0] = mono.shape[1]-l[1]
        l[1] = tmp
        #l[0] = 0
        #l[1] = 0
    for l in label[2]:
        tmp = l[0]
        l[0] = mono.shape[1]-l[1]-l[3]  # 左上の座標を求めるため検出したボックスのサイズも考慮
        l[1] = tmp
        tmp = l[2]
        l[2] = l[3]
        l[3] = tmp
    '''
        
        


    # sampling object's information each items
    n = label[0] - 1
    data  = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    """
    print("len_data", len(data))
    print("n", n)
    print("data", data)
    """


    data_sort_num = int(len(data) / 4)

    # Sort each column of disk grid by x-axis. 
    '''
    # [変更点]xではなくy-axisでソート
    for i in range(data_sort_num):
        if (i+1)*4 <= len(data):
            data[i*4 : (i+1)*4] = data[i*4 : (i+1)*4][np.argsort(data[i*4 : (i+1)*4][:, 1])]
            center[i*4 : (i+1)*4] = center[i*4 : (i+1)*4][np.argsort(center[i*4 : (i+1)*4][:, 1])]
    '''
    for i in range(data_sort_num):
        if (i+1)*4 <= len(data):
            data[i*4 : (i+1)*4] = data[i*4 : (i+1)*4][np.argsort(data[i*4 : (i+1)*4][:, 0])]
            center[i*4 : (i+1)*4] = center[i*4 : (i+1)*4][np.argsort(center[i*4 : (i+1)*4][:, 0])]

    # position list of midair imag
    midair_lists = center[0:16]
    x_list = [midair[0] for midair in midair_lists]

    # display result of labeling using object's information
    for i in range(n):
        x0 = data[i][0]
        y0 = data[i][1]
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        cv2.rectangle(color1, (x0, y0), (x1, y1), (0, 0, 255))
        cv2.rectangle(color2, (x0, y0), (x1, y1), (0, 0, 255))

        if i < 16:
            cv2.putText(color1, str(i + 1), (x1 - 20, y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            # display Barycentric coodinate of each object with yellow colored string
            cv2.putText(color2, 'X: ' + str(int(center[i][0])), (x1 - 30, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            cv2.putText(color2, 'Y: ' + str(int(center[i][1])), (x1 - 30, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    """
    print('NUM LABLES : {}'.format(n))
    print('SIDE{}_DIST{}_HEIGHT{}'.format(side, dist, height))
    print(x_list)
    """

    # display result
    # cv2.imshow('side{}_dist{}_height{}_1'.format(side, dist, height), color1)
    # cv2.imshow('side{}_dist{}_height{}_2'.format(side, dist, height), color2)

    save_path = resStereoFolder+'/'
    # print(save_path + sr + lr + str(height) + '_color1.png')
    cv2.imwrite(save_path + sr + lr + str(height) + '_color1.png', color1)
    cv2.imwrite(save_path + sr + lr + str(height) + '_color2.png', color2)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return np.array(x_list)

# 最小二乗法
def reg1dim(x, y):
    n = len(x)
    a = ((np.dot(x, y) - y.sum() * x.sum()/n) / ((x**2).sum() - x.sum()**2 / n))
    b = (y.sum() - a * x.sum()) / n
    return a, b

# pixel     : 変換対象のpixel値
# diagonal  : 画像面のサイズ[cm]
# size      : 画像面のサイズ[pixel]
def pix2cm(pixel, diagonal, size):
    return (diagonal / size) * pixel

def plotData(ave_list, std_list, height_list, is_ans):
    height_list = np.array(height_list)

    # 最小二乗法により直線の傾きと切片を求める
    a, b = reg1dim(height_list, ave_list)

    color = 'orangered' if is_ans else 'royalblue'
    label = 'Direct' if is_ans else 'Mid-Air Image'
    fmt = 'x' if is_ans else 'o'
    markersize = 7 if is_ans else 4

    # 視差の平均から求めたmidair-MMAPs'をプロット
    err_upper = ave_list + std_list
    err_lower = ave_list - std_list
    alpha = 0.15
    # plt.errorbar(height_list, ave_list, yerr=std_list, capsize=3, fmt=fmt, markersize=markersize, ecolor=color, markeredgecolor=color, color=color, label=label)
    plt.fill_between(height_list, err_lower, err_upper, color=color, alpha=alpha)
    plt.plot(height_list, ave_list, marker=fmt, markersize=markersize, color=color, label=label)
    plt.plot([0, height_list.max()], [b, a * height_list.max() + b], color=color, linestyle='dashed')

    # write linear equation of both data
    b_sign = '+' if b >= 0 else '-'
    # plt.text(120, 75 if is_ans else 25, "y = {:.4f}x{}{:.4f}".format(a,b_sign, abs(b)), size=15, color=color)
    plt.text(220, 200 if is_ans else 180, "$\it y = {:.4f}\it x{}{:.4f}$".format(a,b_sign, abs(b)), size=15, color=color)
    print("plotData y plot")
    print("y = {:.4f}x{}{:.4f}".format(a,b_sign, abs(b)))
    print("plotData done")

# Camera to Objectの平均と標準偏差を返す．
def getAverageDistance(l_poses, r_poses, img_size, side, focal_length, disp2mmaps):
    disparities = l_poses - r_poses
    size = img_size[1]
    global diagonal

    # print(disparities)

    # focal length
    # this value is focal length when horizontal fov set to 76 degree
    h = side * 2.0
    
    sizes = np.full(disparities.shape, size)
    diagonals = np.full(disparities.shape, diagonal)
    disp_cm = np.array([pix2cm(disp, diag, size) for (disp, diag, size) in zip(disparities, diagonals, sizes)])

    

    # それぞれの視差に対してmidair-MMAPs'を計算
    fh = np.full(disparities.shape, focal_length * h) # 焦点距離 x 2カメラ間距離
    dists = np.full(disparities.shape, disp2mmaps)
    print("averageDisp:", np.average(disparities))
    mid2mmaps = dists - (fh / disp_cm)
    # mid2mmaps = dists - (fh / (h - disp_cm))

    # 平均を計算
    mid2mmaps_ave = np.average(mid2mmaps)
    print("averageMid2Mmaps:", mid2mmaps_ave)

    # 標準偏差を計算
    mid2mmaps_std = np.std(mid2mmaps)

    return mid2mmaps_ave, mid2mmaps_std

def cm2mm(list):
    # print("#"*30)
    # print(list)
    l = [np.multiply(i, 10) for i in list]
    return np.array(l)



# main
if len(sys.argv)>=2:
    folderBase = sys.argv[1]
    folderAns  = sys.argv[2]
    print("new folderBase: ", folderBase)
    resIMGFolder =      folderBase + "resIMG"
    resCSVFolder =      folderBase + "resCSV"
    resStereoFolder =   folderBase + "resStreo"
    mkdir(resIMGFolder)
    mkdir(resCSVFolder)
    mkdir(resStereoFolder)
if len(sys.argv)>=4:
    ikichi = int(sys.argv[3])
if len(sys.argv)>=5:
    kernel = int(sys.argv[4])
    pass
    
# データ格納パス
resIMGFolder =      folderAns + "resIMG"
resCSVFolder =      folderAns + "resCSV"
resStereoFolder =   folderAns + "resStreo"
mkdir(resIMGFolder)
mkdir(resCSVFolder)
mkdir(resStereoFolder)

# MMAPsで結像した像と正解画像のMMAPs-像間距離の平均と標準偏差
# 各 height ごとに算出した値がここに入る．
ave_list = []
std_list = []
#ans_ave_list = []
#ans_std_list = []

for height in tqdm.tqdm(heights):
    print("hight = " + str(height))

    # 画像のパス
    l_img_path =        folderBase + 'L/{:04}'.format(int(height)) + extension
    r_img_path =        folderBase + 'R/{:04}'.format(int(height)) + extension

    print(l_img_path)
    l_img =     cv2.imread(l_img_path)
    r_img =     cv2.imread(r_img_path)
    
    # 左右の空中像部分の値を取得
    l_poses = getMidairPositions(l_img, side, dist, height, 'L', 'S', ikichi)
    r_poses = getMidairPositions(r_img, side, dist, height, 'R', 'S', ikichi)

    # 視差から midair Image to MMAPs の距離の平均と標準偏差を取得
    dist_ave, dist_std = getAverageDistance(l_poses, r_poses, l_img.shape[:2], side, focal_length, dist)
    #dist_ans_ave, dist_ans_std = getAverageDistance(l_ans_poses, r_ans_poses, l_img_ans.shape[:2], side, focal_length, dist)
    
    # プロットするための平均と標準偏差リストを生成
    ave_list.append(dist_ave)
    std_list.append(dist_std)

# リストをnp.ndarrayに変換
ave_list = np.array(ave_list)
std_list = np.array(std_list)

# 単位をcmからmmに変換
ave_list = cm2mm(ave_list)
std_list = cm2mm(std_list)
heights = cm2mm(heights)

# データをプロット
# import pdb; pdb.set_trace()

plt.clf()
plt.rcParams['font.family'] ='Times New Roman'
plotData(ave_list, std_list, heights, False)
print("plot data done")

# グラフの横軸と縦軸に上限を設定
plt.xlim([min(heights), max(heights)])
plt.ylim([min(heights)-1, max(heights)+1])
plt.grid(color="silver", linestyle="dotted", linewidth=1)

mathTextConfig(default = True)
plt.xlabel('$\it{L_a}$ [mm]', fontsize=16)
# plt.xlabel('Distance from Light source to beam splitter [mm]', fontsize=12)
plt.ylabel('$\it{L_b - L_c}$ [mm]', fontsize=16)
# plt.ylabel('Distance from beam splitter to calculated position [mm]', fontsize=12)
plt.legend()
# plt.show()
plt.savefig(resIMGFolder+'/ex1_result.png')

print("img saved")
df = pd.DataFrame(ave_list.reshape(1,ave_list.shape[0]), columns=heights)
df.to_csv(resCSVFolder+"/stereo_mid.csv")
