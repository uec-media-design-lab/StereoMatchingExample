import cv2
import numpy as np
import os
import glob

def getCamMatrix(checker_dir_path, side = 'left'):
    # cv2.cornerSubPixにおいて精細化を止める条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # オブジェクトの場所を準備する (0,0,0), (1,0,0), (2,0,0), ...., (6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

    # オブジェクトとコーナーの位置を保存する配列を用意
    objpoints = []
    imgpoints = []

    # 画像を取得
    images = glob.glob(os.path.join(checker_dir_path, side+'*.jpg'))


    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # チェスボードのコーナーを取得
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

        # チェスボードが発見された場合
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    
    # カメラ行列，レンズ歪みパラメータ，回転・並進ベクトルを取得
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints

def unDistortion1(img_path, mtx, dist):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # 歪み補正
    dst = cv2.undistort(img, mtx, dist, None, new_camera_matrix)

    # 画像をトリミングする
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    # cv2.imwrite('calibresult.png', dst)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dst

# 平行化と歪み補正を行う
def unDistortion2(img_path, R, P, mtx, dist):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, R, P, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # 画像をトリミングする
    x,y,w,h = roi 
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dst

# 2つのカメラを平行化する
def stereoParallelization(img_l_path, img_r_path, obj_points, img_points1, img_points2, mtx_l, dist_l, mtx_r, dist_r):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_size = img_l.shape[:2][::-1]
    # 左右のカメラの回転、並進ベクトルなどの外部パラメータを取得
    retval, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(obj_points, img_points1, img_points2, img_size, mtx_l, dist_l, mtx_r, dist_r, criteria)

    # 平行化のための回転行列・射影行列、Qを求める
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T)

    # 回転行列、射影行列から平行化と歪み補正を行う
    undist_img_l = unDistortion2(img_l_path, R, P1, mtx_l, dist_l)
    undist_img_r = unDistortion2(img_r_path, R, P2, mtx_r, dist_r)

    return undist_img_l, undist_img_r

def stereoMatching(img_l, img_r):
    window_size = 3
    # SGBM法
    stereo = cv2.StereoSGBM_create(
        minDisparity = 0,
        numDisparities = 32,
        blockSize = 15,
        P1 = 8 * 3 * window_size ** 2,
        P2 = 32 * 3 * window_size ** 2,
        disp12MaxDiff = 1,
        uniquenessRatio = 15,
        speckleWindowSize = 0,
        speckleRange = 2,
        preFilterCap = 31,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )

    # 視差を計算
    disparity = stereo.compute(img_l, img_r)
    # 視差を正規化
    disparity = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('disparity', disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return disparity

def main():
    checker_dir_path = './checker'
    # 各カメラの内部パラメータを求める
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l, obj_points_l, img_points_l = getCamMatrix(checker_dir_path, 'left')
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r, obj_points_r, img_points_r = getCamMatrix(checker_dir_path, 'right')

    img_l_path = 'left.jpg'
    img_r_path = 'right.jpg'
    
    # 平行化
    img_l, img_r = stereoParallelization(img_l, img_r, obj_points_l, img_points_l, img_points_r, mtx_l, dist_l, mtx_r, dist_r)

    # 視差を求める
    disparity = stereoMatching(img_l, img_r)

main()