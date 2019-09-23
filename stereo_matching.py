import cv2
import numpy as np
import os
import glob

def getCamMatrix(checker_dir_path):
    # cv2.cornerSubPixにおいて精細化を止める条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # オブジェクトの場所を準備する (0,0,0), (1,0,0), (2,0,0), ...., (6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

    # オブジェクトとコーナーの位置を保存する配列を用意
    objpoints = []
    imgpoints = []

    # 画像を取得
    images = glob.glob(os.path.join(checker_dir_path, '*.jpg'))


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

    return ret, mtx, dist, rvecs, tvecs

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

def unDistortion2(img_path, mtx, dist):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_matrix, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # 画像をトリミングする
    x,y,w,h = roi 
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    checker_dir_path = './checker'
    ret, mtx, dist, rvecs, tvecs = getCamMatrix(checker_dir_path)

    # unDistortion1('./checker/left12.jpg', mtx, dist)
    unDistortion2('./checker/left12.jpg', mtx, dist)

main()