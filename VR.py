import math
import sys
import cv2
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget,QApplication,QGraphicsPixmapItem,QGraphicsScene

import numpy as np
import glob
import time
from matplotlib import pyplot as plt

# pic=""#全部变量
# img=""

#实现摄像头实时监控
def show_cam_pics(expose=-5):
    cap=cv2.VideoCapture(1,cv2.CAP_DSHOW)
    if not(cap.isOpened()):
        QMessageBox.information(QWidget(),"ErrorBox","Can't find camera!")
        return

    #拍摄图像宽度
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    #拍摄图像高度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    # cap.set(3,5000)  # width=1920
    # cap.set(4, 5200)  # height=1080
    print(cap.get(5),cap.get(4),cap.get(3))
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # cap.set(3, 1280)  # width=1920
    # cap.set(4, 720)  # height=1080
    #摄像头曝光值
    cap.set(cv2.CAP_PROP_EXPOSURE,expose)
    #摄像头色调
    cap.set(cv2.CAP_PROP_HUE,0)
    # 获取 OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # 对于 webcam 不能采用 get(CV_CAP_PROP_FPS) 方法
    # 而是：
    if int(major_ver) < 3:
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # Number of frames to capture
    num_frames = 120;
    print("Capturing {0} frames".format(num_frames))

    # Start time
    start = time.time()
    # Grab a few frames
    for i in range(0, num_frames):
        ret, frame = cap.read()
    # End time
    end = time.time()

    # Time elapsed
    seconds = end - start
    print("Time taken : {0} seconds".format(seconds))

    # 计算FPS，alculate frames per second
    fps = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps))

    # 释放 video
    # cap.release()



    # frame,picWin=np.array([[[255,0,0],[0,255,0],[0,0,255]],
    #                 [[255,255,0],[255,0,255],[0,255,255]],
    #                 [[255,255,255],[128,128,128],[0,0,0]],],dtype=np.uint8)
    # frame,picWin=cap.read()如果写在外面只能获取第一帧

    # 拍摄图像格式  MJPG 格式是motion jpeg，也就是将视频的每一帧都按照jpg格式压缩了，数据量大大降低。然而还有一个问题是，不知道默认的图像质量是多少，可不可以设置。
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    while(cv2.waitKey(30)!=27):
        # 获取帧
        # cap.read(frame)
        suc, picWin = cap.read()
        # cv2.flip(frame, frame, 0)
        # frame.copyTo(pic)#存入全局变量
        #创建监视画面缩略图，从而能在屏幕完整显示
        # frame.copyTo(picWin)
        h=cv2.CAP_PROP_FRAME_HEIGHT
        w =cv2.CAP_PROP_FRAME_WIDTH

        # picW=cv2.resize(picWin,(w // 2,h // 2),interpolation=cv2.INTER_CUBIC)
        # fScale=0.4
        # dsize=(int(h*fScale),int(w*fScale))
        # left_frame=picWin[0:480,640:1280]
        # right_frame=picWin[0:480,0:640]
        # picWin=np.hstack((left_frame,right_frame))
        #创建窗口输出缩略图像
        cv2.namedWindow("capturing",1)
        # cv2.resize(picWin, dsize)
        cv2.imshow("capturing",picWin)

    #按下esc回收窗口，释放摄像头
    cv2.destroyWindow("capturing")
    cap.release()

# 函数功能:图像去畸变.
# intrinsics:相机内参矩阵
# distortion_ coff: 相机畸变系数
# distort_img: 输入的畸变图像
# undistort_ img: 输出的去畸变图像
# cutrect:裁剪矩形
def undistortion(images,mrx,dist):
    mrx = np.array([[613.17778151,0. ,356.52588194],[  0.,612.75811158,258.99840619],[  0.,0.,1.        ]])
    dist = ( 0.19895288,-0.26079854,-0.0076386, 0.00936797, 0.21936881)
    # #前者表示迭代次数最大停止，后者表示角点位置变化最小值停止
    # criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    # #棋盘的点
    # objp=np.zeros((6*8,3),np.float32)
    # objp[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)
    # #记录像素点
    # objpoints=[]
    # imgpoints=[]
    # images=glob.glob('./chess/*.jpg')
    # # #图片灰度化
    # # for fname in images:
    # #     img=cv2.imread(fname)
    # #     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # #     cv2.imshow('gray',gray)
    # #     while(cv2.waitKey(30)!=27):
    # #         if cv2.getWindowProperty('gray',cv2.WND_PROP_VISIBLE)<=0:
    # #             break
    # # cv2.destroyAllWindows()
    #
    # for fname in images:
    #     img=cv2.imread(fname)
    #     # 灰度化
    #     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #     # 找角点
    #     ret,corners=cv2.findChessboardCorners(gray,(8,6),None)
    #     # 找到角点之后寻找亚像素坐标系
    #     if ret==True:
    #         objpoints.append(objp)
    #         corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    #         imgpoints.append(corners2)
    #
    #         img=cv2.drawChessboardCorners(img,(8,6),corners2,ret)
    #         cv2.imshow('img',img)
    #         while(cv2.waitKey(30)!=27):
    #              if cv2.getWindowProperty('gray',cv2.WND_PROP_VISIBLE)<=0:
    #                 break
    # cv2.destroyAllWindows()
    # #返回结果ret
    # #相机内参mrx
    # #相机畸变dist
    # #旋转扭曲rvecs
    # #平移扭曲tveces
    # ret,mrx,dist,rvecs,tveces=cv2.calibrateCamera(objpoints,imgpoints,(8,6),None,None)
    # for fname in images:
    #     h,w=fname.shape[:2]
    #     #roi裁剪图片
    #     newcameramtx,roi=cv2.getOptimalNewCameraMatrix(mrx,dist,(w,h),1,(w,h))
    #     dst=cv2.undistort(fname,mrx,dist,None,newcameramtx)
    #     x,y,w,h=roi
    #     #得到去畸化
    #     # dst=dst[y:y+h,x:x+w]
    #
    #     # print(x,y,w,h)
    h,w=images.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mrx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(images, mrx, dist, None, newcameramtx)
    x, y, w, h = roi
    # 得到去畸化
    dst=dst[y:y+h,x:x+w]
    # print(x, y, w, h)
    # while (cv2.waitKey(30) != 27):
    #     cv2.imshow('img2', dst)
    # cv2.destroyAllWindows()

    return dst

#图片opencv转Qimage
def show_picked_pic(self,img,width):
    pic=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#转换图像通道
    QtImg=QImage(pic.data,pic.shape[1],pic.shape[0],pic.shape[1]*3,QImage.Format_RGB888)

    myimg=QPixmap.fromImage(QtImg)
    myimg=myimg.scaledToWidth(width-30)#自适应高度

    self.item=QGraphicsPixmapItem(myimg)#创建像素图元
    self.scene=QGraphicsScene()#创建场景
    self.scene.addItem(self.item)
    return self.scene
    # self.graphicsView.setScene(self.scene)
    # self.graphicsView.show()

#图片二值化
def pic_binarization(img):
    #灰度化
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #计算灰度直方图
    rows,cols=img.shape
    histogram=np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            histogram[img[r][c]]+=1
    #找灰度直方图第一个直方图
    maxLoc=np.where(histogram==np.max(histogram))
    firstPeak=maxLoc[0][0]
    #找第二个峰值
    measureDists =np.zeros([256],np.float32)
    for k in range(256):
        measureDists[k]=pow(k-firstPeak,3)*histogram[k]
    maxLoc2=np.where(measureDists==np.max(measureDists))
    secondPeak=maxLoc2[0][0]

    suc,img=cv2.threshold(img,secondPeak-100,255,cv2.THRESH_BINARY)

    return img

#裁剪去黑边
def change_size(img):
    #先统计行和列最大白色像素
    maxrow=0
    maxcol=0
    x = img.shape[0]#高
    y = img.shape[1]#宽

    for i in range(x):
        temp=0
        for j in range(y):
            if img[i][j] == 255:
                temp=temp+1

        if maxrow<=temp:
            maxrow=temp

    for i in range(y):
        temp = 0
        for j in range(x):
            if img[j][i] == 255:
                temp = temp + 1

        if maxcol <= temp:
            maxcol = temp
    #统计长宽高截取白色区域
    #上下左右依次进行
    row1=0;row2=0;col1=0;col2=0
    for i in range(x):
        temp = 0
        for j in range(y):
            if img[i][j] == 255:
                temp = temp + 1
        if temp>=maxrow*0.85:
            row1=i
            break

    for i in range(x-1,-1,-1):
        temp = 0
        for j in range(y):
            if img[i][j] == 255:
                temp = temp + 1
        if temp>=maxrow*0.85:
            row2=i
            break

    for i in range(y):
        temp = 0
        for j in range(x):
            if img[j][i] == 255:
                temp = temp + 1
        if temp>=maxcol*0.85:
            col1=i
            break

    for i in range(y-1,-1,-1):
        temp = 0
        for j in range(x):
            if img[j][i] == 255:
                temp = temp + 1
        if temp>=maxcol*0.85:
            col2=i
            break
    # 平常的图像为RGB三通道，而灰度图本身为单通道，自然不会正确的显示边缘轮廓的颜色，所以要将三幅灰度图叠在一起
    colorimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    cv2.rectangle(colorimg, (col1, row1),(col2, row2),  (255, 0, 255), 2)

    # while (cv2.waitKey(30) != 27):
    #     cv2.imshow('img2', colorimg)
    # cv2.destroyAllWindows()

    resultimg = img[row1:row2,col1:col2]

    # print((col1, row1),(col2, row2))
    #返回彩色图像和需求图像分析
    return colorimg,resultimg,(col1, row1),(col2, row2),
    # image = cv2.imread(read_file, 1)  # 读取图片 image_name应该是变量
    # img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    # b = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    # binary_image = b[1]  # 二值图--具有三通道
    # binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    # # print(binary_image.shape)  # 改为单通道
    #
    # x = binary_image.shape[0]
    # # print("高度x=", x)
    # y = binary_image.shape[1]
    # # print("宽度y=", y)
    # edges_x = []
    # edges_y = []
    # for i in range(x):
    #     for j in range(y):
    #         if binary_image[i][j] == 255:
    #             edges_x.append(i)
    #             edges_y.append(j)
    #
    # left = min(edges_x)  # 左边界
    # right = max(edges_x)  # 右边界
    # width = right - left  # 宽度
    # bottom = min(edges_y)  # 底部
    # top = max(edges_y)  # 顶部
    # height = top - bottom  # 高度
    #
    # test=image.copy()
    # pre1_picture = image[left:left + width, bottom:bottom + height] # 图片截取
    # cv2.rectangle(test,(left,bottom),(left + width,bottom + height),(0,255,0), 2)
    # while(cv2.waitKey(30)!=27):
    #     cv2.imshow('img', test)
    #
    # cv2.destroyAllWindows()
    # while (cv2.waitKey(30) != 27):
    #     cv2.imshow('img2', pre1_picture)
    #
    # cv2.destroyAllWindows()
    # return pre1_picture  # 返回图片数据

#亮度50%来分界
def FOV_detection(img,pt1,pt4,h,w,a,b,c,d,a2,b2,c2,d2,expose):
    pt0=(int((pt1[0]+pt4[0])/2), int((pt1[1]+pt4[1])/2))
    Lum_standard=calculate_luminance(img[pt0[0]][pt0[1]][2],img[pt0[0]][pt0[1]][1],img[pt0[0]][pt0[1]][0],a,b,c,d,a2,b2,c2,d2,expose)
#测试位置
def position_dection(self,img):
    #分割画面
    left_img=img[:,0:640,:]
    right_img=img[:,640:1280,:]
    #灰度化
    left_img=cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
    right_img=cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)
    #大津法二值化
    ret,left_bin=cv2.threshold(left_img,0,255,cv2.THRESH_OTSU)
    ret2,right_bin = cv2.threshold(right_img, 0, 255, cv2.THRESH_OTSU)

    Row=right_img.shape[0]#高
    Col = left_img.shape[1]  # 宽

    #左镜头检测
    #检测边框
    sum=0
    for x in range(Row):
        sum=int(left_bin[x,0])+int(left_bin[x,Col-1])
    for y in range(Col):
        sum = int(left_bin[0, y]) + int(left_bin[Row-1, y])
    if sum!=0:
        return 1
    #棋盘格检测,找角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret3,corners=cv2.findChessboardCorners(left_img,(5,3),3)
    if ret3==0:
        return 1
    else:
        corners2 = cv2.cornerSubPix(left_img, corners, (5, 5), (-1, -1), criteria)
        self.imgL = cv2.drawChessboardCorners(left_img, (5, 3), corners2, ret3)

        pL=corners2[7]

        if((pL[0][0]<Col*0.45)|(pL[0][0]>Col*0.55)):
            #左镜头横坐标错误
            return 2
        if ((pL[0][1] < Row * 0.45) | (pL[0][1] >Row * 0.55)):
            # 左镜头纵坐标错误
            return 3

        # 右镜头检测
        # 检测边框
    sum = 0
    for x in range(Row):
        sum = int(right_bin[x, 0]) + int(right_bin[x, Col - 1])
    for y in range(Col):
        sum = int(right_bin[0, y]) + int(right_bin[Row - 1, y])
    if sum != 0:
        return 4
    # 棋盘格检测,找角点

    ret4, corners3 = cv2.findChessboardCorners(right_img, (5, 3), 3)
    if ret4 == 0:
        return 4
    else:
        corners4 = cv2.cornerSubPix(right_img, corners3, (5, 5), (-1, -1), criteria)
        self.imgR = cv2.drawChessboardCorners(right_img, (5, 3), corners4, ret4)

        pR = corners4[7]
        if ((pR[0][0] < Col * 0.45) | (pR[0][0]> Col * 0.55)):
            # 右镜头横坐标错误
            return 5
        if ((pR[0][1] < Row * 0.45) | (pR[0][1]> Row * 0.55)):
                # 右镜头纵坐标错误
            return 6

    return 0

#九点位置寻找
def get_nine_corners(img,img2):
    # 前者表示迭代次数最大停止，后者表示角点位置变化最小值停止
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    # 记录像素点
    imgpoints = []
    images=img.copy()
    images2=img2.copy()
#图一
    # 灰度化
    gray=cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
    # 找角点
    corners=cv2.goodFeaturesToTrack(gray,7,0.0000001,30,blockSize=6,k=0.04)
    # corners = np.int0(corners)

    # 找到角点之后寻找亚像素坐标系
    cornerssub=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
    # print(cornerssub)

    # 转化并画图
    cornersd = np.int0(corners)
    for i in cornersd:
        x, y = i.ravel()
        cv2.circle(images, (x, y), 5, (0,0,255), -1)

    # while(cv2.waitKey(30)!=27):
    #     cv2.imshow('img', images)

# #图二
    gray2 = cv2.cvtColor(images2, cv2.COLOR_BGR2GRAY)
    # 找角点
    corners2=cv2.goodFeaturesToTrack(gray2,7,0.0000001,30,blockSize=6,k=0.04)
    # 找到角点之后寻找亚像素坐标系
    cornerssub2=cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)

    #转化并画图
    cornersd2 = np.int0(corners2)
    for i in cornersd2:
        x, y = i.ravel()
        cv2.circle(images2, (x, y), 5, (255,0,255), -1)

    # while(cv2.waitKey(30)!=27):
    #     cv2.imshow('img', images2)

   #合并角点
    for i in cornerssub:
        k=0
        for j in cornerssub2:
            if -2<(i-j)[0][0]<2 and -2<(i-j)[0][1]<2:

                k=1
                break
        if(k==0):
            imgpoints.append(list(i[0]))
    for i in cornerssub2:
        imgpoints.append(list(i[0]))

    #对角点排序，先根据y轴排序，再根据x轴3个元素来排序
    imgpoints = sorted(imgpoints, key=lambda tup: tup[1])
    for i in range(0,3):
        imgpoints[i*3:i*3+3]=sorted(imgpoints[i*3:i*3+3],key=lambda tup:tup[0])

    # 去重合并
    # imgpoints=np.unique(np.vstack([cornerssub,cornerssub2]),axis=0)
    # sortpoints=[]
    # for i in imgpoints:
    #     for j in i:
    #         sortpoints.append(sum(j))
    # index=np.argsort(sortpoints)#记录下标排序
    # sortpoints=imgpoints.copy()#深拷贝
    # for i in range(0,len(index)):
    #     imgpoints[index[i]]=sortpoints[i]
    # cv2.destroyAllWindows()
    # imgpoints得出角点
    # images角点显示图1 images2角点显示图2

    return imgpoints,images,images2

#将点转化为矩形
#img图像 points为角点 size为方框大小
def turn_points_to_rects(img,points,size):
    rects=[]
    if(len(points)==2):
        rects.append([(int(points[0] - size / 2), int(points[1] - size / 2)),
                      (int(points[0] + size / 2), int(points[1] + size / 2))])

        cv2.rectangle(img,rects[0][0],rects[0][1],(0,0,255),2)
    else:
        for point in points:
            rects.append([(int(point[0]-size/2),int(point[1]-size/2)),(int(point[0]+size/2),int(point[1]+size/2))])

        for i in range(0,len(points)):
            cv2.rectangle(img,rects[i][0],rects[i][1],(0,0,255),2)

    # cv2.rectangle(img,rects[0][0],rects[0][1],(0,0,255),2)
    # rects=(img[int(point[0][0]):int(point[0][0]+size),int(point[0][1]):int(point[0][1]+size)])
    # cv2.imshow('roi',rects)

    # cv2.imshow('img', img)
    # if cv2.waitKey(0) == 27:  # 按Esc 键即可退出
    #     cv2.destroyAllWindows()

    return rects,img


#FOV测量
# pt0:中心点
# aov_h:相机水平视角
# aov_v:相机竖直视角
# pt1:内切矩形左上顶点
# pt4:内切矩形右下顶点
def fov_Ned(pt0,aov_h,aov_v,pt1,pt4):

    M=pt0[0]#图像长度一半
    N=pt0[1]#图像宽度一半
    aov_h=aov_h/180*3.14
    aov_v=aov_v/180*3.14
    fov=[]
    fov.append(math.atan((M-pt1[0])*math.tan(aov_h/2)/M)+math.atan((pt4[0]-M)*math.tan(aov_h/2)/M))
    fov.append(math.atan((N-pt1[1])*math.tan(aov_v/2)/N)+math.atan((pt4[1]-N)*math.tan(aov_v/2)/N))
    fov.append(2*math.acos(math.cos(fov[0]/2)*math.cos(fov[1]/2)))
    #弧度转换为角度
    fov[0]=fov[0]/3.14*180#水平
    fov[1]=fov[1]/3.14*180#竖直
    fov[2]=fov[2]/3.14*180#对角线

    return fov

#亮度测量
# def luminance(img,rect,a,b,c):
#     rects=[]
#     result1=[]
#     # rects = turn_points_to_rects(img, corners, 10)
#     rects=rect.copy()
#
#     # print(rects)
#     # tempimg = img[rects[2][0][1]:rects[2][1][1],rects[2][0][0]:rects[2][1][0]].copy()
#     # cv2.imshow('img', img)
#     # if cv2.waitKey(0) == 27:  # 按Esc 键即可退出
#     #     cv2.destroyAllWindows()
#
#     for i in range(0,len(rects)):
#         tempimg=img[rects[i][0][1]:rects[i][1][1],rects[i][0][0]:rects[i][1][0]].copy()
#         color=cv2.mean(tempimg)
#         # print(color[2], color[1], color[0])
#         templuminance=calculate_luminance(color[2],color[1],color[0],a,b,c)
#
#         result1.append(templuminance)
#
#     result2=calculate_luminance_uniformity(result1)
#
#     return result1,result2
# #亮度计算
# #到时候根据曝光，屏幕调整公式
# def calculate_luminance(para1,para2,para3,a,b,c):
#     D=(299*para1+587*para2+114*para3)/1000
#     Lum=D
#     # Lum = a + b * D + c * (D ** 2)
#     # if(D<50 or D>230):
#     #     return 0
#     # else:
#     #     # Lum=(257*para1+504*para2+98*para3)/1000+16
#     #
#     #     Lum=a+b*D+c*(D**2)
#     return Lum
def luminance(img,rect,a,b,c,d,a2,b2,c2,d2,expose):
    rects=[]
    result1=[]
    # rects = turn_points_to_rects(img, corners, 10)
    rects=rect.copy()

    # print(rects)
    # tempimg = img[rects[2][0][1]:rects[2][1][1],rects[2][0][0]:rects[2][1][0]].copy()
    # cv2.imshow('img', img)
    # if cv2.waitKey(0) == 27:  # 按Esc 键即可退出
    #     cv2.destroyAllWindows()

    for i in range(0,len(rects)):
        tempimg=img[rects[i][0][1]:rects[i][1][1],rects[i][0][0]:rects[i][1][0]].copy()
        color=cv2.mean(tempimg)
        # print(color[2], color[1], color[0])
        templuminance=calculate_luminance(color[2],color[1],color[0],a,b,c,d,a2,b2,c2,d2,expose)

        result1.append(templuminance)

    result2=calculate_luminance_uniformity(result1)

    return result1,result2
#亮度计算
#到时候根据曝光，屏幕调整公式
def calculate_luminance(para1,para2,para3,a,b,c,d,a2,b2,c2,d2,expose):
    D=(299*para1+587*para2+114*para3)/1000
    print("灰度值：",D)
    # Lum=D
    if(expose==-4):
        if( D<=115):
            Lum = a + b * D + c * (D ** 2)+d*(D**3)
        elif(D>115):
            Lum = a2 + b2 * D + c2 * (D ** 2) + d2 * (D ** 3)

    if(expose==-5):
        if(D<=120):
            Lum = a + b * D + c * (D ** 2) + d * (D ** 3)
        elif (D > 120):
            Lum = a2 + b2 * D + c2 * (D ** 2) + d2 * (D ** 3)
    # Lum = a + b * D + c * (D ** 2)
    # if(D<50 or D>230):
    #     return 0
    # else:
    #     # Lum=(257*para1+504*para2+98*para3)/1000+16
    #
    #     Lum=a+b*D+c*(D**2)
    print("亮度值：",Lum)
    print()
    return Lum

#亮度均匀值计算
def calculate_luminance_uniformity(templuminance):
    result=[]
    maxlumin=max(templuminance)
    minlumin=min(templuminance)
    avg=sum(templuminance)/len(templuminance)
    luminance_uniformity='%.4f%%'%((maxlumin-minlumin)/avg*100)
    # print(maxlumin,minlumin,avg,luminance_uniformity)
    result.append(maxlumin)
    result.append(minlumin)
    result.append(avg)
    result.append(luminance_uniformity)

    return result


#色度测量
# def chromaticity(img,is_left_pic,location_L,location_R,expose,color,modeid):
def chromaticity(img,rect,m):
    rects = []
    result1 = []
    rects=rect.copy()
    # if (is_left_pic):
    #     rects = turn_points_to_rects(img, location_L, 10)
    # else:
    #     rects = turn_points_to_rects(img, location_R, 10)
    for i in range(0,len(rects)):

        tempimg = img[rects[i][0][1]:rects[i][1][1],rects[i][0][0]:rects[i][1][0]].copy()

        color = cv2.mean(tempimg)
        print('rgb:',color[2], color[1], color[0])
        tempchromaticity = calculate_chromaticity(color[2], color[1], color[0],m)
        result1.append(tempchromaticity)

    # result2=calculate_chromaticity_uniformity(result1)
    result2=0
    # print('uv值：',result1)
    print()
    # print(result2)
    return result1,result2

#色度计算
#到时候根据曝光，屏幕，颜色调整公式
def calculate_chromaticity(para1,para2,para3,m):
    result=[]
    n=np.squeeze(np.array([para1,para2,para3,1]))
    m=np.squeeze(np.mat(m))
    t=m.dot(n)
    X=t[0,0]
    Y=t[0,1]
    Z=t[0,2]
    # X=(49000*para1+31000*para2+20000*para3)/10000
    # Y=(17697*para1+81240*para2+1063*para3)/10000
    # Z=(1000*para2+99000*para3)/10000
    u=4*X/(X+15*Y+3*Z)
    v=9*Y/(X+15*Y+3*Z)
    #uv值
    # result.append((u,v))
    print("xyz值",X/255,Y/255,Z/255)
    print()
    result.append((X/255,Y/255))
    # result.append(v)

    return result

#色度均匀度计算
def calculate_chromaticity_uniformity(UCScolors):
    result2=[]
    centrecolor=UCScolors[4]
    tempdifference=0
    maxdifference=0
    for i in range(0,len(UCScolors)):
        tempdifference=math.sqrt(math.pow(UCScolors[i][0][0]-centrecolor[0][0],2)+math.pow(UCScolors[i][0][1]-centrecolor[0][1],2))
        if(tempdifference>maxdifference):
            maxdifference=tempdifference

    result2.append(maxdifference)
    return result2

#畸变计算
def distortion(ninepoints,aov_h,aov_v):
    print(ninepoints)
    result=[]
    # # 左上方位角
    # PLU=math.atan((ninepoints[0][1]-ninepoints[4][1])/(ninepoints[0][0]-ninepoints[4][0]))
    # # 上方位角
    # PU = math.atan((ninepoints[1][1] - ninepoints[4][1]) / (ninepoints[1][0] - ninepoints[4][0]))
    # # 右上方位角
    # PRU = math.atan((ninepoints[2][1] - ninepoints[4][1]) / (ninepoints[2][0] - ninepoints[4][0]))
    # #左方位角
    # PL=math.atan((ninepoints[3][1]-ninepoints[4][1])/(ninepoints[3][0]-ninepoints[4][0]))
    # #右方位角
    # PR = math.atan((ninepoints[5][1] - ninepoints[4][1]) / (ninepoints[5][0] - ninepoints[4][0]))
    # #左下方位角
    # PLD=math.atan((ninepoints[6][1]-ninepoints[4][1])/(ninepoints[6][0]-ninepoints[4][0]))
    # #下方位角
    # PD=math.atan((ninepoints[7][1]-ninepoints[4][1])/(ninepoints[7][0]-ninepoints[4][0]))
    # #右下方位角
    # PRD=math.atan((ninepoints[8][1]-ninepoints[4][1])/(ninepoints[8][0]-ninepoints[4][0]))
    #
    # idea_PL=(PLU+2*PL+PLD)/4
    # idea_PR=(PRU+2*PR+PRD)/4
    # idea_PU=(PRU+2*PU+PLU)/4
    # idea_PD=(PRD+2*PD+PLD)/4
    #
    # idea_PLU=np.std([idea_PL,idea_PU])
    # idea_PRU=np.std([idea_PR,idea_PU])
    # idea_PLD=np.std([idea_PL,idea_PD])
    # idea_PRD = np.std([idea_PR, idea_PD])

    #中心点坐标
    aov_h = aov_h / 180 * 3.14
    aov_v = aov_v / 180 * 3.14
    M=ninepoints[4][0]
    N = ninepoints[4][1]
    print(M,N)

    # 左上方位角
    PLU =math.atan(abs(M - ninepoints[0][0]) * math.tan(aov_h/2) / M)
    # 上方位角
    PU = math.atan(abs(M - ninepoints[1][0]) * math.tan(aov_h/2) / M)
    # 右上方位角
    PRU = math.atan(abs(ninepoints[2][0]-M) * math.tan(aov_h/2) / M)
    # 左方位角
    PL = math.atan(abs(M - ninepoints[3][0]) * math.tan(aov_h/2) / M)
    # 右方位角
    PR = math.atan(abs(ninepoints[5][0]-M) * math.tan(aov_h/2) / M)
    # 左下方位角
    PLD = math.atan(abs(M - ninepoints[6][0]) * math.tan(aov_h/2) / M)
    # 下方位角
    PD = math.atan(abs(M - ninepoints[7][0]) * math.tan(aov_h/2) / M)
    # 右下方位角
    PRD = math.atan(abs( ninepoints[8][0]-M) * math.tan(aov_h/2) / M)

    idea_PL = (PLU + 2 * PL + PLD) / 4
    idea_PR = (PRU + 2 * PR + PRD) / 4
    idea_PU = (PRU + 2 * PU + PLU) / 4
    idea_PD = (PRD + 2 * PD + PLD) / 4

    idea_PLU = np.std([idea_PL, idea_PU])
    idea_PRU = np.std([idea_PR, idea_PU])
    idea_PLD = np.std([idea_PL, idea_PD])
    idea_PRD = np.std([idea_PR, idea_PD])

    #垂直角
    #左上垂直角
    FLU = math.atan(abs(N - ninepoints[0][1]) * math.tan(aov_v/2) / N)
    #上垂直角
    FU=math.atan(abs(N - ninepoints[1][1]) * math.tan(aov_v/2) / N)
    #右上垂直角
    FRU = math.atan(abs(N - ninepoints[2][1]) * math.tan(aov_v/2) / N)
    #左垂直角
    FL = math.atan(abs(N - ninepoints[3][1]) * math.tan(aov_v/2) / N)
    #右垂直角
    FR = math.atan(abs( ninepoints[5][1]-N) * math.tan(aov_v/2) / N)
    # 左下垂直角
    FLD = math.atan(abs( ninepoints[6][1]-N) * math.tan(aov_v/2) / N)
    # 下垂直角
    FD =math.atan(abs( ninepoints[7][1]-N) * math.tan(aov_v/2) / N)
    # 右下垂直角
    FRD = math.atan(abs( ninepoints[8][1]-N) * math.tan(aov_v/2) / N)

    idea_FL = (FLU + 2 * FL + FLD) / 4
    idea_FR = (FRU + 2 * FR + FRD) / 4
    idea_FU = (FRU + 2 * FU + FLU) / 4
    idea_FD = (FRD + 2 * FD + FLD) / 4

    idea_FLU = np.std([idea_FL, idea_FU])
    idea_FRU = np.std([idea_FR, idea_FU])
    idea_FLD = np.std([idea_FL, idea_FD])
    idea_FRD = np.std([idea_FR, idea_FD])

    # print(PL,PR,PU,PD)
    # print(idea_PL,idea_PR,idea_PU,idea_PD)
    #
    # print(FL, FR, FU, FD)
    # print(idea_FL, idea_FR, idea_FU, idea_FD)

    # AL=math.pow(math.tan(math.sqrt(math.tan(PL)**2+math.tan(FL)**2)),-1)
    # AR=math.pow(math.tan(math.sqrt(math.tan(PR) ** 2 + math.tan(FR) ** 2)), -1)
    # AU=math.pow(math.tan(math.sqrt(math.tan(PU)**2+math.tan(FU)**2)),-1)
    # AD=math.pow(math.tan(math.sqrt(math.tan(PD) ** 2 + math.tan(FD) ** 2)), -1)
    #
    # idea_AL=math.pow(math.tan(math.sqrt(math.tan(idea_PL)**2+math.tan(idea_FL)**2)),-1)
    # idea_AR =math.pow(math.tan(math.sqrt(math.tan(idea_PR) ** 2 + math.tan(idea_FR) ** 2)), -1)
    # idea_AU =math.pow(math.tan(math.sqrt(math.tan(idea_PU)**2+math.tan(idea_FU)**2)),-1)
    # idea_AD =math.pow(math.tan(math.sqrt(math.tan(idea_PD) ** 2 + math.tan(idea_FD) ** 2)), -1)

    AL = math.atan(math.sqrt(math.tan(PL) ** 2 + math.tan(FL) ** 2))
    AR = math.atan(math.sqrt(math.tan(PR) ** 2 + math.tan(FR) ** 2))
    AU = math.atan(math.sqrt(math.tan(PU) ** 2 + math.tan(FU) ** 2))
    AD = math.atan(math.sqrt(math.tan(PD) ** 2 + math.tan(FD) ** 2))

    idea_AL = math.atan(math.sqrt(math.tan(idea_PL) ** 2 + math.tan(idea_FL) ** 2))
    idea_AR = math.atan(math.sqrt(math.tan(idea_PR) ** 2 + math.tan(idea_FR) ** 2))
    idea_AU = math.atan(math.sqrt(math.tan(idea_PU) ** 2 + math.tan(idea_FU) ** 2))
    idea_AD = math.atan(math.sqrt(math.tan(idea_PD) ** 2 + math.tan(idea_FD) ** 2))

    print("--------------------------------------")
    print(AL, AR, AU, AD)
    print(idea_AL, idea_AR, idea_AU, idea_AD)
    print("--------------------------------------")

    result.append('%.8f%%' % ((AL - idea_AL) / idea_AL * 100))
    result.append('%.8f%%' % ((AR - idea_AR) / idea_AR * 100))
    result.append('%.8f%%' % ((AU - idea_AU) / idea_AU * 100))
    result.append('%.8f%%' % ((AD - idea_AD) / idea_AD * 100))

    ALU=math.atan(math.sqrt(math.tan(PLU)**2+math.tan(FLU)**2))
    ARU = math.atan(math.sqrt(math.tan(PRU) ** 2 + math.tan(FRU) ** 2))
    ALD = math.atan(math.sqrt(math.tan(PLD) ** 2 + math.tan(FLD) ** 2))
    ARD = math.atan(math.sqrt(math.tan(PRD) ** 2 + math.tan(FRD) ** 2))

    idea_ALU=math.atan(math.sqrt(math.tan(idea_PLU)**2+math.tan(idea_FLU)**2))
    idea_ARU = math.atan(math.sqrt(math.tan(idea_PRU) ** 2 + math.tan(idea_FRU) ** 2))
    idea_ALD = math.atan(math.sqrt(math.tan(idea_PLD) ** 2 + math.tan(idea_FLD) ** 2))
    idea_ARD = math.atan(math.sqrt(math.tan(idea_PRD) ** 2 + math.tan(idea_FRD) ** 2))

    print(ALU,ARU,ALD,ARD)
    print(idea_ALU,idea_ARU,idea_ALD,idea_ARD)
    #
    # result.append('%.8f%%'%((ALU-idea_ALU)/idea_ALU*100))
    # result.append('%.8f%%'%((ARU - idea_ARU) / idea_ARU * 100))
    # result.append('%.8f%%'%((ALD - idea_ALD) / idea_ALD * 100))
    # result.append('%.8f%%'%((ARD - idea_ARD) / idea_ARD * 100))
    print(result)
    return result

#畸变计算
def distortion(ninepoints):
    print(ninepoints)
    result=[]
    width=((ninepoints[2][0]-ninepoints[0][0])+(ninepoints[5][0]-ninepoints[3][0])+(ninepoints[8][0]-ninepoints[6][0]))/3
    height = ((ninepoints[6][1] - ninepoints[0][1]) + (ninepoints[7][1] - ninepoints[1][1]) + (
                ninepoints[8][1] - ninepoints[2][1])) / 3
    diagonal1=math.sqrt((ninepoints[8][0]-ninepoints[0][0])**2+(ninepoints[8][1]-ninepoints[0][1])**2)
    diagonal2 = math.sqrt((ninepoints[2][0] - ninepoints[6][0]) ** 2 + (ninepoints[2][1] - ninepoints[6][1]) ** 2)
    dis=1-2*math.sqrt(width**2+height**2)/(diagonal1+diagonal2)
    result.append('%.8f%%'%(dis * 100))
    print(result)
    return result



# #迈克尔对比度
# # def michelson_contrast(templuminance):
# def michelson_contrast(img,rect,a,b,c):
#     # print(type(img))
#     # print(rect)
#     result1=[]
#     result2=[]
#     tempimg = img[rect[0][0][1]:rect[0][1][1],rect[0][0][0]:rect[0][1][0]].copy()
#     # gray = cv2.cvtColor(tempimg, cv2.COLOR_BGR2GRAY)
#     # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
#     # print((minVal, maxVal, minLoc, maxLoc))
#     # cv2.imshow('img', tempimg)
#     # if cv2.waitKey(0) == 27:  # 按Esc 键即可退出
#     #     cv2.destroyAllWindows()
#     for i in tempimg:
#         for j in i:
#             color=j
#             # print("lzq")
#             # print(j)
#             # color=cv2.mean(j)
#             # print(color)
#             templuminance = calculate_luminance(color[2], color[1], color[0], a, b,c)
#             result2.append(templuminance)
#
#     print(result2)
#     maxlumin = max(result2)
#     minlumin = min(result2)
#     contrast='%.8f%%'%((maxlumin-minlumin)/(maxlumin+minlumin)*100)
#
#     result1.append(maxlumin)
#     result1.append(minlumin)
#     result1.append(contrast)
#
#     return result1
#迈克尔对比度
# def michelson_contrast(templuminance):
def michelson_contrast(img,rect,a,b,c,d,a2,b2,c2,d2,expose):
    # print(type(img))
    # print(rect)
    result1=[]
    result2=[]
    tempimg = img[rect[0][0][1]:rect[0][1][1],rect[0][0][0]:rect[0][1][0]].copy()
    # gray = cv2.cvtColor(tempimg, cv2.COLOR_BGR2GRAY)
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # print((minVal, maxVal, minLoc, maxLoc))
    # cv2.imshow('img', tempimg)
    # if cv2.waitKey(0) == 27:  # 按Esc 键即可退出
    #     cv2.destroyAllWindows()
    for i in tempimg:
        for j in i:
            color=j
            # print("lzq")
            # print(j)
            # color=cv2.mean(j)
            # print(color)
            templuminance = calculate_luminance(color[2],color[1],color[0],a,b,c,d,a2,b2,c2,d2,expose)
            result2.append(templuminance)

    print(result2)
    maxlumin = max(result2)
    minlumin = min(result2)
    contrast='%.8f%%'%((maxlumin-minlumin)/(maxlumin+minlumin)*100)

    result1.append(maxlumin)
    result1.append(minlumin)
    result1.append(contrast)

    return result1

if __name__ == "__main__":
    app=QApplication(sys.argv)#一定要写应用依赖

    # show_cam_pics()

    img_obj=cv2.imread('../1.png')
    img_obj2=cv2.imread('../2.png')
    img_obj3=cv2.imread('../5.png')
    img_obj4 = cv2.imread('../7.png')
    x=[[0,0],[5,0],[10,0],[0,5],[5,5],[10,5],[0,10],[5,10],[10,10]]
    distortion(x)
    # undistortion(img_obj4,1,2)
    # change_size(pic_binarization(img_obj3))
    # pic_binarization(img_obj)
    # position_dection(img_obj)
    # points,a,b=get_nine_corners(img_obj3, img_obj4)
    # # print(points)
    # turn_points_to_rects(img_obj3,points,20)
    # distortion(points)
    app.exec_()



