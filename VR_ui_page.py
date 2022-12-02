import copy

import numpy as np

from VR_detection import Ui_VR_detection

import math
import VR
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QMessageBox,QGraphicsPixmapItem,QGraphicsScene
import sys

# import warnings#避免过多错误
# warnings.filterwarnings("ignore")
#标定数据参数(最先完成)
#畸变参数 mrx内参矩阵 dist畸变系数
mrxl=np.array([[613.17778151,0. ,356.52588194],[  0.,612.75811158,258.99840619],[  0.,0.,1.        ]])
mrxr=np.array([[613.17778151,0. ,356.52588194],[  0.,612.75811158,258.99840619],[  0.,0.,1.        ]])
distl = (0.19895288, -0.26079854, -0.0076386, 0.00936797, 0.21936881)
distr = (0.19895288, -0.26079854, -0.0076386, 0.00936797, 0.21936881)
# distl=[[-0.103343,0.284581,-1.39716,2.18507]]
# distr=[[-0.0858416,0.0126673,0.02139,-0.178977]]

#FOV参数 AOVh水平视角 AOVv竖直视角
AOVhl=83.80
AOVhr=41.62
AOVvl=54.30
AOVvr=41.28

#亮度参数 abc参数
a_4_l_big=-310.4355649744981
b_4_l_big=6.160838583871758
c_4_l_big=-0.03765782911015589
d_4_l_big=8.016941493761997e-05
a_4_l_small=-0.2148218818179798
b_4_l_small=0.35272086836004063
c_4_l_small=-0.003482960386730112
d_4_l_small= 2.000105401551479e-05
a_4_r_big=0
b_4_r_big=0
c_4_r_big=0
d_4_r_big=0
a_4_r_small=0
b_4_r_small=0
c_4_r_small=0
d_4_r_small=0

a_5_l_big=-701.0570267069038
b_5_l_big=13.699431522201802
c_5_l_big=-0.08290481335084757
d_5_l_big=0.00017392745893538
a_5_l_small=-0.5273567121390397
b_5_l_small=0.7935747758419579
c_5_l_small=-0.007993102840073018
d_5_l_small=4.2611087187514304e-05
a_5_r_big=0
b_5_r_big=0
c_5_r_big=0
d_5_r_big=0
a_5_r_small=0
b_5_r_small=0
c_5_r_small=0
d_5_r_small=0


#色度参数 m矩阵
chromm_4_red_l=np.array([[-0.27870,0.89013,0,9.96701],[-0.15831,0.53679,0,4.32853],[0.10075,0.04349,0,-0.32034]])
chromm_4_green_l=np.array([[-92.18233,0.12933,0.21052,9.18109],[-202.68947,0.30647,0.02588,18.40960],[18.82986,-0.12853,1.32359,14.20706]])
chromm_4_blue_l=np.array([[0,-0.01519,0.10904,8.66747],[0,0.14145,-0.04023,11.15544],[0,-0.44980,0.72396,31.29144]])
chromm_4_red_r=None
chromm_4_green_r=None
chromm_4_blue_r=None



# 函数功能:图像去畸变.
# images:输入畸变图像
# eye: 左右眼
# dst: 输出的去畸变图像
def undistortion(images,eye):
    if(eye=='l'):
        mrx=mrxl
        dist=distl
    else:
        mrx=mrxr
        dist=distr

    h,w=images.shape[:2]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mrx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(images, mrx, dist, None, newcameramtx)
    x, y, w, h = roi
    # 得到去畸化
    dst=dst[y:y+h,x:x+w]

    return dst

#图片opencv转Qimage
#img:输入图像
#width：图像宽度
#scene：返回图像
# def show_picked_pic(img,width):
#     pic=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#转换图像通道
#     QtImg=QImage(pic.data,pic.shape[1],pic.shape[0],pic.shape[1]*3,QImage.Format_RGB888)
#
#     myimg=QPixmap.fromImage(QtImg)
#     myimg=myimg.scaledToWidth(width-100)#自适应高度
#
#     item=QGraphicsPixmapItem(myimg)#创建像素图元
#     scene=QGraphicsScene()#创建场景
#     scene.addItem(item)
#     return scene


#图片二值化
#img：输入图片
#img：输出二值化图像
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

    suc,img=cv2.threshold(img,secondPeak-50,255,cv2.THRESH_BINARY)

    return img

#裁剪去黑边
#img输入图像
#colorimg裁剪后带有裁剪框图像
# resultimg裁剪区域图像
# (col2, row2)左上角点坐标
# (col1, row1)右下角点坐标
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

    for i in range(x-1,-1,-1):
        temp = 0
        for j in range(y):
            if img[i][j] == 255:
                temp = temp + 1
        if temp>=maxrow*0.85:
            row2=i

    for i in range(y):
        temp = 0
        for j in range(x):
            if img[j][i] == 255:
                temp = temp + 1
        if temp>=maxcol*0.85:
            col1=i

    for i in range(y-1,-1,-1):
        temp = 0
        for j in range(x):
            if img[j][i] == 255:
                temp = temp + 1
        if temp>=maxcol*0.85:
            col2=i
    # 平常的图像为RGB三通道，而灰度图本身为单通道，自然不会正确的显示边缘轮廓的颜色，所以要将三幅灰度图叠在一起
    colorimg=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.rectangle(colorimg, (col2, row2), (col1, row1), (255, 0, 255), 2)

    resultimg = img[row2:row1,col2:col1]

    #返回彩色图像和需求图像分析
    return colorimg,resultimg,(col2, row2),(col1, row1)


#测试位置
#img：输入图像
#0123456：代表当时状态
def position_dection(img,w):
    #分割画面
    left_img=img[:,0:w,:]
    right_img=img[:,w:2*w,:]
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
        imgL = cv2.drawChessboardCorners(left_img, (5, 3), corners2, ret3)

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
        imgR = cv2.drawChessboardCorners(right_img, (5, 3), corners4, ret4)

        pR = corners4[7]
        if ((pR[0][0] < Col * 0.45) | (pR[0][0]> Col * 0.55)):
            # 右镜头横坐标错误
            return 5
        if ((pR[0][1] < Row * 0.45) | (pR[0][1]> Row * 0.55)):
                # 右镜头纵坐标错误
            return 6

    return 0

#九点位置寻找
#img：拟合九点图一
#img2：拟合九点图二
#imgpoint：返回九点坐标
#images：返回有角点坐标的图一
#images2：返回有角点坐标的图二
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
    # 找到角点之后寻找亚像素坐标系
    cornerssub=cv2.cornerSubPix(gray,corners,(5,5),(-1,-1),criteria)
    # 转化并画图
    cornersd = np.int0(corners)
    for i in cornersd:
        x, y = i.ravel()
        cv2.circle(images, (x, y), 5, (0,0,255), -1)

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

    return imgpoints,images,images2

#将点转化为矩形
#img：输入图像
#points：角点
#size：矩阵方框大小
#rects：返回矩阵坐标
#img：返回有矩阵的图像
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

    return rects,img


#FOV测量
# pt0:中心点
# aov_h:相机水平视角
# aov_v:相机竖直视角
# eye:分辨眼睛
# pt1:内切矩形左上顶点
# pt4:内切矩形右下顶点
# fov：返回视场角数据
def fov_Ned(pt0,eye,pt1,pt4):
    if(eye=='l'):
        aov_h=AOVhl
        aov_v=AOVvl
    else:
        aov_h = AOVhr
        aov_v = AOVvr
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
#img：输入图像
#rect：输入需要计算的矩阵位置
#expose：当时图像的曝光值
#eye：当时计算的眼睛
#到时候根据曝光，屏幕调整公式
def luminance(img,rect,expose,eye):
    rects=[]
    result1=[]
    # rects = turn_points_to_rects(img, corners, 10)
    rects=rect.copy()

    for i in range(0,len(rects)):
        tempimg=img[rects[i][0][1]:rects[i][1][1],rects[i][0][0]:rects[i][1][0]].copy()
        color=cv2.mean(tempimg)
        print(color[2], color[1], color[0])
        templuminance=calculate_luminance(color[2],color[1],color[0],expose,eye)
        result1.append(templuminance)
   #统计均匀度
    result2=calculate_luminance_uniformity(result1)

    return result1,result2

#亮度计算
#para1，para2，para3：分别为RGB三通道值
#expose：当时图像的曝光值
#eye：当时计算的眼睛
#到时候根据曝光，屏幕调整公式
def calculate_luminance(para1,para2,para3,expose,eye):
    D=(299*para1+587*para2+114*para3)/1000
    if (expose == -4 and eye=='l'):
        if (D <= 115):
            Lum = a_4_l_small + b_4_l_small * D + c_4_l_small * (D ** 2) + d_4_l_small * (D ** 3)
        elif (D > 115):
            Lum = a_4_l_big + b_4_l_big * D + c_4_l_big * (D ** 2) + d_4_l_big * (D ** 3)
    elif(expose == -4 and eye=='r'):
        if (D <= 115):
            Lum = a_4_r_small + b_4_r_small * D + c_4_r_small * (D ** 2) + d_4_r_small * (D ** 3)
        elif (D > 115):
            Lum = a_4_r_big + b_4_r_big * D + c_4_r_big * (D ** 2) + d_4_r_big * (D ** 3)

    if (expose == -5 and eye=='l'):
        if (D <= 120):
            Lum = a_5_l_small + b_5_l_small * D + c_5_l_small * (D ** 2) + d_5_l_small * (D ** 3)
        elif (D > 120):
            Lum = a_5_l_big + b_5_l_big * D + c_5_l_big * (D ** 2) + d_5_l_big * (D ** 3)
    elif (expose == -5 and eye == 'r'):
        if (D <= 120):
            Lum = a_5_r_small + b_5_r_small * D + c_5_r_small * (D ** 2) + d_5_r_small * (D ** 3)
        elif (D > 120):
            Lum = a_5_r_big + b_5_r_big * D + c_5_r_big * (D ** 2) + d_5_r_big * (D ** 3)
    return Lum

#亮度均匀值计算
#templuminance：亮度值
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
#img：输入图像
#rect：输入需要计算的矩阵位置
#eye：当时计算的眼睛
#col:根据不同颜色选择不同公式
#到时候根据曝光，屏幕调整公式
def chromaticity(img,rect,eye,col):
    rects = []
    m=None
    result1 = []
    rects=rect.copy()
    if(eye=='l'and col=="Red"):
        m=chromm_4_red_l
    elif(eye=='l'and col=="Green"):
        m=chromm_4_green_l
    elif(eye=='l'and col=="Blue"):
        m=chromm_4_blue_l

    if (eye == 'r' and col == "Red"):
        m = chromm_4_red_r
    elif (eye == 'r' and col == "Green"):
        m = chromm_4_green_r
    elif (eye == 'r' and col == "Blue"):
        m = chromm_4_blue_r

    for i in range(0,len(rects)):
        tempimg = img[rects[i][0][1]:rects[i][1][1],rects[i][0][0]:rects[i][1][0]].copy()
        color = cv2.mean(tempimg)
        tempchromaticity = calculate_chromaticity(color[2], color[1], color[0],m)
        result1.append(tempchromaticity)

    result2=calculate_chromaticity_uniformity(result1)
    print(result1)
    print(result2)
    return result1,result2

#色度计算
#para1，para2，para3：分别为RGB三通道值
#m：颜色矩阵
def calculate_chromaticity(para1,para2,para3,m):
    result=[]
    n=np.squeeze(np.array([para1,para2,para3,1]))
    m=np.squeeze(np.mat(m))
    t=m.dot(n)
    X=t[0,0]
    Y=t[0,1]
    Z=t[0,2]

    u=4*X/(X+15*Y+3*Z)
    v=9*Y/(X+15*Y+3*Z)
    result.append((u,v))
    # result.append(v)

    return result

#色度均匀度计算
#UCScolors:uv坐标值
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
#ninepoints：九点坐标
#eye：判断测量那只眼睛
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

#迈克尔对比度
# def michelson_contrast(templuminance):
def michelson_contrast(img,rect,expose,eye):
    result1=[]
    result2=[]
    tempimg = img[rect[0][0][1]:rect[0][1][1],rect[0][0][0]:rect[0][1][0]].copy()

    for i in tempimg:
        for j in i:
            color=j
            templuminance = calculate_luminance(color[2], color[1], color[0], expose,eye)
            result2.append(templuminance)

    maxlumin = max(result2)
    minlumin = min(result2)
    contrast='%.8f%%'%((maxlumin-minlumin)/(maxlumin+minlumin)*100)

    result1.append(maxlumin)
    result1.append(minlumin)
    result1.append(contrast)

    return result1

#图片opencv转Qimage
#img:输入图像
#width：图像宽度
#scene：返回图像
def show_picked_pic(img,width):
    show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
    showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],show.shape[1]*3,
                             QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式

    showImage = QtGui.QPixmap.fromImage(showImage)
    showImage = showImage.scaledToWidth(width - 80)  # 自适应高度
    item = QGraphicsPixmapItem(showImage)  # 创建像素图元
    scene = QGraphicsScene()  # 创建场景
    scene.addItem(item)

    return scene



#主窗口
class mainWindow(QtWidgets.QWidget,Ui_VR_detection):
    h=480
    w=640
    expose=-4
    # 角点资源
    leftcorners = []
    rightcorners = []
    def __init__(self):
        super(mainWindow,self).__init__()
        #初始化
        self.setupUi(self)
        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture(0 )  # 视频流
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))# 拍摄图像格式  MJPG 格式是motion jpeg，也就是将视频的每一帧都按照jpg格式压缩了，数据量大大降低。
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w * 2)# 拍摄图像宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)# 拍摄图像高度
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.expose)# 摄像头曝光值

        self.CAM_NUM = 1  # 为0时表示视频流来自笔记本内置摄像头
        self.spinBox.valueChanged.connect(self.change_expose)#选择曝光值
        self.spinBox_2.valueChanged.connect(self.change_expose)
        self.spinBox_3.valueChanged.connect(self.change_expose)
        self.spinBox_4.valueChanged.connect(self.change_expose)
        self.spinBox_5.valueChanged.connect(self.change_expose)
        self.spinBox_6.valueChanged.connect(self.change_expose)
        self.spinBox_7.valueChanged.connect(self.change_expose)
        #NED
        self.pushButton.clicked.connect(self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.pushButton_2.clicked.connect(self.NED_capture)
        self.pushButton_3.clicked.connect(self.NED_detection)
        #ninepoint
        self.pushButton_4.clicked.connect(self.button_open_camera_clicked)
        self.pushButton_5.clicked.connect(self.ninepoint_capture1)
        self.pushButton_6.clicked.connect(self.ninepoint_capture2)
        self.pushButton_7.clicked.connect(self.ninepoint)
        # self.pushButton_8.clicked.connect()#保存结果
        #FOV
        self.pushButton_9.clicked.connect(self.button_open_camera_clicked)
        self.pushButton_10.clicked.connect(self.FOV_capture)
        self.pushButton_11.clicked.connect(self.FOV_binarization)
        self.pushButton_12.clicked.connect(self.FOV_cutrect)
        # self.pushButton_13.clicked.connect(self.ninepoint)#保存结果
        #亮度
        self.pushButton_14.clicked.connect(self.button_open_camera_clicked)
        self.pushButton_15.clicked.connect(self.Lum_capture)
        self.pushButton_16.clicked.connect(self.Lum_detection)
        # self.pushButton_17.clicked.connect(self.FOV_binarization)
        #色度
        self.pushButton_18.clicked.connect(self.button_open_camera_clicked)
        self.pushButton_19.clicked.connect(self.Chrom_capture)
        self.pushButton_20.clicked.connect(self.Chrom_detection)
        # self.pushButton_21.clicked.connect(self.FOV_binarization)
        #畸变
        self.pushButton_22.clicked.connect(self.button_open_camera_clicked)
        self.pushButton_23.clicked.connect(self.Distortion_capture1)
        self.pushButton_24.clicked.connect(self.Distortion_capture2)
        self.pushButton_25.clicked.connect(self.Distortion_detection)
        # self.pushButton_26.clicked.connect(self.FOV_binarization)
        #对比度
        self.pushButton_27.clicked.connect(self.button_open_camera_clicked)
        self.pushButton_28.clicked.connect(self.Michelson_capture)
        self.pushButton_29.clicked.connect(self.Michelson_detection)
        # self.pushButton_30.clicked.connect(self.FOV_binarization)
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()



    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确",
                                                        buttons=QtWidgets.QMessageBox.Ok)
            else:

                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                if (self.tabWidget.currentIndex() == 0):
                    self.pushButton.setText('关闭相机')  # 往显示视频的Label里 显示QImage
                elif (self.tabWidget.currentIndex() == 1):
                    self.pushButton_4.setText('关闭相机')
                elif (self.tabWidget.currentIndex() == 2):
                    self.pushButton_9.setText('关闭相机')
                elif (self.tabWidget.currentIndex() == 3):
                    self.pushButton_14.setText('关闭相机')
                elif (self.tabWidget.currentIndex() == 4):
                    self.pushButton_18.setText('关闭相机')
                elif (self.tabWidget.currentIndex() == 5):
                    self.pushButton_22.setText('关闭相机')
                elif (self.tabWidget.currentIndex() == 6):
                    self.pushButton_27.setText('关闭相机')
        else:

            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.graphicsView.setScene(QGraphicsScene())  # 清空视频显示区域
            self.graphicsView_2.setScene(QGraphicsScene())
            self.graphicsView_3.setScene(QGraphicsScene())
            self.graphicsView_4.setScene(QGraphicsScene())
            self.graphicsView_7.setScene(QGraphicsScene())
            self.graphicsView_8.setScene(QGraphicsScene())
            self.graphicsView_11.setScene(QGraphicsScene())
            self.graphicsView_12.setScene(QGraphicsScene())
            self.graphicsView_15.setScene(QGraphicsScene())
            self.graphicsView_16.setScene(QGraphicsScene())
            self.graphicsView_19.setScene(QGraphicsScene())
            self.graphicsView_20.setScene(QGraphicsScene())
            self.graphicsView_23.setScene(QGraphicsScene())
            self.graphicsView_24.setScene(QGraphicsScene())

            if (self.tabWidget.currentIndex() == 0):
                self.pushButton.setText('打开相机')  # 往显示视频的Label里 显示QImage
            elif (self.tabWidget.currentIndex() == 1):
                self.pushButton_4.setText('打开相机')
            elif (self.tabWidget.currentIndex() == 2):
                self.pushButton_9.setText('打开相机')
            elif (self.tabWidget.currentIndex() == 3):
                self.pushButton_14.setText('打开相机')
            elif (self.tabWidget.currentIndex() == 4):
                self.pushButton_18.setText('打开相机')
            elif (self.tabWidget.currentIndex() == 5):
                self.pushButton_22.setText('打开相机')
            elif (self.tabWidget.currentIndex() == 6):
                self.pushButton_27.setText('打开相机')

    def show_camera(self):
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P','G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w * 2)  # 拍摄图像宽度
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)  # 拍摄图像高度
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.expose)  # 摄像头曝光值

        flag, self.image = self.cap.read()  # 从视频流中读取

        #分左右,因为左右眼反了需要重新合并
        self.left_frame = self.image[0:self.h, self.w:self.w*2]
        self.right_frame = self.image[0:self.h, 0:self.w]
        self.image = np.hstack((self.left_frame, self.right_frame))
        # show = cv2.resize(self.image, (self.w, self.h))
        img0=show_picked_pic(self.image,self.graphicsView.width())# 把读到的帧的大小重新设置为 640x480
        img1 = show_picked_pic(self.left_frame, int(self.graphicsView.width()/2))
        img2 = show_picked_pic(self.right_frame, int(self.graphicsView.width()/2))

        if(self.tabWidget.currentIndex()==0):
            self.graphicsView.setScene(img0)  # 往显示视频的Label里 显示QImage
        elif(self.tabWidget.currentIndex()==1):
            self.graphicsView_3.setScene(img1)
            self.graphicsView_4.setScene(img2)
        elif (self.tabWidget.currentIndex() == 2):
            self.graphicsView_7.setScene(img1)
            self.graphicsView_8.setScene(img2)
        elif (self.tabWidget.currentIndex() == 3):
            self.graphicsView_11.setScene(img1)
            self.graphicsView_12.setScene(img2)
        elif (self.tabWidget.currentIndex() == 4):
            self.graphicsView_15.setScene(img1)
            self.graphicsView_16.setScene(img2)
        elif (self.tabWidget.currentIndex() == 5):
            self.graphicsView_19.setScene(img1)
            self.graphicsView_20.setScene(img2)
        elif (self.tabWidget.currentIndex() == 6):
            self.graphicsView_23.setScene(img1)
            self.graphicsView_24.setScene(img2)



    def change_expose(self):

        if (self.tabWidget.currentIndex() == 0):#选择曝光值
            self.expose=self.spinBox.value()
        elif (self.tabWidget.currentIndex() == 1):
            self.expose=self.spinBox_2.value()
        elif (self.tabWidget.currentIndex() == 2):
            self.expose=self.spinBox_3.value()
        elif (self.tabWidget.currentIndex() == 3):
            self.expose=self.spinBox_4.value()
        elif (self.tabWidget.currentIndex() == 4):
            self.expose=self.spinBox_5.value()
        elif (self.tabWidget.currentIndex() == 5):
            self.expose=self.spinBox_6.value()
        elif (self.tabWidget.currentIndex() == 6):
            self.expose=self.spinBox_7.value()


#NED位置测量
    def NED_capture(self):
        self.img_NED = self.image.copy()  # 把读到的帧的大小重新设置为 640x480

        self.graphicsView_2.setScene(show_picked_pic(self.image ,self.graphicsView_2.width()) ) # 往显示视频的Label里 显示QImage
    def NED_detection(self):
        self.textBrowser.clear()
        i=position_dection(self.img_NED,self.w)
        if(i==0):
            self.textBrowser.append("位置正确,允许测量")
        elif(i==1):
            self.textBrowser.append("左镜头位置错误，图片显示不完全")
        elif(i==2):
            self.textBrowser.append( "左镜头横坐标错误")
        elif (i == 3):
            self.textBrowser.append( "左镜头纵坐标错误")
        elif (i == 4):
            self.textBrowser.append( "右镜头位置错误，图片显示不完全")
        elif (i == 5):
            self.textBrowser.append( "右镜头横坐标错误")
        elif (i == 6):
            self.textBrowser.append( "右镜头纵坐标错误")

#九点位置测量
    def ninepoint_capture1(self):
        if(self.radioButton.isChecked()):
            self.img_ninpoint1=self.left_frame.copy()
        elif(self.radioButton_2.isChecked()):
            self.img_ninpoint1=self.right_frame.copy()

        self.graphicsView_5.setScene(show_picked_pic(self.img_ninpoint1, self.graphicsView_5.width()))

    def ninepoint_capture2(self):
        if(self.radioButton.isChecked()):
            self.img_ninpoint2=self.left_frame.copy()
        elif(self.radioButton_2.isChecked()):
            self.img_ninpoint2=self.right_frame.copy()

        self.graphicsView_6.setScene(show_picked_pic(self.img_ninpoint2, self.graphicsView_6.width()))

    def ninepoint(self):
        self.result, self.img_ninpoint1, self.img_ninpoint2 = VR.get_nine_corners(self.img_ninpoint1, self.img_ninpoint2)

        self.graphicsView_5.setScene(show_picked_pic(self.img_ninpoint1, self.graphicsView_5.width()))
        self.graphicsView_6.setScene(show_picked_pic( self.img_ninpoint2, self.graphicsView_6.width()))

        self.textBrowser_2.clear()
        if (self.radioButton.isChecked()):
            self.textBrowser_2.append("左眼角点leftcorners:")
            for i in self.result:
                self.textBrowser_2.append(str(i))
        elif (self.radioButton_2.isChecked()):
            self.textBrowser_2.append("右眼角点rightcorners:")
            for i in self.result:
                self.textBrowser_2.append(str(i))

        if (self.radioButton.isChecked()):
            self.leftcorners=self.result.copy()
        elif (self.radioButton_2.isChecked()):
            self.rightcorners=self.result.copy()

#FOV测量
    #拍摄图片
    def FOV_capture(self):
        if (self.radioButton_3.isChecked()):
            self.img_FOV = self.left_frame.copy()
            # self.aov_h = AOVhl
            # self.aov_v = AOVvl
        elif (self.radioButton_4.isChecked()):
            self.img_FOV = self.right_frame.copy()
            # self.aov_h = AOVhr
            # self.aov_v = AOVvr
        self.graphicsView_9.setScene(show_picked_pic(self.img_FOV, self.graphicsView_9.width()))
    #图片二值化
    def FOV_binarization(self):
        self.img_FOV_cut=pic_binarization(self.img_FOV)
        self.graphicsView_10.setScene(show_picked_pic(self.img_FOV_cut, self.graphicsView_10.width()))

    #矩形截取
    def FOV_cutrect(self):

        self.img_FOV_cut,self.hw,self.pt1,self.pt4=change_size(self.img_FOV_cut)
        h, w = self.img_FOV_cut.shape[0:2]
        self.pt0 = (int(w / 2), int(h / 2))
        self.graphicsView_10.setScene(show_picked_pic(self.img_FOV_cut, self.graphicsView_10.width()))
        if (self.radioButton_3.isChecked()):
            self.result=fov_Ned(self.pt0, 'l', self.pt1, self.pt4)
        elif (self.radioButton_4.isChecked()):
            self.result=fov_Ned(self.pt0, 'r', self.pt1, self.pt4)


        self.textBrowser_3.append("水平FOV为：" + str(self.result[0]))
        self.textBrowser_3.append("竖直FOV为：" + str(self.result[1]))
        self.textBrowser_3.append("对角线FOV为：" + str(self.result[2]))

#亮度测量
    #拍摄照片
    def Lum_capture(self):
        if (self.radioButton_5.isChecked()):
            self.img_Lum = self.left_frame.copy()
            # if(self.spinBox_4.value==-4):
            #     self.a_small=a_4_l_small
            #     self.b_small=b_4_l_small
            #     self.c_small = c_4_l_small
            #     self.d_small = d_4_l_small
            #     self.a_big = a_4_l_big
            #     self.b_big = b_4_l_big
            #     self.c_big = c_4_l_big
            #     self.d_big = d_4_l_big
            # elif (self.spinBox_4.value == -5):
            #     self.a_small = a_5_l_small
            #     self.b_small = b_5_l_small
            #     self.c_small = c_5_l_small
            #     self.d_small = d_5_l_small
            #     self.a_big = a_5_l_big
            #     self.b_big = b_5_l_big
            #     self.c_big = c_5_l_big
            #     self.d_big = d_5_l_big
        elif (self.radioButton_6.isChecked()):
            self.img_Lum = self.right_frame.copy()
            # if (self.spinBox_4.value == -4):
            #     self.a_small = a_4_r_small
            #     self.b_small = b_4_r_small
            #     self.c_small = c_4_r_small
            #     self.d_small = d_4_r_small
            #     self.a_big = a_4_r_big
            #     self.b_big = b_4_r_big
            #     self.c_big = c_4_r_big
            #     self.d_big = d_4_r_big
            # elif (self.spinBox_4.value == -5):
            #     self.a_small = a_5_r_small
            #     self.b_small = b_5_r_small
            #     self.c_small = c_5_r_small
            #     self.d_small = d_5_r_small
            #     self.a_big = a_5_r_big
            #     self.b_big = b_5_r_big
            #     self.c_big = c_5_r_big
            #     self.d_big = d_5_r_big
        self.graphicsView_13.setScene(show_picked_pic(self.img_Lum, self.graphicsView_13.width()))
    #亮度测量
    def Lum_detection(self):

        self.textBrowser_4.clear()
        if (self.radioButton_5.isChecked()):
            if (self.leftcorners==[]):
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '未获取左边角点坐标')
                msg_box.exec_()
                return
            self.rect_Lum,pictem=turn_points_to_rects(self.img_Lum.copy(),self.leftcorners,10)
        elif (self.radioButton_6.isChecked()):
            if (self.rightcorners==[]):
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '未获取右边角点坐标')
                msg_box.exec_()
                return
            self.rect_Lum,pictem= turn_points_to_rects(self.img_Lum.copy(), self.rightcorners, 10)
        # 演示效果
        self.graphicsView_14.setScene(show_picked_pic( pictem, self.graphicsView_14.width()))

        if (self.radioButton_5.isChecked()):
            self.result1, self.result2 = luminance(self.img_Lum, self.rect_Lum, self.expose,'l')

        elif (self.radioButton_6.isChecked()):
            self.result1, self.result2 = luminance(self.img_Lum, self.rect_Lum, self.expose,'r')
        # self.result1, self.result2 = luminance(self.img_Lum, self.rect_Lum, self.a_small, self.b_small, self.c_small, self.d_small, self.a_big,
        #                                           self.b_big, self.c_big
        #                                           , self.d_big, self.expose)
        self.textBrowser_4.append('九点亮度测量值：')  # 未完成部分
        for i in self.result1:
            self.textBrowser_4.append(str(i))
        self.textBrowser_4.append('最大值：' + str(self.result2[0]))
        self.textBrowser_4.append('最小值：' + str(self.result2[1]))
        self.textBrowser_4.append('平均值：' + str(self.result2[2]))
        self.textBrowser_4.append('不均匀值：' + str(self.result2[3]))
#色度测量
    #拍摄图像
    def Chrom_capture(self):
        if (self.radioButton_7.isChecked()):
            self.img_Chrom = self.left_frame.copy()
            # if(self.radioButton_9.isChecked()):
            #     self.m=chromm_4_blue_l
            # elif (self.radioButton_10.isChecked()):
            #     self.m=chromm_4_green_l
            # elif (self.radioButton_11.isChecked()):
            #     self.m=chromm_4_red_l
        elif (self.radioButton_8.isChecked()):
            self.img_Chrom  = self.right_frame.copy()
            # if (self.radioButton_9.isChecked()):
            #     self.m = chromm_4_blue_r
            # elif (self.radioButton_10.isChecked()):
            #     self.m = chromm_4_green_r
            # elif (self.radioButton_11.isChecked()):
            #     self.m = chromm_4_red_r
        self.graphicsView_17.setScene(show_picked_pic(self.img_Chrom, self.graphicsView_17.width()))
    #分析色度
    def Chrom_detection(self):
        self.textBrowser_5.clear()
        if (self.radioButton_7.isChecked()):
            if (self.leftcorners==[]):
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '未获取左边角点坐标')
                msg_box.exec_()
                return
            self.rect_Chrom, pictem = turn_points_to_rects(self.img_Chrom.copy(), self.leftcorners, 10)
        elif (self.radioButton_8.isChecked()):
            if (self.rightcorners==[]):
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '未获取右边角点坐标')
                msg_box.exec_()
                return
            self.rect_Chrom, pictem = turn_points_to_rects(self.img_Chrom.copy(), self.rightcorners, 10)
        # 演示效果
        self.graphicsView_18.setScene(show_picked_pic(pictem, self.graphicsView_18.width()))
        if (self.radioButton_7.isChecked()):
            if(self.radioButton_9.isChecked()):
                self.result1, self.result2 = chromaticity(self.img_Chrom,self.rect_Chrom,'l','Blue')
            elif (self.radioButton_10.isChecked()):
                self.result1, self.result2 = chromaticity(self.img_Chrom,self.rect_Chrom,'l','Green')
            elif (self.radioButton_11.isChecked()):
                self.result1, self.result2 = chromaticity(self.img_Chrom,self.rect_Chrom,'l','Red')
        elif (self.radioButton_8.isChecked()):
            if (self.radioButton_9.isChecked()):
                self.result1, self.result2 = chromaticity(self.img_Chrom,self.rect_Chrom,'r','Blue')
            elif (self.radioButton_10.isChecked()):
                self.result1, self.result2 = chromaticity(self.img_Chrom,self.rect_Chrom,'r','Green')
            elif (self.radioButton_11.isChecked()):
                self.result1, self.result2 = chromaticity(self.img_Chrom,self.rect_Chrom,'r','Red')
        # self.result1, self.result2 = chromaticity(self.img_Chrom,self.rect_Chrom,self.m)

        self.textBrowser_5.append("九点色度测量值：")  # 未完成部分
        for i in self.result1:
            self.textBrowser_5.append(str(i[0]))
        self.textBrowser_5.append("最大色差：")
        self.textBrowser_5.append(str(self.result2[0]))

#畸变测量
    #拍摄图一
    def Distortion_capture1(self):
        if (self.radioButton_12.isChecked()):
            self.img_distortion1 = self.left_frame.copy()
        elif (self.radioButton_13.isChecked()):
            self.img_distortion1 = self.right_frame.copy()

        self.graphicsView_21.setScene(show_picked_pic(self.img_distortion1, self.graphicsView_21.width()))
    #拍摄图二
    def Distortion_capture2(self):
        if (self.radioButton_12.isChecked()):
            self.img_distortion2 = self.left_frame.copy()
        elif (self.radioButton_13.isChecked()):
            self.img_distortion2 = self.right_frame.copy()

        self.graphicsView_22.setScene(show_picked_pic(self.img_distortion2, self.graphicsView_22.width()))

    def Distortion_detection(self):
        #去畸变
        if (self.radioButton_12.isChecked()):
            self.img_distortion1 = undistortion(self.img_distortion1, 'l')
            self.img_distortion2 = undistortion(self.img_distortion2, 'l')
        elif (self.radioButton_13.isChecked()):
            self.img_distortion1 = undistortion(self.img_distortion1, 'r')
            self.img_distortion2 = undistortion(self.img_distortion2, 'r')
        self.graphicsView_21.setScene(show_picked_pic(self.img_distortion1, self.graphicsView_21.width()))
        self.graphicsView_22.setScene(show_picked_pic(self.img_distortion2, self.graphicsView_22.width()))
        #获得九点坐标
        self.result, self.img_distortion1, self.img_distortion2 = get_nine_corners(self.img_distortion1, self.img_distortion2)
        self.graphicsView_21.setScene(show_picked_pic(self.img_distortion1, self.graphicsView_21.width()))
        self.graphicsView_22.setScene(show_picked_pic(self.img_distortion2, self.graphicsView_22.width()))
        #进行测量
        self.result = distortion(self.result)


        self.textBrowser_6.clear()
        if (self.radioButton_12.isChecked()):
            self.textBrowser_6.append("左眼四个拐角的畸变分别为:")
            for i in self.result:
                self.textBrowser_6.append(str(i))

        elif (self.radioButton_13.isChecked()):
            self.textBrowser_6.append("右眼四个拐角的畸变分别为:")
            for i in self.result:
                self.textBrowser_6.append(str(i))
#对比度测量
    def Michelson_capture(self):
        if (self.radioButton_14.isChecked()):
            self.img_Miche = self.left_frame.copy()
            # if(self.spinBox_7.value==-4):
            #     self.a_small=a_4_l_small
            #     self.b_small=b_4_l_small
            #     self.c_small = c_4_l_small
            #     self.d_small = d_4_l_small
            #     self.a_big = a_4_l_big
            #     self.b_big = b_4_l_big
            #     self.c_big = c_4_l_big
            #     self.d_big = d_4_l_big
            # elif (self.spinBox_7.value == -5):
            #     self.a_small = a_5_l_small
            #     self.b_small = b_5_l_small
            #     self.c_small = c_5_l_small
            #     self.d_small = d_5_l_small
            #     self.a_big = a_5_l_big
            #     self.b_big = b_5_l_big
            #     self.c_big = c_5_l_big
            #     self.d_big = d_5_l_big
        elif (self.radioButton_15.isChecked()):
            self.img_Miche = self.right_frame.copy()
            # if (self.spinBox_7.value == -4):
            #     self.a_small = a_4_r_small
            #     self.b_small = b_4_r_small
            #     self.c_small = c_4_r_small
            #     self.d_small = d_4_r_small
            #     self.a_big = a_4_r_big
            #     self.b_big = b_4_r_big
            #     self.c_big = c_4_r_big
            #     self.d_big = d_4_r_big
            # elif (self.spinBox_7.value == -5):
            #     self.a_small = a_5_r_small
            #     self.b_small = b_5_r_small
            #     self.c_small = c_5_r_small
            #     self.d_small = d_5_r_small
            #     self.a_big = a_5_r_big
            #     self.b_big = b_5_r_big
            #     self.c_big = c_5_r_big
            #     self.d_big = d_5_r_big
        self.graphicsView_25.setScene(show_picked_pic(self.img_Miche, self.graphicsView_25.width()))

    def Michelson_detection(self):
        self.textBrowser_7.clear()
        if (self.radioButton_14.isChecked()):
            if (self.leftcorners==[]):
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '未获取左边角点坐标')
                msg_box.exec_()
                return
            self.rect_Miche,pictem=turn_points_to_rects(self.img_Miche.copy(),self.leftcorners[4], 35)
        elif (self.radioButton_15.isChecked()):
            if (self.rightcorners==[]):
                msg_box = QMessageBox(QMessageBox.Warning, '警告', '未获取右边角点坐标')
                msg_box.exec_()
                return
            self.rect_Miche,pictem= turn_points_to_rects(self.img_Miche.copy(), self.rightcorners[4],35)
        # 演示效果
        self.graphicsView_26.setScene(show_picked_pic( pictem, self.graphicsView_25.width()))
        if (self.radioButton_14.isChecked()):
            self.result = michelson_contrast(self.img_Miche, self.rect_Miche, self.expose, 'l')

            # if(self.spinBox_7.value==-4):
            #     self.result = michelson_contrast(self.img_Miche, self.rect_Miche, self.expose,'l')
            # elif (self.spinBox_7.value == -5):
            #     self.result = michelson_contrast(self.img_Miche, self.rect_Miche, self.expose,'l')

        elif (self.radioButton_15.isChecked()):
            self.result = michelson_contrast(self.img_Miche, self.rect_Miche, self.expose,'r')
        # self.result = michelson_contrast(self.img_Miche, self.rect_Miche, self.a_small, self.b_small, self.c_small, self.d_small, self.a_big,
        #                                           self.b_big, self.c_big
        #                                           , self.d_big, self.expose)

        self.textBrowser_7.append('最大亮度：' + str(self.result[0]))
        self.textBrowser_7.append('最小亮度：' + str(self.result[1]))
        self.textBrowser_7.append('区域内迈克尔逊对比度：' + str(self.result[2]))



if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    mainWindow1=mainWindow()
    mainWindow1.show()
    sys.exit(app.exec_())