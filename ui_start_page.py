import numpy as np

from mainWindow import Ui_mainWindow
from NED import Ui_NED
from NinePoint import Ui_NinePoint
from FOVCapture import Ui_FOV_capture
from FOVMeasure import Ui_FOV_measure
from Luminance import Ui_Luminance
from Chromaticity import Ui_Chromaticity
from Distortion import Ui_Distortion
from MichelsonContrast import Ui_MichelsonContrast
from Result import Ui_Result

import VR
import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
import sys

#标定数据参数
#畸变参数 mrx内参矩阵 dist畸变系数
mrxl=np.array([[1462.57,0,679.184],[0,1459.77,533.486],[0,0,1]])
mrxr=np.array([[1467.16,0,696.055],[0,1464.95,605.525],[0,0,1]])
distl=[[-0.103343,0.284581,-1.39716,2.18507]]
distr=[[-0.0858416,0.0126673,0.02139,-0.178977]]

#FOV参数 AOVh水平视角 AOVv竖直视角
AOVhl=83.80
AOVhr=41.62
AOVvl=54.30
AOVvr=41.28
# AOVhl=115
# AOVhr=41.62
# AOVvl=86
# AOVvr=41.28
#亮度参数 abc参数
a1_4=-0.2148218818179798
b1_4= 0.35272086836004063
c1_4=-0.003482960386730112
d1_4= 2.000105401551479e-05

a2_4=-310.4355649744981
b2_4= 6.160838583871758
c2_4= -0.03765782911015589
d2_4=8.016941493761997e-05
#
# a1_5=-0.703499222420712
# b1_5=0.8085529218345551
# c1_5=-0.008295102047078357
# d1_5=4.4306824903953195e-05

a1_5=-0.5273567121390397
b1_5=0.7935747758419579
c1_5=-0.007993102840073018
d1_5= 4.2611087187514304e-05

a2_5=-701.0570267069038
b2_5= 13.699431522201802
c2_5= -0.08290481335084757
d2_5=0.00017392745893538


a_LCD_3_l=32.36782
b_LCD_3_l=-0.43597
c_LCD_3_l=0.00358
a_LCD_3_r=None
b_LCD_3_r=None
c_LCD_3_r=None

a_LCD_4_l=19.33070
b_LCD_4_l=-0.17049
c_LCD_4_l=0.00421
a_LCD_4_r=None
b_LCD_4_r=None
c_LCD_4_r=None

a_LCD_5_l=-14.81647
b_LCD_5_l=0.88537
c_LCD_5_l=8.28876e-4
a_LCD_5_r=None
b_LCD_5_r=None
c_LCD_5_r=None


#色度参数 m矩阵
chromm_LCD_4_red_l=np.array([[0.01846893, 0., 0.,157.36181465],[0.00642268, 0., 0.,85.08667434],[-0.02489227,  0. ,  0.,12.5515705]])
chromm_LCD_4_green_l=np.array([[0., 0.00363362, 0.02257666,77.31158946],[0., 0.00777469, 0.00266523,149.01194047],[ 0., -0.0114083 , -0.02524189,28.67647007]])
chromm_LCD_4_blue_l=np.array([[ 0.,-0.00021209,-0.00151631,36.28928985],[0.,0.00775745, -0.00066772,13.7314609],[0., -0.00754099,  0.00218311,204.97929403]])
chromm_LCD_4_red_r=None
chromm_LCD_4_green_r=None
chromm_LCD_4_blue_r=None

chromm_OLED_5_red_l=np.array([[0.01846893, 0., 0.,157.36181465],[0.00642268, 0., 0.,85.08667434],[-0.02489227,  0. ,  0.,12.5515705]])
chromm_OLED_5_green_l=np.array([[0., 0.00363362, 0.02257666,77.31158946],[0., 0.00777469, 0.00266523,149.01194047],[ 0., -0.0114083 , -0.02524189,28.67647007]])
chromm_OLED_5_blue_l=np.array([[ 0.,-0.00021209,-0.00151631,36.28928985],[0.,0.00775745, -0.00066772,13.7314609],[0., -0.00754099,  0.00218311,204.97929403]])
chromm_OLED_5_red_r=None
chromm_OLED_5_green_r=None
chromm_OLED_5_blue_r=None


#设置曝光
expose=-3
#储存当时图像
pic=None
#关闭摄像头资源
camera=1
#角点资源
leftcorners=[]
rightcorners=[]

camera_w=1280
camera_h=480
# camera_w=1280
# camera_h=960
#主界面
class mainWindow(QtWidgets.QWidget,Ui_mainWindow):

    def __init__(self):
        super(mainWindow,self).__init__()
        #初始化
        self.setupUi(self)

        #按钮事件连接
        self.pushButton.clicked.connect(self.clicked_NED)
        self.pushButton.clicked.connect(self.open_cam)
        self.pushButton_2.clicked.connect(self.clicked_NinePoint)
        self.pushButton_2.clicked.connect(self.open_cam)
        self.pushButton_3.clicked.connect(self.clicked_FOVCapture)
        self.pushButton_3.clicked.connect(self.open_cam)
        self.pushButton_4.clicked.connect(self.clicked_Luminance)
        self.pushButton_5.clicked.connect(self.clicked_Chromaticity)
        self.pushButton_6.clicked.connect(self.clicked_Distortion)
        self.pushButton_6.clicked.connect(self.open_cam)
        self.pushButton_7.clicked.connect(self.clicked_MichelsonContrast)
        self.spinBox.valueChanged.connect(self.clicked_expose)

    def clicked_NED(self):
        self.close()
        self.NED=NED()
        self.NED.show()


    def clicked_NinePoint(self):
        self.hide()
        self.NinePoint=NinePoint()
        self.NinePoint.show()

    def clicked_FOVCapture(self):
        self.hide()
        self.FOVCapture=FOVCapture()
        self.FOVCapture.show()

    def clicked_Luminance(self):
        self.hide()
        self.Luminance=Luminance()
        self.Luminance.show()

    def clicked_Chromaticity(self):
        self.hide()
        self.Chromaticity=Chromaticity()
        self.Chromaticity.show()

    def clicked_Distortion(self):
        self.hide()
        self.Distortion=Distortion()
        self.Distortion.show()

    def clicked_MichelsonContrast(self):
        self.hide()
        self.MichelsonContrast=MichelsonContrast()
        self.MichelsonContrast.show()

    def clicked_expose(self):
        global expose
        expose=self.spinBox.value()

    def open_cam(self):
        #全局变量用来关闭摄像头和存储图像
        global camera,pic
        cap = cv2.VideoCapture(0)
        camera=1
        if not (cap.isOpened()):

            QMessageBox.information(self,"ErrorBox", "Can't find camera!")
            return
        # 拍摄图像格式  MJPG 格式是motion jpeg，也就是将视频的每一帧都按照jpg格式压缩了，数据量大大降低。然而还有一个问题是，不知道默认的图像质量是多少，可不可以设置。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # 拍摄图像宽度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_w)
        # 拍摄图像高度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,camera_h)
        # 摄像头曝光值
        cap.set(cv2.CAP_PROP_EXPOSURE, expose)
        # 摄像头色调
        cap.set(cv2.CAP_PROP_HUE, 0)

        while (cv2.waitKey(30)!=27 and camera==1):
            # 获取帧
            suc, pic = cap.read()
            # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH),pic.shape)
            # h = cv2.CAP_PROP_FRAME_HEIGHT
            # w = cv2.CAP_PROP_FRAME_WIDTH
            # picW=cv2.resize(picWin,(w // 2,h // 2),interpolation=cv2.INTER_CUBIC)
            # fScale=0.4
            # dsize=(int(h*fScale),int(w*fScale))
            # cv2.resize(pic,dsize)

            left_frame = pic[0:480, 640:1280]
            right_frame = pic[0:480, 0:640]
            pic = np.hstack((left_frame, right_frame))
            # 创建窗口输出缩略图像
            cv2.namedWindow("capturing", 1)
            cv2.imshow("capturing", pic)

        # 按下esc回收窗口，释放摄像头
        cv2.destroyWindow("capturing")
        cap.release()



#NED测量界面
class NED(QtWidgets.QWidget,Ui_NED):
    picNED=None
    def __init__(self):
        super(NED, self).__init__()
        self.setupUi(self)
        self.textBrowser.setText("请先截图再进行测量")
        self.button.clicked.connect(self.clicked_scr)
        self.button2.clicked.connect(self.clicked_check)
        self.button3.clicked.connect(self.clicked_back)

    def clicked_scr(self):
        self.picNED=pic.copy()

        left_img=VR.show_picked_pic(self,self.picNED[:, 0:640, :],self.left_image.width())
        right_img=VR.show_picked_pic(self, self.picNED[:, 640:1280, :],self.right_image.width())

        self.left_image.setScene(left_img)
        self.right_image.setScene(right_img)

    def clicked_check(self):
        self.textBrowser.clear()
        i=VR.position_dection(self,self.picNED)
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

    def clicked_back(self):
        global camera
        self.close()
        camera=0
        cv2.destroyWindow("capturing")
        self.mainWindow1=mainWindow()
        self.mainWindow1.show()



#九点测量界面
class NinePoint(QtWidgets.QWidget,Ui_NinePoint):
    picNine=None
    pic1=None
    pic2=None
    result=[]
    def __init__(self):
        super(NinePoint, self).__init__()
        self.setupUi(self)
        self.left.setChecked(True)
        self.pushButton.clicked.connect(self.click_scr1)
        self.pushButton_2.clicked.connect(self.click_scr2)
        self.pushButton_3.clicked.connect(self.click_dectninepoint)
        self.pushButton_4.clicked.connect(self.click_saveresult)
        self.pushButton_5.clicked.connect(self.click_back)

    def click_scr1(self):
        self.picNine=pic.copy()
        if(self.left.isChecked()):
            self.pic1=self.picNine[:, 0:640, :]
        elif(self.right.isChecked()):
            self.pic1=self.picNine[:, 640:1280, :]
        pic1tem=VR.show_picked_pic(self,self.pic1,self.graphicsView.width())
        self.graphicsView.setScene(pic1tem)

    def click_scr2(self):
        self.picNine = pic.copy()
        if (self.left.isChecked()):
            self.pic2 = self.picNine[:, 0:640, :]
        elif (self.right.isChecked()):
            self.pic2 = self.picNine[:, 640:1280, :]
        pic2tem = VR.show_picked_pic(self, self.pic2, self.graphicsView_2.width())
        self.graphicsView_2.setScene(pic2tem)

    def click_dectninepoint(self):
        self.result,self.pic1,self.pic2=VR.get_nine_corners(self.pic1,self.pic2)
        self.pic1= VR.show_picked_pic(self, self.pic1, self.graphicsView.width())
        self.pic2 = VR.show_picked_pic(self, self.pic2, self.graphicsView_2.width())
        self.graphicsView.setScene(self.pic1)
        self.graphicsView_2.setScene(self.pic2)

    def click_saveresult(self):
        self.resultpage=Result('NinePoint')
        self.resultpage.textBrowser.clear()
        if (self.left.isChecked()):
            self.resultpage.textBrowser.append("左眼角点leftcorners:")
            for i in self.result:
                self.resultpage.textBrowser.append(str(i))
        elif (self.right.isChecked()):
            self.resultpage.textBrowser.append("右眼角点rightcorners:")
            for i in self.result:
                self.resultpage.textBrowser.append(str(i))

        self.resultpage.show()

    def click_back(self):
        global camera,leftcorners,rightcorners
        self.close()
        camera = 0
        if (self.left.isChecked()):
            leftcorners=self.result.copy()
        elif (self.right.isChecked()):
            rightcorners=self.result.copy()

        self.mainWindow1 = mainWindow()
        self.mainWindow1.show()


#FOV测量界面1
class FOVCapture(QtWidgets.QWidget,Ui_FOV_capture):
    pic_FOV=None
    def __init__(self):
        super(FOVCapture, self).__init__()
        self.setupUi(self)
        self.radioButton_2.setChecked(True)
        self.pushButton.clicked.connect(self.click_scr)
        self.pushButton_2.clicked.connect(self.click_binarization)
        self.pushButton_3.clicked.connect(self.click_FOV_changesize)
        self.pushButton_4.clicked.connect(self.click_back)

    def click_scr(self):
        self.pic_FOV=pic.copy()
        if(self.radioButton_2.isChecked()):
            self.pic=self.pic_FOV[:, 0:640, :]
        elif(self.radioButton.isChecked()):
            self.pic=self.pic_FOV[:, 640:1280, :]

        # self.pic=self.pic_FOV

        pictem=VR.show_picked_pic(self,self.pic,self.graphicsView.width())
        self.graphicsView.setScene(pictem)

    def click_binarization(self):
        self.pic=VR.pic_binarization(self.pic)
        self.graphicsView_2.setScene(VR.show_picked_pic(self,self.pic,self.graphicsView_2.width()))

    def click_FOV_changesize(self):
        self.FOV_measure=FOVmeasure(self.pic,self.radioButton_2.isChecked())
        self.FOV_measure.show()

    def click_back(self):
        global camera
        self.close()
        camera = 0
        self.mainWindow1 = mainWindow()
        self.mainWindow1.show()


#FOV测量界面2
#未完成
class FOVmeasure(QtWidgets.QWidget,Ui_FOV_measure):
    pt0=None
    pt1=None
    pt4=None
    aov_h=None
    aov_v=None
    pic_cut=None
    def __init__(self,pic_fromFOV,is_left):
        super(FOVmeasure, self).__init__()
        self.setupUi(self)
        self.pic_cut,self.hw,self.pt1,self.pt4=VR.change_size(pic_fromFOV)
        #确认中心点
        h, w = self.pic_cut.shape[0:2]
        # self.pt0 = (int((self.pt1[0]+self.pt4[0])/2), int((self.pt1[1]+self.pt4[1])/2))
        self.pt0=(int(w/2),int(h/2))
        if (is_left):
            self.aov_h=AOVhl
            self.aov_v=AOVvl
        else:
            self.aov_h = AOVhr
            self.aov_v = AOVvr
        self.graphicsView.setScene(VR.show_picked_pic(self,self.pic_cut,390))
        self.pushButton.clicked.connect(self.click_dectFOV)
        self.pushButton_2.clicked.connect(self.click_save)

    def click_dectFOV(self):

        self.result = VR.fov_Ned(self.pt0, self.aov_h,self.aov_v, self.pt1, self.pt4)##不知道传什么点
        print(self.pt0, self.aov_h,self.aov_v, self.pt1, self.pt4)
        self.textBrowser.append("水平FOV为："+str(self.result[0]))
        self.textBrowser.append("竖直FOV为：" + str(self.result[1]))
        self.textBrowser.append("对角线FOV为：" + str(self.result[2]))

    def click_save(self):
        try:
            StrText = self.textBrowser.toPlainText()
            qS = str(StrText)
            f = open('/***/FOV.txt', 'a')
            print(f.write('\n{}'.format(qS)))
            f.close()
        except Exception as e:
            print(e)




#亮度测量界面
#未完成
class Luminance(QtWidgets.QWidget,Ui_Luminance):
    #expose曝光 win_type屏幕类型 pic——cutLU全画幅截图 camera控制摄像头 pic_Lumin当前处理图像 rects为9点区域
    expose=-5
    win_type="OLED"
    pic_cutLU=None
    camera=0
    pic_Lumin=None
    rects=None
    a=None
    b=None
    c=None
    d=None
    a2= None
    b2= None
    c2= None
    d2= None
    tone=0
    def __init__(self):
        super(Luminance, self).__init__()
        self.setupUi(self)
        self.radioButton_2.setChecked(True)
        self.comboBox.currentIndexChanged.connect(self.choose_wintype)
        self.spinBox.valueChanged.connect(self.choose_expose)
        self.pushButton.clicked.connect(self.open_cam)
        self.pushButton_2.clicked.connect(self.click_scr)
        self.pushButton_3.clicked.connect(self.click_dectLumin)
        self.pushButton_4.clicked.connect(self.click_saveresult)
        self.pushButton_5.clicked.connect(self.click_back)

    def choose_wintype(self):
        self.win_type=self.comboBox.currentText()

    def choose_expose(self):
        self.expose=self.spinBox.value()


    def open_cam(self):
        # 全局变量用来关闭摄像头和存储图像
        cap = cv2.VideoCapture(0)
        if self.camera == 1:
            # 按下esc回收窗口，释放摄像头
            self.camera = 0
            self.pushButton.setText("打开摄像头")
            cv2.destroyWindow("capturing")
            cap.release()

        else:
            if not (cap.isOpened()):
                QMessageBox.information(self, "ErrorBox", "Can't find camera!")
                return
            self.camera = 1

            self.pushButton.setText("关闭摄像头")
        # 拍摄图像格式  MJPG 格式是motion jpeg，也就是将视频的每一帧都按照jpg格式压缩了，数据量大大降低。然而还有一个问题是，不知道默认的图像质量是多少，可不可以设置。
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            # 拍摄图像宽度
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # 拍摄图像高度
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # 摄像头曝光值
            cap.set(cv2.CAP_PROP_EXPOSURE, self.expose)
            # 摄像头色调
            cap.set(cv2.CAP_PROP_HUE, self.tone)

            while (cv2.waitKey(30) != 27 and self.camera == 1):
                # 获取帧
                suc, self.pic = cap.read()
                left_frame = self.pic[0:480, 640:1280]
                right_frame = self.pic[0:480, 0:640]
                self.pic = np.hstack((left_frame, right_frame))
                # 创建窗口输出缩略图像
                cv2.namedWindow("capturing", 1)
                cv2.imshow("capturing", self.pic)
            # 按下esc回收窗口，释放摄像头
            cv2.destroyWindow("capturing")
            cap.release()

    def click_scr(self):
        self.pic_cutLU=self.pic.copy()
        if (self.radioButton_2.isChecked()):
            self.pic_Lumin = self.pic_cutLU[:, 0:640, :]
        elif (self.radioButton.isChecked()):
            self.pic_Lumin = self.pic_cutLU[:, 640:1280, :]
        pictem = VR.show_picked_pic(self, self.pic_Lumin, self.graphicsView.width())
        self.graphicsView.setScene(pictem)
        if(self.expose==-4):
            self.a=a1_4
            self.b=b1_4
            self.c=c1_4
            self.d=d1_4
            self.a2=a2_4
            self.b2=b2_4
            self.c2=c2_4
            self.d2=d2_4
        elif(self.expose==-5):
            self.a = a1_5
            self.b = b1_5
            self.c = c1_5
            self.d = d1_5
            self.a2 = a2_5
            self.b2 = b2_5
            self.c2 = c2_5
            self.d2 = d2_5
        # if (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -3):
        #     self.a = a_LCD_3_l
        #     self.b = b_LCD_3_l
        #     self.c = c_LCD_3_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -4):
        #     self.a = a_LCD_4_l
        #     self.b = b_LCD_4_l
        #     self.c = c_LCD_4_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -5):
        #     self.a = a_LCD_5_l
        #     self.b = b_LCD_5_l
        #     self.c = c_LCD_5_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -3):
        #     self.a = a_OLED_3_l
        #     self.b = b_OLED_3_l
        #     self.c = c_OLED_3_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -4):
        #     self.a = a_OLED_4_l
        #     self.b = b_OLED_4_l
        #     self.c = c_OLED_4_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -5):
        #     self.a = a_OLED_5_l
        #     self.b = b_OLED_5_l
        #     self.c = c_OLED_5_l
        # elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -3):
        #     self.a = a_LCD_3_r
        #     self.b = b_LCD_3_r
        #     self.c = c_LCD_3_r
        # elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -4):
        #     self.a = a_LCD_4_r
        #     self.b = b_LCD_4_r
        #     self.c = c_LCD_4_r
        # elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -5):
        #     self.a = a_LCD_5_l
        #     self.b = b_LCD_5_l
        #     self.c = c_LCD_5_l
        # elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -3):
        #     self.a = a_OLED_3_r
        #     self.b = b_OLED_3_r
        #     self.c = c_OLED_3_r
        # elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -4):
        #     self.a = a_OLED_4_r
        #     self.b = b_OLED_4_r
        #     self.c = c_OLED_4_r
        # elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -5):
        #     self.a = a_OLED_5_r
        #     self.b = b_OLED_5_r
        #     self.c = c_OLED_5_r



    def click_dectLumin(self):
        self.textBrowser.clear()
        self.rects,pictem=VR.turn_points_to_rects(self.pic_Lumin.copy(),[320,240],20)

        # if (self.radioButton_2.isChecked()):
        #     self.rects,pictem=VR.turn_points_to_rects(self.pic_Lumin.copy(),leftcorners,10)
        # elif (self.radioButton.isChecked()):
        #     self.rects,pictem= VR.turn_points_to_rects(self.pic_Lumin.copy(), rightcorners, 10)
        #演示效果
        pictem = VR.show_picked_pic(self, pictem, self.graphicsView.width())
        self.graphicsView.setScene(pictem)

        # self.result1,self.result2=VR.luminance(self.pic_Lumin,self.rects,self.a,self.b,self.c)
        self.result1, self.result2=VR.luminance(self.pic_Lumin,self.rects,self.a,self.b,self.c,self.d,self.a2,self.b2,self.c2
                                                ,self.d2,self.expose)
        self.textBrowser.append('九点亮度测量值：')#未完成部分
        for i in self.result1:
            self.textBrowser.append(str(i))
        self.textBrowser.append('最大值：'+str(self.result2[0]))
        self.textBrowser.append('最小值：' + str(self.result2[1]))
        self.textBrowser.append('平均值：' + str(self.result2[2]))
        self.textBrowser.append('不均匀值：' + str(self.result2[3]))


    def click_saveresult(self):
        try:
            StrText = self.textBrowser.toPlainText()
            qS = str(StrText)
            f = open('/***/Luminance.txt', 'a')
            print(f.write('\n{}'.format(qS)))
            f.close()
        except Exception as e:
            print(e)

    def click_back(self):
        self.close()
        self.camera = 0
        self.mainWindow1 = mainWindow()
        self.mainWindow1.show()


#色度测量界面
class Chromaticity(QtWidgets.QWidget,Ui_Chromaticity):
    # expose曝光 win_type屏幕类型 pic——cutLU全画幅截图 camera控制摄像头 pic_Lumin当前处理图像 rects为9点区域 color为颜色
    expose = -3
    win_type = "OLED"
    pic_cutCH = None
    camera = 0
    pic_Chrom = None
    rects = None
    color="Green"
    m=None
    tone=0
    def __init__(self):
        super(Chromaticity, self).__init__()
        self.setupUi(self)
        self.radioButton_2.setChecked(True)
        self.comboBox.currentIndexChanged.connect(self.choose_wintype)
        self.expose=self.spinBox.value()
        self.spinBox.valueChanged.connect(self.choose_expose)
        self.buttonGroup_2.buttonClicked.connect(self.choose_color)
        self.pushButton.clicked.connect(self.open_cam)
        self.pushButton_2.clicked.connect(self.click_scr)
        self.pushButton_3.clicked.connect(self.click_dectChrom)
        self.pushButton_4.clicked.connect(self.click_saveresult)
        self.pushButton_5.clicked.connect(self.click_back)

    def choose_wintype(self):
        self.win_type=self.comboBox.currentText()

    def choose_expose(self):
        self.expose=self.spinBox.value()

    def choose_color(self):
        if(self.radioButton_3.isChecked()):
            self.color='Red'
            self.tone=1000
        elif (self.radioButton_4.isChecked()):
            self.color = 'Green'
        elif (self.radioButton_5.isChecked()):
            self.color = 'Blue'



    def open_cam(self):
        # 全局变量用来关闭摄像头和存储图像
        cap = cv2.VideoCapture(0)

        if self.camera == 1:
            # 按下esc回收窗口，释放摄像头
            self.camera = 0
            self.pushButton.setText("打开摄像头")
            cv2.destroyWindow("capturing")
            cap.release()

        else:
            if not (cap.isOpened()):
                QMessageBox.information(self, "ErrorBox", "Can't find camera!")
                return
            self.camera = 1

            self.pushButton.setText("关闭摄像头")
            # 拍摄图像格式  MJPG 格式是motion jpeg，也就是将视频的每一帧都按照jpg格式压缩了，数据量大大降低。然而还有一个问题是，不知道默认的图像质量是多少，可不可以设置。
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            # 拍摄图像宽度
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            # 拍摄图像高度
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # 摄像头曝光值
            cap.set(cv2.CAP_PROP_EXPOSURE, self.expose)
            # 摄像头色调
            cap.set(cv2.CAP_PROP_HUE, 0)

            while (cv2.waitKey(30) != 27 and self.camera == 1):
                # 获取帧
                suc, self.pic = cap.read()
                # 创建窗口输出缩略图像
                left_frame = self.pic[0:480, 640:1280]
                right_frame = self.pic[0:480, 0:640]
                self.pic = np.hstack((left_frame, right_frame))
                cv2.namedWindow("capturing", 1)
                cv2.imshow("capturing", self.pic)
            # 按下esc回收窗口，释放摄像头
            cv2.destroyWindow("capturing")
            cap.release()

    def click_scr(self):
        self.pic_cutCH=self.pic.copy()
        if (self.radioButton_2.isChecked()):
            self.pic_Chrom = self.pic_cutCH[:, 0:640, :]
        elif (self.radioButton.isChecked()):
            self.pic_Chrom = self.pic_cutCH[:, 640:1280, :]
        pictem = VR.show_picked_pic(self, self.pic_Chrom, self.graphicsView.width())
        self.graphicsView.setScene(pictem)

    #未完成
    def click_dectChrom(self):
        self.textBrowser.clear()
        self.rects, pictem = VR.turn_points_to_rects(self.pic_Chrom.copy(), [320, 240], 20)
        # if (self.radioButton_2.isChecked()):
        #     self.rects,pictem=VR.turn_points_to_rects(self.pic_Chrom.copy(),leftcorners,10)
        # elif (self.radioButton.isChecked()):
        #     self.rects,pictem= VR.turn_points_to_rects(self.pic_Chrom.copy(), rightcorners, 10)
        # 演示效果
        pictem = VR.show_picked_pic(self, pictem, self.graphicsView.width())
        self.graphicsView.setScene(pictem)
        if (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -5 and self.color=='Red'):
            self.m=chromm_LCD_4_red_l
        elif (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -5 and self.color=='Green'):
            self.m=chromm_LCD_4_green_l
        elif (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -5 and self.color=='Blue'):
            self.m=chromm_LCD_4_blue_l
        elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -5 and self.color=='Red'):
            self.m=chromm_OLED_5_red_l
        elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -5 and self.color=='Green'):
            self.m=chromm_OLED_5_green_l
        elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -5 and self.color=='Blue'):
            self.m=chromm_OLED_5_blue_l
        elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -5 and self.color=='Red'):
            self.m=chromm_LCD_4_red_r
        elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -5 and self.color=='Green'):
            self.m=chromm_LCD_4_green_r
        elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -5 and self.color=='Blue'):
            self.m=chromm_LCD_4_blue_r
        elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -5 and self.color=='Red'):
            self.m=chromm_OLED_5_red_r
        elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -5 and self.color=='Green'):
            self.m=chromm_OLED_5_green_r
        elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -5 and self.color=='Blue'):
            self.m=chromm_OLED_5_blue_r
        self.result1,self.result2=VR.chromaticity(self.pic_Chrom,self.rects,self.m)
        self.textBrowser.append("九点色度测量值：")#未完成部分
        for i in self.result1:
            self.textBrowser.append(str(i[0]))

        self.textBrowser.append("最大色差：")
        # self.textBrowser.append(str(self.result2[0]))



    def click_saveresult(self):
        try:
            StrText = self.textBrowser.toPlainText()
            qS = str(StrText)
            f = open('/***/Chromaticity.txt', 'a')
            print(f.write('\n{}'.format(qS)))
            f.close()
        except Exception as e:
            print(e)

    def click_back(self):
        self.close()
        self.camera = 0
        self.mainWindow1 = mainWindow()
        self.mainWindow1.show()

#畸变测量界面
class Distortion(QtWidgets.QWidget,Ui_Distortion):
    picDis=None
    pic1=None
    pic2=None

    def __init__(self):
        super(Distortion, self).__init__()
        self.setupUi(self)
        self.radioButton.setChecked(True)
        self.pushButton.clicked.connect(self.click_scr1)
        self.pushButton_2.clicked.connect(self.click_scr2)
        self.pushButton_3.clicked.connect(self.click_dectdistortion)
        self.pushButton_4.clicked.connect(self.click_saveresult)
        self.pushButton_5.clicked.connect(self.click_back)

    def click_scr1(self):
        self.picDis = pic.copy()
        if (self.radioButton.isChecked()):
            self.pic1 = self.picDis[:, 0:640, :]
        elif (self.radioButton_2.isChecked()):
            self.pic1 = self.picDis[:, 640:1280, :]
        pic1tem = VR.show_picked_pic(self, self.pic1, self.graphicsView.width())
        self.graphicsView.setScene(pic1tem)

    def click_scr2(self):
        self.picDis = pic.copy()
        if (self.radioButton.isChecked()):
            self.pic2 = self.picDis[:, 0:640, :]
        elif (self.radioButton_2.isChecked()):
            self.pic2 = self.picDis[:, 640:1280, :]
        pic2tem = VR.show_picked_pic(self, self.pic2, self.graphicsView_2.width())
        self.graphicsView_2.setScene(pic2tem)

    def click_dectdistortion(self):
        #先去畸变
        if (self.radioButton.isChecked()):
            self.pic1=VR.undistortion(self.pic1,mrxl,distl)
            self.pic2=VR.undistortion(self.pic2,mrxl,distl)
        elif (self.radioButton_2.isChecked()):
            self.pic1 = VR.undistortion(self.pic1, mrxr, distr)
            self.pic2 = VR.undistortion(self.pic2, mrxr, distr)

        pic1tem = VR.show_picked_pic(self, self.pic1, self.graphicsView.width())
        self.graphicsView.setScene(pic1tem)
        pic2tem = VR.show_picked_pic(self, self.pic2, self.graphicsView_2.width())
        self.graphicsView_2.setScene(pic2tem)

        self.result, self.pic1, self.pic2 = VR.get_nine_corners(self.pic1, self.pic2)

        self.pic1 = VR.show_picked_pic(self, self.pic1, self.graphicsView.width())
        self.pic2 = VR.show_picked_pic(self, self.pic2, self.graphicsView_2.width())
        self.graphicsView.setScene(self.pic1)
        self.graphicsView_2.setScene(self.pic2)
        if (self.radioButton.isChecked()):
            # self.result=VR.distortion(self.result,AOVhl,AOVvl)
            self.result = VR.distortion(self.result)

        elif (self.radioButton_2.isChecked()):
            self.result=VR.distortion(self.result,AOVhr,AOVvr)
    def click_saveresult(self):
        #计算畸变结果

        self.resultpage = Result('Distortion')
        self.resultpage.textBrowser.clear()
        if (self.radioButton.isChecked()):
            self.resultpage.textBrowser.append("左眼四个拐角的畸变分别为:")
            for i in self.result:
                self.resultpage.textBrowser.append(str(i))


        elif (self.radioButton_2.isChecked()):
            self.resultpage.textBrowser.append("右眼四个拐角的畸变分别为:")
            for i in self.result:
                self.resultpage.textBrowser.append(str(i))

        self.resultpage.show()

    def click_back(self):
        global camera
        self.close()
        camera = 0
        self.mainWindow1 = mainWindow()
        self.mainWindow1.show()

#迈克尔逊测量界面
class MichelsonContrast(QtWidgets.QWidget,Ui_MichelsonContrast):
    expose = -5
    win_type = "OLED"
    pic_cutMi = None
    camera = 0
    pic_Michel = None
    rects = None
    pic=None
    a=None
    b=None
    c=None
    result=[]
    def __init__(self):
        super(MichelsonContrast, self).__init__()
        self.setupUi(self)
        self.radioButton_2.setChecked(True)
        self.comboBox.currentIndexChanged.connect(self.choose_wintype)
        self.spinBox.valueChanged.connect(self.choose_expose)
        self.pushButton.clicked.connect(self.open_cam)
        self.pushButton_2.clicked.connect(self.click_scr)
        self.pushButton_3.clicked.connect(self.click_dectMichel)
        self.pushButton_4.clicked.connect(self.click_saveresult)
        self.pushButton_5.clicked.connect(self.click_back)

    def choose_wintype(self):
        self.win_type = self.comboBox.currentText()

    def choose_expose(self):
        self.expose = self.spinBox.value()

    def open_cam(self):
        # 全局变量用来关闭摄像头和存储图像
        cap = cv2.VideoCapture(0)
        self.camera = 1
        if not (cap.isOpened()):
            QMessageBox.information(self, "ErrorBox", "Can't find camera!")
            return
        # 拍摄图像格式  MJPG 格式是motion jpeg，也就是将视频的每一帧都按照jpg格式压缩了，数据量大大降低。然而还有一个问题是，不知道默认的图像质量是多少，可不可以设置。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # 拍摄图像宽度
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # 拍摄图像高度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 摄像头曝光值
        cap.set(cv2.CAP_PROP_EXPOSURE, self.expose)
        # 摄像头色调
        cap.set(cv2.CAP_PROP_HUE, 0)

        while (cv2.waitKey(30) != 27 and self.camera == 1):
            # 获取帧
            suc, self.pic = cap.read()
            # 创建窗口输出缩略图像
            left_frame = self.pic[0:480, 640:1280]
            right_frame = self.pic[0:480, 0:640]
            self.pic = np.hstack((left_frame, right_frame))
            cv2.namedWindow("capturing", 1)
            cv2.imshow("capturing", self.pic)
        # 按下esc回收窗口，释放摄像头
        cv2.destroyWindow("capturing")
        cap.release()

    def click_scr(self):
        self.pic_cutMi = self.pic.copy()
        if (self.radioButton_2.isChecked()):
            self.pic_Michel = self.pic_cutMi[:, 0:640, :]
        elif (self.radioButton.isChecked()):
            self.pic_Michel = self.pic_cutMi[:, 640:1280, :]
        pictem = VR.show_picked_pic(self, self.pic_Michel, self.graphicsView.width())
        self.graphicsView.setScene(pictem)

    def click_dectMichel(self):
        self.textBrowser.clear()
        # if (self.radioButton_2.isChecked()):
        #     self.rects, pictem = VR.turn_points_to_rects(self.pic_Michel.copy(), leftcorners[4], 35)
        # elif (self.radioButton.isChecked()):
        #     self.rects, pictem = VR.turn_points_to_rects(self.pic_Michel.copy(), rightcorners[4], 35)
        #修改之后
        self.rects, pictem = VR.turn_points_to_rects(self.pic_Michel.copy(), [320, 240], 35)

        # 演示效果
        pictem = VR.show_picked_pic(self, pictem, self.graphicsView.width())
        self.graphicsView.setScene(pictem)

        # self.result=VR.michelson_contrast(self.pic_Michel,self.rects,self.expose,self.win_type)
        # self.textBrowser.append()#未完成部分
        if (self.expose == -4):
            self.a = a1_4
            self.b = b1_4
            self.c = c1_4
            self.d = d1_4
            self.a2 = a2_4
            self.b2 = b2_4
            self.c2 = c2_4
            self.d2 = d2_4
        elif (self.expose == -5):
            self.a = a1_5
            self.b = b1_5
            self.c = c1_5
            self.d = d1_5
            self.a2 = a2_5
            self.b2 = b2_5
            self.c2 = c2_5
            self.d2 = d2_5
        # if (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -3):
        #     self.a = a_LCD_3_l
        #     self.b = b_LCD_3_l
        #     self.c = c_LCD_3_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -4):
        #     self.a = a_LCD_4_l
        #     self.b = b_LCD_4_l
        #     self.c = c_LCD_4_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'LCD' and self.expose == -5):
        #     self.a = a_LCD_5_l
        #     self.b = b_LCD_5_l
        #     self.c = c_LCD_5_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -3):
        #     self.a = a_OLED_3_l
        #     self.b = b_OLED_3_l
        #     self.c = c_OLED_3_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -4):
        #     self.a = a_OLED_4_l
        #     self.b = b_OLED_4_l
        #     self.c = c_OLED_4_l
        # elif (self.radioButton_2.isChecked() and self.win_type == 'OLED' and self.expose == -5):
        #     self.a = a_OLED_5_l
        #     self.b = b_OLED_5_l
        #     self.c = c_OLED_5_l
        # elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -3):
        #     self.a = a_LCD_3_r
        #     self.b = b_LCD_3_r
        #     self.c = c_LCD_3_r
        # elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -4):
        #     self.a = a_LCD_4_r
        #     self.b = b_LCD_4_r
        #     self.c = c_LCD_4_r
        # elif (self.radioButton.isChecked() and self.win_type == 'LCD' and self.expose == -5):
        #     self.a = a_LCD_5_l
        #     self.b = b_LCD_5_l
        #     self.c = c_LCD_5_l
        # # elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -3):
        #     self.a = a_OLED_3_r
        #     self.b = b_OLED_3_r
        #     self.c = c_OLED_3_r
        # elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -4):
        #     self.a = a_OLED_4_r
        #     self.b = b_OLED_4_r
        #     self.c = c_OLED_4_r
        # elif (self.radioButton.isChecked() and self.win_type == 'OLED' and self.expose == -5):
        #     self.a = a_OLED_5_r
        #     self.b = b_OLED_5_r
        #     self.c = c_OLED_5_r
        self.result = VR.michelson_contrast(self.pic_Michel, self.rects, self.a, self.b, self.c, self.d, self.a2,
                                                  self.b2, self.c2
                                                  , self.d2, self.expose)
        # self.result=VR.michelson_contrast(self.pic_Michel,self.rects,self.a,self.b,self.c)

        self.textBrowser.append('最大亮度：'+str(self.result[0]))
        self.textBrowser.append('最小亮度：'+str(self.result[1]))
        self.textBrowser.append('区域内迈克尔逊对比度：' + str(self.result[2]))

    def click_saveresult(self):
        try:
            StrText = self.textBrowser.toPlainText()
            qS = str(StrText)
            f = open('/***/MichelsonContrast.txt', 'a')
            print(f.write('\n{}'.format(qS)))
            f.close()
        except Exception as e:
            print(e)

    def click_back(self):
        self.close()
        self.camera = 0
        self.mainWindow1 = mainWindow()
        self.mainWindow1.show()

#结果页面
class Result(QtWidgets.QWidget,Ui_Result):
    name=None
    def __init__(self,name):
        super(Result, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.clicked_back)
        self.pushButton_2.clicked.connect(self.clicked_ok)
        self.name=name

    def clicked_ok(self):
        try:
            StrText = self.textBrowser.toPlainText()
            qS = str(StrText)
            file_name='/***/'+self.name+'.txt'
            f = open(file_name, 'a')
            print(f.write('\n{}'.format(qS)))
            f.close()
        except Exception as e:
            print(e)

    def clicked_back(self):
        self.close()




if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    mainWindow1=mainWindow()
    mainWindow1.show()
    sys.exit(app.exec_())