import sys,cv2,time,os
import jiemian
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication,QMainWindow,QWidget,QMessageBox,QFileDialog,\
QPushButton,QLabel,QHBoxLayout,QVBoxLayout,QGridLayout
#UI界面用的是MainWindow,所以类中要写QMainWindow(不用Qwidegt)
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon,QImage,QPalette,QBrush
from PyQt5.QtCore import Qt,pyqtSignal,QTimer,QDateTime   #(QDateTime为显示当前时间)
import test_AVQVAE
#plt.rcParams['font.sans-serif']=['SimHei']      #修改默认字体,使matplotlib支持中文

class Main_chuangkou(QMainWindow,jiemian.Ui_MainWindow): #类继承

    singal_matrix=pyqtSignal(np.ndarray,np.ndarray,np.ndarray,np.ndarray)
    #建立四幅图片传送的信号(传给检测窗口)


    def __init__(self):
        super(Main_chuangkou, self).__init__()
        self.initUI()
    def initUI(self):
        self.setupUi(self)  #装载UI界面类传入self
        #self.setWindowTitle("张伟伟")  #主窗口标题
        #self.setWindowIcon(QIcon("图标.jpg"))     #添加图标
        
        background = QPalette()
        #设置背景颜色
        background.setColor(background.Background, QColor(192,253,123))
        #background.setColor(background.Background, QColor(53,38,242))
        self.setPalette(background)

        
        self.setFixedSize(self.width(), self.height())  # 设置窗体不能拉伸,在resize后用好
        self.setWindowFlags(Qt.WindowMinimizeButtonHint)  #设置最小化按钮可用
              
        self.btn_exit.clicked.connect(self.closeEvent)  #点击退出按钮
        
        
#*************************对打开摄像头按钮进行设置****************************************
        self.btn_open_video.clicked.connect(self.open_camera) #点击打开摄像头按钮
        
        self.timer_camera = QTimer(self)   #定义定时器,用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()      #视频流
        self.CAM_NUM = 0                   #0表示电脑自带摄像头;1表示外接摄像头*************
        
        self.timer_camera.timeout.connect(self.show_camera) #若定时器结束,则调用show_camera()
#*****************************************************************************************
        
        time = QTimer(self)   #定义定时器,标签实时显示时间
        time.timeout.connect(self.showtime)
        time.start()
        
        self.camera_detect = QTimer(self)  #定义视频每隔多长时间自动检测一次的定时器
        self.camera_detect.timeout.connect(self.video_detect) #定时时间到调用函数
        
        self.radio_real_detect.clicked.connect(self.setting_read_video) #实时检测单选按钮选中
        self.radio_load_image_detect.clicked.connect(self.setting_read_image)#单个图片导入单选按钮选中
       
        self.btn_load_model.clicked.connect(self.load_model) #点击导入模型按钮
        self.btn_load_image.clicked.connect(self.read_image) #点击导入图片按钮(单个导入图片按钮)
        self.btn_start_detect.clicked.connect(self.start_detect) #点击开始检测按钮
        self.btn_output_folder.clicked.connect(self.output_folder) #输出图片路径按钮
        
        self.btn_processing_detect.clicked.connect(self.detect_window) #打开检测窗口
        
        self.model = '' #初始模型路径设置为空
        self.output_file = '' #初始输出图片路劲设置
        
    def closeEvent(self):  #退出事件
        print("用户已退出软件")
        QApplication.instance().quit()
        
#*********************************调用摄像头***********************************************
    def open_camera(self): #开启摄像头
        if self.timer_camera.isActive() == False:  #若定时器未启动
            flag = self.cap.open(self.CAM_NUM)     #0表示电脑自带的,1
            if flag == False:       #flag表示open()成不成功
                QMessageBox.warning(self,'警告',"请检查摄像头与电脑是否正确连接",buttons=QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  #定时器开始计时30ms,即每过30ms从摄像头中取一帧显示
                self.btn_open_video.setText('关闭摄像头') #按钮文字改变
                self.btn_start_detect.setEnabled(True) #开始检测按钮可用
                print('摄像头成功打开')
        else:
            self.timer_camera.stop()              #关闭定时器
            self.cap.release()                    #释放视频流
            self.lab_video.clear()                #清空视频显示区域
            self.btn_open_video.setText('打开摄像头')  #按钮文字改变
            self.btn_start_detect.setEnabled(False) #开始检测按钮可用
            self.lab_video.setText('Image&Video') #关闭摄像头重设标签上的文字
            print('关闭摄像头')


    def show_camera(self): #摄像头相关参数配置     
        flag,self.image = self.cap.read()           #从视频流中读取

        self.camera_show = cv2.resize(self.image,(640,480))     #视频重设长宽大小
        self.camera_rgb = cv2.cvtColor(self.camera_show,cv2.COLOR_BGR2RGB) #RGB通道转换
        showImage = QImage(self.camera_rgb,self.camera_rgb.shape[1],self.camera_rgb.shape[0],QImage.Format_RGB888) #把读取到的视频数据变成QImage形式
        self.lab_video.setPixmap(QPixmap.fromImage(showImage))  #往显示视频的Label里显示QImage
        self.lab_video.setScaledContents(True)    #设置图片尺寸自适应
 #******************************************************************************************       
        

    def showtime(self):         #在标签上实时显示当前时间的函数
        datetime = QDateTime.currentDateTime()
        text = datetime.toString("hh:mm:ss")       #设置显示时间的格式
        self.lab_time.setText(text)
        
    
    def setting_read_video(self): #实时检测单选按钮事件
        print('在线检测模式')
        self.btn_open_video.setEnabled(True)   #打开摄像头按钮可用
        self.btn_load_image.setEnabled(False)  #导入图片按钮不可用
        self.btn_start_detect.setEnabled(False)
        self.lab_video.setText('Image&Video') 
        #self.btn_load_image.setText('抓取图片')
        
    
    def setting_read_image(self): #单个图片导入检测单选按钮事件
        print('读取单个图片检测模式')
        #self.btn_load_image.setText('导入图片')
        self.btn_open_video.setEnabled(False) #打开摄像头按钮不可用
        self.btn_load_image.setEnabled(True)  #导入图片按钮可用
        self.btn_start_detect.setEnabled(False)#开始检测按钮不可用    
############################这部分保证摄像头会关闭#############################
        self.timer_camera.stop()              #关闭定时器
        self.cap.release()                    #释放视频流
        self.lab_video.clear()                #清空视频显示区域
        self.btn_open_video.setText('打开摄像头')  #按钮文字改变
        self.lab_video.setText('Image&Video') #关闭摄像头重设标签上的文字
###############################################################################   

    def read_image(self): #选择导入的单个图片
        #if self.btn_load_image.text()=='导入图片':  #如果按钮为导入图片
        self.image_path, _ = QFileDialog.getOpenFileName(self, '选择待检测的图片', 'C:/', 'Image files(*.jpg *.gif *.png  *.bmp  *.TIF)')
        self.lab_video.setPixmap(QPixmap(self.image_path))
        self.lab_video.setScaledContents(True)  #图片自适应大小
        if self.lab_video.text() == "" and self.image_path == "": #成立表明用户取消添加图片
            self.lab_video.setText('Image&Video')
            self.btn_start_detect.setEnabled(False)
        
        if self.image_path != "":   #成立表明已经添加图片      
            self.btn_start_detect.setEnabled(True) #开始检测按钮可用
        
        #if self.btn_load_image.text()=='抓取图片': #如果按钮为抓取图片
        #    pass


    def load_model(self): #导入pytorch的模型参数
#########################################模型导入时开始计时##################
        #self.t0 = time.time()################################################
#############################################################################
        self.model, _ = QFileDialog.getOpenFileName(self, '选择导入的权重模型', 'C:/', 'pytorch(*.pth *.pt *.tar)')
        if self.model != "":  #成立模型导入成功
            self.lab_information.setText('模型导入成功\n路径'+str(self.model))
        if self.model =='':    #模型未导入
            self.lab_information.setText('模型未导入')
          
#self.model保存的为pytorch的模型路径##############################
#self.image_path保存的为单个导入图片的绝对路径####################

    def start_detect(self): #开始检测按钮点击事件
        if self.model =='': #表示模型还没有导入
            QMessageBox.warning(self, "提示", "请导入模型参数", QMessageBox.Ok, QMessageBox.Ok)
        if self.model != "":  #成立模型导入成功
            if self.radio_load_image_detect.isChecked()==True: #单张图片检测
            
                print('开始检测')
                #T0 = time.time()
                #调用pytorch测试程序###########################################################
                original_image ,recon_image, residual_image , binary_image, TIME = test_AVQVAE.main(self.model,self.image_path) #pytorch模型
                #对二值化疵点图转换格式后保存
                defect_image = Image.fromarray(255*binary_image.astype(np.uint8)).convert('L')
                defect_image.save('quexiantu_image.jpg') #路径
                self.lab_result.setPixmap(QPixmap('quexiantu_image.jpg'))#加载路径
                self.lab_result.setScaledContents(True)
                os.remove('quexiantu_image.jpg')  #删除保存的图片
                
                self.btn_processing_detect.setEnabled(True) #查看检测过程按钮可用
                #T1 = time.time()
                
                
                #保存图片
                defect_image.save(self.output_file+'/result_image.jpg')
                #print(self.output_file+'/result_image.jpg')
                
                
                self.lab_time_2.setText('本次检测所用时间'+ TIME +'ms')
                
                self.singal_matrix.emit(original_image,recon_image,residual_image,binary_image)
    ##########为检测窗口发送信号(原图;重构图;残差图;疵点图)#################
    
            if self.radio_real_detect.isChecked()==True:  #视频检测模式
                if self.btn_start_detect.text()=='开始检测':
                    self.btn_start_detect.setText('停止检测')
                    self.camera_detect.start(5000) #定义视频每隔多少ms检测一次
                    #t2=0
                    
                    #while(t2!=20): 
                    #    t1 = time.time()
                    #    t2 = int(t1-self.t0)
                    #    print(t2)
                    #    if int(t1-self.t0)%20==0:
                    #传送过去的为pytorch文件
                    '''
                    original_image ,recon_image, residual_image , binary_image = test_VQVAE.main(self.model,self.camera_rgb) #pytorch模型
                    #对二值化疵点图转换格式后保存
                    defect_image = Image.fromarray(255*binary_image.astype(np.uint8)).convert('L')
                    defect_image.save('quexiantu_image.jpg') #路径
                    self.lab_result.setPixmap(QPixmap('quexiantu_image.jpg'))#加载路径
                    self.lab_result.setScaledContents(True)
                    os.remove('quexiantu_image.jpg')  #删除保存的图片
                
                    self.singal_matrix.emit(original_image,recon_image,residual_image,binary_image)
                    '''
                elif self.btn_start_detect.text()=='停止检测':
                    self.camera_detect.stop()                 
                    self.btn_start_detect.setText('开始检测')
                    
                    
    def video_detect(self): #视频帧图片进行检测
        original_image ,recon_image, residual_image , binary_image, TIME = test_AVQVAE.main(self.model,self.camera_rgb) #pytorch模型
        #对二值化疵点图转换格式后保存
        defect_image = Image.fromarray(255*binary_image.astype(np.uint8)).convert('L')
        defect_image.save('quexiantu_image.jpg') #路径
        self.lab_result.setPixmap(QPixmap('quexiantu_image.jpg'))#加载路径
        self.lab_result.setScaledContents(True)
        os.remove('quexiantu_image.jpg')  #删除保存的图片
        
        self.lab_time_2.setText('本次检测所用时间'+ TIME +'ms')
    
        self.singal_matrix.emit(original_image,recon_image,residual_image,binary_image)
        
        self.btn_processing_detect.setEnabled(True) #查看检测过程按钮可用
        #保存图片
        defect_image.save(self.output_file+'/result_image.jpg')

        
    def output_folder(self):
        #print("用户自行选择存储路径")
        dir_path= QFileDialog.getExistingDirectory(self, "选择存储路径", "C:/")  #自行设置存储文件夹

        if dir_path!="":                          #判断路径是否为空
            if dir_path[-1] == '/':  # 解决路径中的/问题,当直接保存为(C,D)盘中时(不保存到某个文件夹下),把最后的/去掉
                dir_path = dir_path[:-1]
                #self.line.setText(dir_path)
            print("用户更改路径为" + str(dir_path)) #表明路径不为空
            self.output_file=str(dir_path)          #self.output_file设置输出图片的路径
                                                   
        
        
    def detect_window(self):    
        m2.show()
    
            
class check_window(QWidget):    #检测窗口
    def __init__(self):
        super(check_window, self).__init__()
        self.initUI()
    def initUI(self):
        self.setWindowIcon(QIcon("图标.jpg"))
        self.setWindowTitle("检测窗口")  
        self.resize(800,500)        
        self.setWindowModality(Qt.ApplicationModal)#把检测窗口设置为应用的模式窗口,其它窗口不能输入
        
        self.figure = plt.figure(facecolor='white') #添加一个画图的画板,设置画板的背景颜色
        self.canvas = FigureCanvas(self.figure)
        
        self.btn_jian=QPushButton("减少阈值")       #加阈值按钮
        self.btn_jian.setFont(QFont("微软雅黑", 12))
        #self.btn_jian.clicked.connect(self.jian_yuzhi)

        #self.btn_jia=QPushButton("增加阈值")         #减阈值按钮
        #self.btn_jia.setFont(QFont("微软雅黑", 12))
        #self.btn_jia.clicked.connect(self.jia_yuzhi)
        
    
        btn_origin=QPushButton("defect image")
        btn_origin.setFont(QFont("微软雅黑", 12))
        btn_origin.clicked.connect(self.origin_plt)
        
        btn_recon=QPushButton("reconstruction image")
        btn_recon.setFont(QFont("微软雅黑", 12))
        btn_recon.clicked.connect(self.recon_plt)
        
        btn_residual=QPushButton("residual image")
        btn_residual.setFont(QFont("微软雅黑", 12))
        btn_residual.clicked.connect(self.residual_plt)
        
        btn_bijiao=QPushButton("compare")
        btn_bijiao.setFont(QFont("微软雅黑", 12))
        btn_bijiao.clicked.connect(self.bijiao_plt)
        
        binary_image=QPushButton("binary image")
        binary_image.setFont(QFont("微软雅黑", 12))
        binary_image.clicked.connect(self.binary_plt)
        
        btn_exit = QPushButton("返回")       #返回按钮
        btn_exit.setFont(QFont("微软雅黑", 12))
        btn_exit.clicked.connect(self.close)
        
        grid = QGridLayout(self)    #设置检测窗口网格布局和水平布局
        hlayout = QHBoxLayout()
        #grid.addWidget(self.btn_jian, 0, 0)
        grid.addWidget(self.canvas, 0, 0)
        #grid.addWidget(self.btn_jia, 0, 2)
        grid.addLayout(hlayout, 1, 0)   #设置水平布局在网格中位置(位于图正下面)
        hlayout.addWidget(btn_origin)
        hlayout.addWidget(btn_recon)
        hlayout.addWidget(btn_residual)
        hlayout.addWidget(btn_bijiao)
        hlayout.addWidget(binary_image)
        hlayout.addWidget(btn_exit)
        
        m1.singal_matrix.connect(self.connect_image)  #接从主界面传过来四幅图片矩阵
        
    #该函数用于接收从主界面传过来的矩阵参数     
    def connect_image(self,origin_image,recon_image,residual_image,binary_image):    
        self.origin_image = origin_image
        self.recon_image = recon_image
        self.residual_image = residual_image
        self.binary_image = binary_image
        #画板初始绘制疵点分布图
        plt.clf()
        plt.cla()
        plt.imshow(self.binary_image,'gray')
        plt.xticks([]),plt.yticks([])
        #plt.title("疵点图",fontsize=20)
        self.canvas.draw()
        
        
    #绘制原始图
    def origin_plt(self):
        plt.clf()   #清除之前的绘图,初始绘图为疵点分布图
        plt.cla()
        plt.imshow(self.origin_image)
        plt.xticks([]),plt.yticks([])
        #plt.title("原始图",fontsize=20)
        self.canvas.draw()
        
    #绘制重构图
    def recon_plt(self):
        plt.clf()
        plt.cla()
        plt.imshow(self.recon_image)
        plt.xticks([]),plt.yticks([])
        #plt.title("重构图",fontsize=20)
        self.canvas.draw()
        
    #绘制残差图
    def residual_plt(self):
        plt.clf()
        plt.cla()
        plt.imshow(self.residual_image,'gray')
        plt.xticks([]),plt.yticks([])
        #plt.title("残差图",fontsize=20)
        self.canvas.draw()

    #绘制三图比较
    def bijiao_plt(self):
        plt.clf()
        plt.cla()

        plt.subplot(131)
        plt.imshow(self.origin_image)
        plt.axis("off")
        #plt.title("原始图",fontsize=12)
        plt.subplot(132)
        plt.imshow(self.recon_image)
        plt.axis("off")
        #plt.title("重构图",fontsize=12)
        plt.subplot(133)
        plt.imshow(self.residual_image,'gray')
        plt.axis('off')
        #plt.title("残差图",fontsize=12,color='red')
        self.canvas.draw()
    #绘制最终疵点分布图    
    def binary_plt(self):
        plt.clf()
        plt.cla()
        plt.imshow(self.binary_image,'gray')
        plt.xticks([]),plt.yticks([])
        #plt.title("检测结果图",fontsize=20)
        self.canvas.draw()
 
 
class introduce(QWidget):   #封面
    def __init__(self):
        super(introduce, self).__init__()

        self.initUI()
    def initUI(self):
        self.setWindowFlags(Qt.FramelessWindowHint)    #去掉边框
        self.resize(600, 400)
        #self.setStyleSheet("background-color:blue")  #背景
        background = QPalette()
        #设置背景颜色
        #background.setColor(background.Background, QColor(192,253,123))            #设置背景颜色
        background.setBrush(background.Background, QBrush(QPixmap('800×600.png')))  #设置背景图片
        #background.setColor(background.Background, QColor(53,38,242))
        self.setPalette(background)

        print("用户打开了软件")

        lab=QLabel("欢迎使用织物缺陷检测软件")
        lab.setFont(QFont("黑体",40,QFont.Bold))
        lab.setStyleSheet("color:aqua")

        lab_my=QLabel("制作人:张伟伟")
        lab_my.setFont(QFont("微软雅黑", 20))
        lab_my.setStyleSheet("color:yellow")

        timer=QTimer(self)   #该定时器用于实时显示时间
        timer.timeout.connect(self.showtime)
        timer.start()

        self.lab_time=QLabel("")    #显示时间的标签
        self.lab_time.setFont(QFont("华文行楷", 30))
        self.lab_time.setStyleSheet("color:orange")

        btn=QPushButton("跳过")
        btn.clicked.connect(self.time1)  #点击按钮直接进入主界面
        btn.setStyleSheet("color:red;background-color:#96f97b")
        btn.setFont(QFont("",20,QFont.Bold))

        vlayout = QVBoxLayout(self)    #设置封面的布局
        grid=QGridLayout(self)
        hlayout=QHBoxLayout()
        vlayout.addWidget(lab)
        vlayout.addLayout(grid)
        grid.addWidget(lab_my,0,0)
        grid.addWidget(btn,0,1)
        vlayout.addWidget(self.lab_time)

        self.time = QTimer(self)     #创建定时器(封面界面的显示延时)
        self.time.start(4000)        #定时时间(ms)
        self.time.timeout.connect(self.time1) #定时时间到

    def time1(self):    #封面界面延时时间到,消失
        m1.show()       #打开主界面
        m0.close()      #封面界面关闭
        self.time.stop()    #关闭定时器(定时器只用1次)

        print("用户进入了界面")


    def showtime(self):     #标签实时显示时间函数
        datetime=QDateTime.currentDateTime()
        text=datetime.toString("yyyy年MM月dd日 hh时mm分ss秒 dddd")
        # text=datetime.toString("MM月dd日 hh时mm分ss秒 dddd")
        self.lab_time.setText(text)     
        
if __name__=='__main__':
    app = QApplication(sys.argv)
    m0 = introduce()
    m0.show()
    m1 = Main_chuangkou() #主界面窗口
    #m1.show()             #窗口show出来
    m2 = check_window()
    app.exec_()
