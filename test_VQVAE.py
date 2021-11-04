from __future__ import print_function
import os,sys
import time
import matplotlib.pyplot as plt
import numpy as np
#from scipy.signal import savgol_filter
#from six.moves import xrange
import cv2
from skimage import morphology
from skimage import color
from scipy.misc import imresize
#import scipy.ndimage
import torch
#import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader
#import torch.optim as optim
#from torch.optim import Adam,lr_scheduler
#import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torchvision.utils import make_grid
#from my_dataset import FabricDataset
from PIL import Image
#from save_image import save_one_image_loader
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

os.environ['CUDA_VISIBLE_DEVICES']='0' #GPU设备

from models import VQVAE          #VQVAE
# from model_CBAM import VqVae_cbam #AVQ-VAE
#from model_ECA import VqVae_eca
#from model_SK import VqVae_sk
#%matplotlib inline

def main(checkpoint,image_path):
    checkpoint = checkpoint #模型导入
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE=1
    model_name = VQVAE #VQVAE    
    ##############################################################
    if type(image_path)==str: #字符串类型为单个图片导入
           
        data_info = list()
        image_path = image_path #输入测试图片路径
        data_info.append(os.path.join(image_path))
        test_dir=data_info
        img = Image.open(test_dir[0]).convert('RGB') 
        
    if type(image_path)==np.ndarray: #矩阵为视频流图片
        img = Image.fromarray(image_path.astype(np.uint8))#转换为PIL

        '''
        #定义保存log的文件夹
        model_name_str = str(model_name).split("\'")[-2].split(".")[-1]
        dataset_name_str = data_path.split("/")[-1]
        log_dir = os.path.join("log",dataset_name_str+"_"+model_name_str)

        if not os.path.exists(log_dir): os.makedirs(log_dir)

        #构建完整的测试集路径
        #train_dir=os.path.join(data_path,"train")
        #_train_dir=os.path.join(data_path,"train")
        #test_dir=os.path.join(data_path,"test")

        #构建测试集


        class Logger(object):
        def __init__(self, filename="Default.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass
        path = os.path.abspath(os.path.join('log',dataset_name_str+'_'+model_name_str))
        type = sys.getfilesystemencoding()
        #sys.stdout = Logger('/media/linux/harddisk1/lst/hanhan/log')
        sys.stdout = Logger("residual.txt")
        print(path)

        train_transform=transforms.Compose([
        transforms.Resize((256,256)),
        #transforms.RandomCrop(32,padding=4),
        AddPepperNoise(0.7, p=0.7),
        transforms.ToTensor(),
        #transforms.Normalize(norm_mean,norm_std),
        ])
        _train_transform=transforms.Compose([
        transforms.Resize((256,256)),
        #transforms.RandomCrop(32,padding=4),
        transforms.ToTensor(),
        #transforms.Normalize(norm_mean,norm_std),

        ])
        '''

    test_transform=transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    #transforms.Normalize(norm_mean,norm_std),

    ])

    
    img = test_transform(img)
    img = img.unsqueeze(0)  #把三维图片扩充变为四维

    #training_data=FabricDataset(data_dir=train_dir,transform=train_transform)
    #_training_data=FabricDataset(data_dir=_train_dir,transform=_train_transform)
    #testing_data=FabricDataset(data_dir=test_dir,transform=test_transform)

    #training_loader = DataLoader(dataset=training_data,batch_size=BATCH_SIZE)#shuffle=True)
    #_training_loader =DataLoader(dataset=_training_data,batch_size=BATCH_SIZE)
    testing_loader =DataLoader(dataset=img,batch_size=BATCH_SIZE)

    #num_training_updates = 15000
    #epoch=num_training_updates
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = 64
    num_embeddings = 256

    commitment_cost = 0.25
    decay = 0.99
    #learning_rate = 1e-3
    #定义模型
    model = model_name(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    #载入检查点
    if checkpoint:
        point = torch.load(checkpoint,map_location=device)
        model.load_state_dict(point["state_dict"])
        #print("Loading model form epoch: ",point["epoch"])


    model.eval() #测试阶段
    with torch.no_grad(): #不进行计算图构建
        for i in enumerate(testing_loader):
            (test_originals) = next(iter(testing_loader))
            test_originals = test_originals.to(device)
            
            #test_originals = test_originals.cuda()
            #test_originals = Variable(test_originals,volatile=True)
            vq_output_test = model._pre_vq_conv(model._encoder(test_originals))
            _, test_quantize, _, _ = model._vq_vae(vq_output_test)
            test_reconstructions = model._decoder(test_quantize)

           #保存测试原图和重构图
            #save_one_image_loader(test_originals,'/data/students/master/2020/xiongwb/xwb/VQVAE')
            #save_one_image_loader(test_reconstructions,'/data/students/master/2020/xiongwb/xwb/VQVAE')
            
           #permute是将tensor的维度换位,序号为换位顺序  contiguous()这个函数把tensor变成在内存中连续分布的形式
            test_originals = test_originals.permute(0, 2, 3, 1).contiguous()
            test_reconstructions = test_reconstructions.permute(0, 2, 3, 1).contiguous()
            
            x_test_=test_reconstructions.cpu().detach().numpy() #tensor数据类型转为numpy
            y_test_=test_originals.cpu().detach().numpy()
            
            #保证显示出来的图片亮度变化不大(看起来不会特别模糊) 
            recon = x_test_[0,:,:,:]
            origin = y_test_[0,:,:,:]
            origin = imresize(origin,[512,512]) #原始图
            recon = imresize(recon,[512,512])   #重构图
            
            
            
            def deprocess_image(img):
                img = img * 127.5 + 127.5
                return img.astype('uint8')

            x_test = deprocess_image(x_test_) #deprocess将数据转换成unit8类型
            y_test = deprocess_image(y_test_)
          

            t0=time.time()
            for i in range(x_test.shape[0]):
                y1 = y_test[i, :, :, :]#原图
                img1 = x_test[i, :, :, :]#重构图
                y1 = np.array(Image.fromarray(np.uint8(y1)).resize((512,512)))
                img1 = np.array(Image.fromarray(np.uint8(img1)).resize((512,512)))
                sub = cv2.absdiff(y1,img1)#残差
                
                sub=color.rgb2gray(sub) #灰度化
                sub = cv2.GaussianBlur(sub,(3,3),1) #滤波
                ##均值
                sub_mean = np.mean(sub)
                ##标准差（无偏）
                sub_std =  np.std(sub,ddof=1)
                T = sub_mean + 2*sub_std
                img3=sub

                sub2=(sub>T)*1.0
                img4=sub2
                kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
                kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                opening1 = morphology.binary_opening(sub2, kernel1) 
                img7=opening1
                opening3 = morphology.binary_opening(sub2, kernel3)
                img5=opening3
                closing1 = morphology.binary_closing(sub2, kernel1)
                img8=closing1
                closing3 = morphology.binary_closing(sub2, kernel3)
                img6=closing3

                min_size=150
                img9 = (opening3>0.7)
                img9 = morphology.remove_small_objects(img9,min_size=min_size,connectivity=1)
                #img9 = morphology.convex_hull_object(img9, neighbors=4)

                img10=(closing3>0.7)
                img10=morphology.remove_small_objects(img10,min_size=min_size,connectivity=1)
                img10 = morphology.convex_hull_object(img10, neighbors=4)

                img11=(opening1>0.5) #开1×1
                img11=morphology.remove_small_objects(img11,min_size=min_size,connectivity=1)
                img11 = morphology.convex_hull_object(img11, neighbors=4)

                img12=(closing1>0.5)
                img12 = morphology.remove_small_objects(img12,min_size=min_size,connectivity=1)
                img12 = morphology.convex_hull_object(img12, neighbors=4)  
                #plt.imshow(img11,'gray')
                #plt.show()
    return origin,recon, sub2 ,img11
'''
plt.figure( figsize=(10,10))
plt.imshow(y1)
plt.figure( figsize=(10,10))
plt.imshow(img1)

plt.figure( figsize=(10,10))
plt.imshow(sub)
plt.axis('off')
plt.title("closing")
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
plt.margins(0,0)
plt.savefig(os.path.join(log_dir,"abs_{}.png".format(i)))
#plt.show()
'''
'''
sub=color.rgb2gray(sub) #灰度化
sub = cv2.GaussianBlur(sub,(3,3),1) #滤波
##均值
sub_mean = np.mean(sub)
##标准差（无偏）
sub_std =  np.std(sub,ddof=1)
T = sub_mean + 2*sub_std
img3=sub

sub2=(sub>T)*1.0
img4=sub2
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening1 = morphology.binary_opening(sub2, kernel1) 
img7=opening1
opening3 = morphology.binary_opening(sub2, kernel3)
img5=opening3
closing1 = morphology.binary_closing(sub2, kernel1)
img8=closing1
closing3 = morphology.binary_closing(sub2, kernel3)
img6=closing3

min_size=150
img9 = (opening3>0.7)
img9 = morphology.remove_small_objects(img9,min_size=min_size,connectivity=1)
#img9 = morphology.convex_hull_object(img9, neighbors=4)

img10=(closing3>0.7)
img10=morphology.remove_small_objects(img10,min_size=min_size,connectivity=1)
img10 = morphology.convex_hull_object(img10, neighbors=4)

img11=(opening1>0.5)
img11=morphology.remove_small_objects(img11,min_size=min_size,connectivity=1)
img11 = morphology.convex_hull_object(img11, neighbors=4)

img12=(closing1>0.5)
img12 = morphology.remove_small_objects(img12,min_size=min_size,connectivity=1)
img12 = morphology.convex_hull_object(img12, neighbors=4)         
'''