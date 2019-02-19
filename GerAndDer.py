# -*- coding: utf-8 -*-
import tensorflow as tf
import math
import numpy as np
from scipy.misc import imsave

# 定义批标准化类
class batch_norm(object):#---
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
            
    def __call__(self, x, train=True):
        res = tf.contrib.layers.batch_norm(x,
		decay=self.momentum, 
		updates_collections=None,
		epsilon=self.epsilon, 
		scale=True, 
		is_training=train,
		scope=self.name)
        
        return res
    
#特征图长宽除以步长
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

#将两个tensor进行融合，x代表图像，y代表条件向量
#融合结果：res=[x,0..0,?,0..0],res中的每个元素都是28*28维的矩阵,?代表在标签位置上为全1矩阵
def conv_cond_concat(x, y):
    x_shape = x.get_shape()#[14,14,11] [7 7 128]
    y_shape = y.get_shape()#[1,1,10] [1 1 10]
    return tf.concat([x, y*tf.ones([x_shape[0], x_shape[1], x_shape[2], y_shape[3]])], 3)
#[1,1,10]*[14,14,10]->[14,14,10]
#[1 1 10]*[7 7 10]->[7 7 ]
'''
inputImg:输入图像
output_dim:输出的特征图的个数
'''
def conv2d(inputImg, output_dim, kernel_h=5, kernel_w=5, stride_x=2, stride_y=2, 
           stddev=0.02, name='conv2d'):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [kernel_h, kernel_w, inputImg.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(inputImg, weight, strides=[1, stride_x, stride_y, 1], padding='SAME')    
        biase = tf.get_variable('biase', [output_dim], initializer=tf.constant_initializer(0))
        
        res = tf.reshape(tf.nn.bias_add(conv, biase), conv.get_shape())#---
        
        return res

'''
反卷积
'''
def deconv2d(inputImg, output_shape, kernel_h=5, kernel_w=5, stride_x=2, stride_y=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [kernel_h, kernel_w, output_shape[-1], inputImg.get_shape()[-1]],
                                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        deconv = tf.nn.conv2d_transpose(inputImg, weight, output_shape=output_shape, strides=[1,stride_x,stride_y,1])
        
        biase = tf.get_variable('biase', [output_shape[-1]], initializer=tf.constant_initializer(0))
        
        res = tf.reshape(tf.nn.bias_add(deconv, biase), deconv.get_shape())#---
        return res


def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak*x)

'''
全连接层
input_v：输入向量
output_dim:输出向量的维度
scope：
'''
def fullConnet(input_v, output_dim, scope, stddev=0.02, bias_start=0.0):
    shape = input_v.get_shape().as_list()
    
    with tf.variable_scope(scope):
        weight = tf.get_variable('weight', [shape[1], output_dim], 
                                 tf.float32 ,initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(bias_start))
        
        res = tf.matmul(input_v, weight) + bias
        return res

Dis_cha = 64 #判别器第一层卷积输出的通道数（第一层卷积核个数）(每层卷积输出的通道数基数)
Dis_FC_dim = 1024 #判别器的全连接层输出维度
batch_size = 64
cond_dim = 10 #条件向量的长度(真实样本种类数)
img_cha = 1 #手写数字图像通道数

Gen_cha = 64 #生成器第一层卷积输出的通道数（第一层卷积核个数）(每层卷积输出的通道数基数)
Gen_FC_dim = 1024 #生成器的全连接层输出维度
output_height = 28      # 输出图像高度
output_width = 28       # 输出图像宽度

'''
判别器
image:输入图像维度[64,28,28,1]手写数字灰度图
condition:条件变量 [64, 10]
'''
def  discriminator(image, condition=None, reuse=False):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        Dis_BN1 = batch_norm(name='Dis_BN1')
        Dis_BN2 = batch_norm(name='Dis_BN2')
        
        # 无条件向量
        if condition  == None:
            Dis_BN3 = batch_norm(name='Dis_BN3')
            
            #第一层卷积，[64,28,28,1]->[64,14,14,64]
            #卷积核5*5，步长为2，SAME填充，输出图像长宽为原来1/2
            L0 = lrelu(conv2d(image, Dis_cha, name='Dis_L0_conv'))
            
            # 第二层卷积，[64,14,14,64]->[64,7,7,128]
            L1 = lrelu(Dis_BN1(conv2d(L0, Dis_cha*2, name='Dis_L1_conv')))

            # 第三层卷积，[64,7,7,128]->[64,4,4,256]
            L2 = lrelu(Dis_BN2(conv2d(L1, Dis_cha*4, name='Dis_L2_conv')))
            
            # 第四层卷积，[64,4,4,256]->[64,2,2,512]
            L3 = lrelu(Dis_BN3(conv2d(L2, Dis_cha*8, name='Dis_L3_conv')))
            
            # 把tensor伸展为64*2048维，
            # [64,2,2,512]->[64, 2048]
            L3 = tf.reshape(L3, [batch_size, -1])
            #输入全连接层[64, 2048]->[64, 1]
            L4 = fullConnet(L3, 1, scope='Dis_L4_fc')
            
            #输入sigmod函数，求出为真的概率，[64, 1]
            res = tf.nn.sigmoid(L4)
            
            return res, L4
        #有条件向量  
        else:
            #将条件向量condition重整为[64, 1, 1, 10]
            condition2 = tf.reshape(condition, [batch_size, 1, 1, cond_dim])
            
            #将image[28,28,1]与condition[1,1,10]融合起来，变成[28,28,11]
            #即11张28*28的图，其中第一张为原图，后面10张中有一张为全一矩阵，其所处的位置标记原图的手写数字
            #，其他9张均为全零矩阵
            '''
            [原图][0][0][0][1][0][0][0][0][0][0]
            [64,28,28,1]+[64,1,1,10] -> [64,28,28,11]
            '''
            concat = conv_cond_concat(image, condition2)
            
            # 第一层卷积，[64,28,28,11]->[64,14,14,11]
            L0 = lrelu(conv2d(concat, img_cha+cond_dim, name='Dis_L0_conv'))
            
            # L0与条件向量融合 [64,14,14,11]->[64,14,14,11+10]
            L0 = conv_cond_concat(L0, condition2)
            
            #第二层卷积，[64,14,14,21]->[64,7,7,74]
            L1 = lrelu(Dis_BN1(conv2d(L0, Dis_cha+cond_dim, name='Dis_L1_conv')))
            
            # [64,7,7,74]->[64,7*7*74]=[64, 3626]
            L1 = tf.reshape(L1, [batch_size, -1])
            
            # [64, 3626]+[64, 10] -> [64, 3626+10]
            L1 = tf.concat([L1, condition], 1)
            
            #全连接层 [64, 3626+10]->[64, 1024]
            L2 = lrelu(Dis_BN2(fullConnet(L1, Dis_FC_dim, scope='Dis_L2_fc')))
            
            # [64, 1024] -> [64, 1024+10]
            L2 = tf.concat([L2, condition], 1)
            
            # [64, 1034] -> [64, 1]
            L3 = fullConnet(L2, 1, scope='Dis_L3_fc')
            
            #输入sigmod函数，求出为真的概率，[64,1]
            res = tf.nn.sigmoid(L3)
            
            return res, L3



'''
生成器
z:均匀分布的随机数噪声,[64, 100]
condition:条件变量[64, 10]
'''
def generator(z, condition):
    with tf.variable_scope('generator'):
        Gen_BN0 = batch_norm(name='Gen_BN0')
        Gen_BN1 = batch_norm(name='Gen_BN1')
        Gen_BN2 = batch_norm(name='Gen_BN2')
        
        # 无条件向量
        if condition == None:
            Gen_BN3 = batch_norm(name='Gen_BN3')
            
            # 28,28
            H, W = output_height, output_width
            
            # 14,14
            H2, W2 = conv_out_size_same(H, 2), conv_out_size_same(W, 2)
            
            # 7,7
            H4, W4 = conv_out_size_same(H2, 2), conv_out_size_same(W2, 2)
            
            # 4,4
            H8, W8 = conv_out_size_same(H4, 2), conv_out_size_same(W4, 2)

            # 2,2
            H16, W16 = conv_out_size_same(H8, 2), conv_out_size_same(W8, 2)

            #[64, 100]->[64, 64*8*2*2]=[64, 2048]
            z_ = fullConnet(z, Gen_cha * 8 * H16 * W16, scope='Gen_L0_fc')

            #[64, 2048]->[64, 2, 2, 64*8]
            L0 = tf.reshape(z_, [-1, H16, W16, Gen_cha*8])
            L0 = tf.nn.relu(Gen_BN0(L0))
            
            #第一层反卷积，[64, 2, 2, 512] -> [64, 4,4, 256]
            L1 = deconv2d(L0, [batch_size, H8, W8, Gen_cha*4], name='Gen_L1_deconv')
            L0 = tf.nn.relu(Gen_BN1(L0))

            #第二层反卷积，[64, 4, 4, 256] -> [64, 7, 7, 128]
            L2 = deconv2d(L1, [batch_size, H4, W4, Gen_cha*2], name='Gen_L2_deconv')
            L0 = tf.nn.relu(Gen_BN2(L0))

            #第三层反卷积，[64, 7, 7, 128] -> [64, 14, 14, 64]
            L3 = deconv2d(L2, [batch_size, H2, W2, Gen_cha*1], name='Gen_L3_deconv')
            L3 = tf.nn.relu(Gen_BN3(L3))
            
            #第四层反卷积，[64, 14, 14, 64] ->[64, 28, 28 1]
            L4 = deconv2d(L3, [batch_size, H, W, img_cha], name='Gen_L4_deconv')
            
            #通过tanh函数转换成[-1,1]之间的值
            res = tf.nn.tanh(L4)
            
            return res
        #有条件向量  
        else:
            # 28,28
            H, W = output_height, output_width
            # 14,14
            H2, W2 = int(H/2), int(W/2)
            # 7,7
            H4, W4 = int(H/4), int(W/4)
            
            #z与condition连接融合[64,100]+[64,10] ->[64, 110]
            z = tf.concat([z, condition], 1)
            
            #全连接层，[64, 110]->[64, 1024]
            L0 = tf.nn.relu(Gen_BN0(fullConnet(z, Gen_FC_dim, scope='Gen_L0_fc')))
            
            # L0与condition连接[64, 1024]->[64, 1034]
            L0 = tf.concat([L0, condition], 1)
            
            # 全连接层[64, 1034]->[64, 64*2*7*7]
            L1 = tf.nn.relu(Gen_BN1(fullConnet(L0, Gen_cha*2*H4*W4, scope='Gen_L1_fc')))
            
            # [64, 64*2*7*7]->[64, 7, 7, 128]
            L1 = tf.reshape(L1, [batch_size, H4, W4, Gen_cha*2])

            # condition[64, 10]->[64, 1, 1, 10]
            condition2 = tf.reshape(condition, [batch_size, 1, 1, cond_dim])
            
            # L1与condition2融合[64, 7, 7, 128]+[64, 1, 1, 10] -> [64, 7, 7, 128+10]
            L1 = conv_cond_concat(L1, condition2)
            
            # 反卷积 [64, 7, 7, 138]->[64, 14, 14, 128]
            L2 = tf.nn.relu(Gen_BN2(deconv2d(L1, [batch_size, H2, W2, Gen_cha*2], name='Gen_L2_deconv')))

            # L2与condition2融合[64, 14, 14, 128]+[64, 1, 1, 10] -> [64, 14, 14, 128+10]
            L2 = conv_cond_concat(L2, condition2)
            
            # 反卷积[64, 14, 14, 128+10] -> [64, 28, 28, 1] 图像值在[0, 1]之间
            res = tf.nn.sigmoid(deconv2d(L2, [batch_size, H, W, img_cha], name='Gen_L3_deconv'))

            return res
            



def get_image(image_batch, saveName):
    #有条件向量的GAN使用了sigmod，无条件向量的GAN使用了tanh
    #这里加入了条件向量
    image_batch = image_batch.reshape((image_batch.shape[0], 28, 28))
    img_h, img_w = 28*8, 28*8
    img_grid = np.zeros((img_h, img_w), dtype=np.uint8)
    for index, img in enumerate(image_batch):
        if index >= 64:
            break
        img = img * 255
        img = img.astype(np.uint8)
        row = index // 8 * (28)
        col = (index % 8) * (28)
        img_grid[row:row+28, col:col+28] = img
    
    imsave(saveName, img_grid)
    






















