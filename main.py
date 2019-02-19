# -*- coding: utf-8 -*-

import tensorflow as tf
from GerAndDer import generator, discriminator, get_image
from Mnist import mnistReader
import os,shutil
import numpy as np

batch_size = 64
cond_dim = 10 #条件向量的长度(真实样本种类数)
img_cha = 1 #手写数字图像通道数
output_height = 28      # 输出图像高度
output_width = 28       # 输出图像宽度
input_height = 28      # 输出图像高度
input_width = 28       # 输出图像宽度
z_dim = 10 #随机噪声向量的长度
output_path='D:\\MyProgramma\\myPy\\21_gun\\8_gun\\new1\\MyDCGAN\\DCGAN'  # 生成图像保存路径
mnistPath = 'D:\\MyProgramma\\myPy\\21_gun\\8_gun\\new1\\testdata\\mnist.pkl' #mnist的路径

condition = tf.placeholder(tf.float32, [batch_size, cond_dim], name='condition')#条件向量
img_dim = [input_height, input_width, img_cha]#图像的大小
real_img = tf.placeholder(tf.float32, [batch_size]+img_dim, name='real_img')#训练图像，真实
z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')#随机噪声

Gen_img = generator(z, condition)#生成器，输入噪声向量与条件向量
#判别器，输入真实图像,输出真实图像的为真概率
Pro_real, Dis_logit_real = discriminator(real_img, condition, reuse=False)
#判别器，输入生成图像,输出生成图像的为真概率
Pro_gen, Dis_logit_gen = discriminator(Gen_img, condition, reuse=True)

# 定义判别器的损失函数，使真实图片的sigmod(logit)尽量判为1,生成图片的sigmod(logit)尽量判为0
Dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_logit_real, labels=tf.ones_like(Dis_logit_real)))
Dis_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_logit_gen, labels=tf.zeros_like(Dis_logit_gen)))
Dis_loss = Dis_loss_real + Dis_loss_gen

# 定义生成器的损失函数，使生成图片的sigmod(logit)尽量被判为1
Gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dis_logit_gen, labels=tf.ones_like(Dis_logit_gen)))

# 列出生成器与判别器各自需要优化的参数
train_vars = tf.trainable_variables()
Dis_train_vars = [var for var in train_vars if 'Dis_' in var.name]
Gen_train_vars = [var for var in train_vars if 'Gen_' in var.name]

# 生成器与判别器对各自的参数进行优化
Dis_train_op = tf.train.AdamOptimizer(1e-4).minimize(Dis_loss, var_list=Dis_train_vars)
Gen_train_op = tf.train.AdamOptimizer(1e-4).minimize(Gen_loss, var_list=Gen_train_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path) #重建生成图像保存的文件夹
    
    mnist = mnistReader(mnistPath)
    z_batch = np.random.uniform(-1, 1, [batch_size, z_dim])
    sample_label = mnist.get_sample_label(batch_size=batch_size)
    
    max_step = 100
    for i in range(max_step):
        for j in range(100):
            print('step:' + str(i) + 'iter' + str(j))
            img_batch, lab_batch = mnist.next_train_batch(batch_size=64)
            
            #训练判别器，每次迭代训练一次
            sess.run(Dis_train_op, feed_dict={real_img:img_batch, z:z_batch, condition:lab_batch})
            
            #训练生成器，每次迭代训练两次
            sess.run(Gen_train_op, feed_dict={z:z_batch, condition:lab_batch})
            sess.run(Gen_train_op, feed_dict={z:z_batch, condition:lab_batch})

        
        sample = sess.run(Gen_img, feed_dict={z:z_batch, condition:sample_label})
        
        get_image(sample, os.path.join(output_path, "random_sample%s.jpg" % i))
        


















