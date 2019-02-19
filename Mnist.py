# -*- coding: utf-8 -*-
import pickle
import numpy as np

class mnistReader():
    def __init__(self, mnistPath):
        self.mnistPath = mnistPath
        self.batch_index = 0
        print('read: ', self.mnistPath)
        with open(self.mnistPath, 'rb') as fo:
            self.train_set, self.valid_set, self.test_set = pickle.load(fo, encoding='bytes')
        self.train_data_label = list(zip(self.train_set[0], self.train_set[1]))
        np.random.shuffle(self.train_data_label)
        
    # 获取下一批batch_size训练数据
    def next_train_batch(self, batch_size=100):
        if self.batch_index < int(len(self.train_data_label)/batch_size):
            train_batch = self.train_data_label[self.batch_index*batch_size : (self.batch_index+1)*batch_size]
            self.batch_index += 1
            return self._decode(train_batch)
        else:
            self.batch_index = 0
            np.random.shuffle(self.train_data_label)
            train_batch = self.train_data_label[self.batch_index*batch_size : (self.batch_index+1)*batch_size]
            self.batch_index += 1
            return self._decode(train_batch)
            
    #获取样本标签，作为生成图片的条件
    def get_sample_label(self, batch_size=64):
        sample = self.train_set[1][0 : batch_size]    
        label = []
        for index in sample:
            hot = np.zeros(10)
            hot[int(index)] = 1
            label.append(hot)
        return label
        
    # 把label转为one-hot向量
    def _decode(self, train_batch):
         label = []
         data = []
         for dat, lab in train_batch:
             data.append(np.reshape(dat, [28, 28, 1]))
             hot = np.zeros(10)
             hot[int(lab)] = 1
             label.append(hot)
         return data, label
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            