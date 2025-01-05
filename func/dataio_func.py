

import sys
sys.path.append("/home/zhangjiwei/pyfunc/lib")

from lib_sys import *

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CustomLoopDataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化数据集
        :param data: 三维输入数据，形状为 [D1, D2, D3]
        :param labels: 三维标签数据，形状为 [D1, D2, 1] 或 [D1, D2]
        """
        self.data = data
        self.labels = labels
        self.D1, self.D2 = data.shape[:2]  # 获取 D1 和 D2 的尺寸
        
    def __len__(self):
        """
        返回数据集的样本数量
        """
        return self.D1 * self.D2  # 使用 D1 和 D2 的所有组合数作为数据集长度

    def __getitem__(self, _):
        """
        生成特定维度的组合数据作为样本和标签
        :param _: 不使用 idx，直接用双重循环生成
        :return: (样本, 标签) 对
        """
        # 初始化一个列表用于存储每次迭代的样本
        batch_samples = []
        batch_labels = []
        
        # 通过两个维度的循环来获取样本和标签
        for i in range(self.D1):
            for j in range(self.D2):
                sample = self.data[i, j, :]      # 获取 [D3] 形状的样本
                label = self.labels[i, j]        # 获取对应的标签
                batch_samples.append(sample)     # 存储到列表中
                batch_labels.append(label)
                
        # 将结果转换为 numpy 数组或 torch.Tensor
        batch_samples = np.array(batch_samples)
        batch_labels = np.array(batch_labels)
        
        return batch_samples, batch_labels

# # 创建示例数据
# data = np.random.rand(4, 5, 6)      # 假设数据为 [D1=4, D2=5, D3=6] 的三维数组
# labels = np.random.randint(0, 2, size=(4, 5))  # 假设标签为 [D1=4, D2=5] 的二维数组

# # 创建数据集和数据加载器
# dataset = CustomLoopDataset(data, labels)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # 测试数据加载器
# for batch_samples, batch_labels in dataloader:
#     print("样本形状:", batch_samples.shape)  # 输出 (20, 6)
#     print("标签形状:", batch_labels.shape)  # 输出 (20,)
#     break



class Dataset_angle_hessian_2D_train(Dataset):
    def __init__(self, coord_z, coord_x, angle_hessian, batch_size=10, random=False):
        """
        初始化数据集
        :param data: 三维输入数据，形状为 [D1, D2, D3]
        :param labels: 三维标签数据，形状为 [D1, D2, 1] 或 [D1, D2]
        """
        self.coord_z       = coord_z;
        self.coord_x       = coord_x;
        self.angle_hessian = angle_hessian;
        self.batch_size    = batch_size;
        self.random        = random;
        self.shape         = list(  angle_hessian.shape  );
        self.numx          = self.coord_x.shape[0]
        self.numz          = self.coord_z.shape[0]
        self.z_loop_num    = (self.numz-1)//batch_size + 1
        
        self.module_ou  = WR.get_module_type(angle_hessian)
        
        self.module_in  = WR.get_module_type(coord_x)
        
        # self.label_l2   = ( self.module_ou.sum( angle_hessian * angle_hessian  ) ).item()
        
        
    def __len__(self):
        """
        返回数据集的样本数量
        """
        return self.numx * self.numz

    def get_data(self, z_loop):
        """
        I will get the angle_hessian using batch_size along the z direction
        """
        
        # Calculate the starting and ending indices in the z direction
        start_idz = z_loop * self.batch_size
        end_idz   = min(start_idz + self.batch_size, self.numz)  # Ensure we don't go out of bounds

        # Select data slice based on the calculated indices
        
        x_coord = self.coord_x
        
        z_coord = self.coord_z[ start_idz: end_idz ]
        
        labels  = self.angle_hessian[:, start_idz:end_idz, :, :, :];
        

        return z_coord, x_coord, labels
    

class Dataset_angle_hessian_2D_train_random(Dataset):
    def __init__(self, coord_z, coord_x, angle_hessian, batch_size=10, random=False):
        """
        初始化数据集
        :param data: 三维输入数据，形状为 [D1, D2, D3]
        :param labels: 三维标签数据，形状为 [D1, D2, 1] 或 [D1, D2]
        """
        self.coord_z       = coord_z;
        self.coord_x       = coord_x;
        self.angle_hessian = angle_hessian;
        self.batch_size    = batch_size;
        self.random        = random;
        self.shape         = list(  angle_hessian.shape  );
        self.numx          = self.coord_x.shape[0]
        self.numz          = self.coord_z.shape[0]
        self.z_loop_num    = (self.numz-1)//batch_size + 1
        
        self.module_ou  = WR.get_module_type(angle_hessian)
        
        self.module_in  = WR.get_module_type(coord_x)
        
        # self.label_l2   = ( self.module_ou.sum( angle_hessian * angle_hessian  ) ).item()
        
        
    def __len__(self):
        """
        返回数据集的样本数量
        """
        return self.numx * self.numz

    def get_data(self, z_loop=0):
        """
        I will get the angle_hessian using batch_size along the z direction
        """
        
        random_intx = self.module_in.randint(0, self.numx, (self.batch_size,))
        
        random_intz = self.module_in.randint(0, self.numz, (self.batch_size,))
        
        x_coord = self.coord_x[random_intx]
        
        z_coord = self.coord_z[random_intz]
        
        labels  = self.angle_hessian[:, random_intz, random_intx, :, :];
        

        return z_coord, x_coord, labels

    

class Dataset_amp_train(Dataset):
    def __init__(self, coord_x, in_ND_arr, batch_size=10, random=False):
        """
        
        """
        
        self.coord_x       = coord_x;
        self.in_ND_arr     = in_ND_arr;
        self.batch_size    = batch_size;
        self.random        = random;
        self.shape         = list(  in_ND_arr.shape  );
        self.num           = self.coord_x.shape[0];

        self.loop_num      = (self.num-1)//batch_size + 1
        
        self.module_ou     = WR.get_module_type(in_ND_arr)
        
        self.module_in     = WR.get_module_type(coord_x)
        
        # self.label_l2   = ( self.module_ou.sum( angle_hessian * angle_hessian  ) ).item()
        
        
    def __len__(self):
        """
        返回数据集的样本数量
        """
        return self.num

    def get_data(self, i_loop=0):
        """
        I will get the angle_hessian using batch_size along the shot
        """
        if not self.random:
            # Calculate the starting and ending indices in the z direction
            start_id = i_loop * self.batch_size
            end_id   = min(start_id + self.batch_size, self.num)  
            # Ensure we don't go out of bounds
    
            # Select data slice based on the calculated indices
            input_coord = self.coord_x[start_id:end_id, ...];
    
            labels      = self.in_ND_arr[start_id:end_id, ...];
        
        else:
            random_intx = self.module_in.randint(0, self.num, (self.batch_size,))
            
            input_coord = self.coord_x[random_intx, ...];
    
            labels      = self.in_ND_arr[random_intx, ...];
        
        
        return input_coord, labels
    



class Dataset_coord_and_data_ND(Dataset):
    def __init__(self, coord_list, 
                 in_ND_arr, 
                 batch_size=10, 
                 random=False):
        """
        
        """
        
        self.coord_list    = coord_list;
        self.in_ND_arr     = in_ND_arr;
        self.batch_size    = batch_size;
        
        self.random        = random;
        
        self.shape         = list(  self.in_ND_arr.shape  );
        
        
        self.module_ou     = WR.get_module_type(   self.in_ND_arr   )
        
        self.module_in     = WR.get_module_type(   self.coord_list[0] )
        
        # self.label_l2   = ( self.module_ou.sum( angle_hessian * angle_hessian  ) ).item()
        
        
    def __len__(self):
        """
        返回数据集的样本数量
        """
        return 1

    def get_data(self, i_loop=0):
        """
        I will get the angle_hessian using batch_size along the shot
        """
        num_indices      = self.batch_size
        random_indices   = []
        input_coord_list = []
        
        for dim in range( len(self.shape) ):
            indices = np.random.choice( self.shape[dim], num_indices, replace=True)
            random_indices.append( indices )
            
            input_coord_list.append( self.coord_list[dim][indices] )
         
        if     len(self.shape)==1:
            labels      = self.in_ND_arr[random_indices[0]]
        if     len(self.shape)==2:
            labels      = self.in_ND_arr[random_indices[0], random_indices[1]]
        elif   len(self.shape)==3:
            labels      = self.in_ND_arr[random_indices[0], random_indices[1], random_indices[2]]
        elif   len(self.shape)==4:
            labels      = self.in_ND_arr[random_indices[0], random_indices[1], random_indices[2], random_indices[3]]
        else:
            labels      = self.in_ND_arr[random_indices[0], random_indices[1], random_indices[2], random_indices[3], random_indices[4]]
            
        input_coord = torch.cat(input_coord_list, dim=1)
        
        return input_coord, labels

    
    