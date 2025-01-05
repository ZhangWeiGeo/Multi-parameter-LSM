


import numpy as np
import torch
import cupy as cp
import torch.nn.functional as F
from torch.utils import dlpack
import itertools
import write_read_func as WR



def array_to_torch(array, device):
    '''
    '''
    if isinstance(array, np.ndarray):
        # numpy 数组转换为 torch.Tensor
        output = torch.from_numpy(array).to(device)
        
    elif isinstance(array, cp.ndarray):
        # cupy 数组转换为 numpy，再转换为 torch.Tensor
        # output = torch.from_numpy(cp.asnumpy(array)).to(device)
        output = dlpack.from_dlpack( array.toDlpack() ).to(device)
    
    elif isinstance(array, torch.Tensor):
        # 直接将 torch.Tensor 移动到指定设备
        output = array.to(device)
        
    else:
        # 不支持的类型，抛出错误
        raise RuntimeError(f"Unsupported array type: {type(array)}. Only numpy.ndarray, cupy.ndarray, and torch.Tensor are supported.")
    
    return output

def list_to_torch(input_list):
    
    return [array_to_torch(output) for output in input_list]

def array_to_numpy(array):
    '''
    Converts array to a NumPy array if it is a CuPy array, PyTorch tensor, or already a NumPy array.
    '''
    if isinstance(array, cp.ndarray):
        # Convert CuPy array to NumPy
        output = cp.asnumpy(array)
        
    elif isinstance(array, torch.Tensor):
        # Move tensor to CPU if necessary and convert to NumPy
        output = array.detach().cpu().numpy()
    
    elif isinstance(array, np.ndarray):
        # Already a NumPy array, so just return it
        output = array
    
    else:
        raise RuntimeError(f"Unsupported array type: {type(array)}")
    
    return output

def list_to_numpy(input_list):
    
    return [array_to_numpy(output) for output in input_list]

def array_to_cupy(array):
    '''
    '''
    if isinstance(array, np.ndarray):
        output       =cp.asarray(array);
        
    elif isinstance(array, torch.Tensor):
        # output       = cp.asarray( array.to('cpu').detach().numpy() );
        output       = cp.fromDlpack( dlpack.to_dlpack( array.detach()) )
    
    elif isinstance(array, cp.ndarray):
        output       = array;
    
    else:
        raise RuntimeError(f"array.dtype is {array.dtype}, array is not np.ndarray and torch.Tensor");
    
    return output

def list_to_cupy(input_list):
    
    return [array_to_cupy(output) for output in input_list]


def arr_zeros_like(input_arr):
    """
    """
    if isinstance(input_arr, np.ndarray):
        return np.zeros_like(input_arr)
    elif isinstance(input_arr, cp.ndarray):
        return cp.zeros_like(input_arr)
    elif isinstance(input_arr, torch.Tensor):
        return torch.zeros_like(input_arr).to( input_arr.device )
    else:
        raise TypeError(f"Unsupported input type: {type(input_arr)}")


def arr_like_initial_constant(input_arr, constant_value=0):
    """
    根据输入数组的类型和形状进行原地初始化，将数组的值设置为给定的常数值，支持 torch、cupy 和 numpy。

    参数:
    - input_arr: 输入数组 (torch.Tensor, cupy.ndarray 或 numpy.ndarray)，将被原地修改
    - constant_value: float/int, 初始化的常数值 (默认为0)

    返回:
    - None: 输入数组被原地修改为指定的常数值
    """
    module = WR.get_module_type(input_arr)

    if module == torch:
        input_arr.fill_(constant_value)  # 原地设置为常数
    elif module == cp:
        input_arr[...] = cp.full(input_arr.shape, constant_value, dtype=input_arr.dtype)  # 原地设置为常数
    elif module == np:
        input_arr[...] = np.full(input_arr.shape, constant_value, dtype=input_arr.dtype)  # 原地设置为常数
    else:
        raise ValueError("Unsupported module. Please use 'torch', 'cp', or 'np'.")
        

def list_arr_like_initial(input_list, mean=0, init_weight=0.001, distribution='normal'):
    """
    """
    for arr in input_list:
        arr_like_initial(arr, mean=mean, init_weight=init_weight, distribution=distribution)
    

def arr_like_initial(input_arr, mean=0, init_weight=0.001, distribution='normal'):
    """
    根据输入数组的类型和形状进行原地初始化，支持 torch、cupy 和 numpy。

    参数:
    - input_arr: 输入数组 (torch, cupy, numpy)，将被原地修改
    - mean: float, 初始化的均值 (仅适用于正态分布)
    - init_weight: float, 初始化的标准差或范围
    - distribution: str, 使用的分布类型 ('normal' 或 'uniform')

    返回:
    - None: 输入数组被原地修改
    """
    module = WR.get_module_type(input_arr)
    
    if module == torch:
        if distribution == 'normal':
            input_arr.normal_(mean, init_weight)  # 原地修改
        elif distribution == 'uniform':
            input_arr.uniform_(-init_weight, init_weight)  # 原地修改
        else:
            raise ValueError("Unsupported distribution. Use 'normal' or 'uniform'.")
    
    elif module == cp:
        if distribution == 'normal':
            input_arr[...] = cp.random.normal(mean, init_weight, input_arr.shape, dtype=input_arr.dtype)
        elif distribution == 'uniform':
            input_arr[...] = cp.random.uniform(-init_weight, init_weight, input_arr.shape, dtype=input_arr.dtype)
        else:
            raise ValueError("Unsupported distribution. Use 'normal' or 'uniform'.")
    
    elif module == np:
        if distribution == 'normal':
            input_arr[...] = np.random.normal(mean, init_weight, input_arr.shape).astype(input_arr.dtype)
        elif distribution == 'uniform':
            input_arr[...] = np.random.uniform(-init_weight, init_weight, input_arr.shape).astype(input_arr.dtype)
        else:
            raise ValueError("Unsupported distribution. Use 'normal' or 'uniform'.")
    
    else:
        raise ValueError("Unsupported module. Please use 'torch', 'cp', or 'np'.")

def list_arr_outter_product(input_list):
    """
    """
    module = WR.get_module_type( input_list[0] )
    
    for idx in range(0, len(input_list)-1):
        
        if   idx ==0 :
            tmp1 = module.einsum(  'a,b->ab', input_list[0],  input_list[1] )
        elif idx == 1:
            tmp1 = module.einsum(  'ab,c->abc', tmp1,  input_list[2] )
        elif idx == 2:
            tmp1 = module.einsum(  'abc,d->abcd', tmp1,  input_list[3] )
        elif idx == 3:
            tmp1 = module.einsum(  'abcd,e->abcde', tmp1,  input_list[4] )
        elif idx == 4:
            tmp1 = module.einsum(  'abcde,f->abcdef', tmp1,  input_list[5] )
        elif idx == 5:
            tmp1 = module.einsum(  'abcdef,g->abcdefg', tmp1,  input_list[6] )

def list_nonzero_bool(input_list):
    """
    检查 input_list 是否存在任何非零值。
    支持类型：数字（int、float）、NumPy数组、CuPy数组、PyTorch张量。
    如果存在返回 True，否则返回 False。
    """
    
    if not input_list: 
        return False
    
    for item in input_list:
        if isinstance(item, (int, float)): 
            if item != 0:
                return True
        elif isinstance(item, np.ndarray):  
            if np.any(item != 0):
                return True
        elif isinstance(item, cp.ndarray): 
            if cp.any(item != 0):
                return True
        elif isinstance(item, torch.Tensor):
            if torch.any(item != 0):
                return True
        else:
            raise ValueError(f"Unsupported type in input_list: {type(item)}")
    return False


def array_np_to_contiguous(array, info=''):
    
    if not array.flags['C_CONTIGUOUS']:
        # print(f"Warning: NumPy array {info} is not contiguous, making it contiguous.")
        array = np.ascontiguousarray(array)
        
    return array

def array_cp_to_contiguous(array, info=''):

    if not array.flags['C_CONTIGUOUS']:
        # print(f"Warning: CuPy array {info} is not contiguous, making it contiguous.")
        array = cp.ascontiguousarray(array)
        
    return array

def array_torch_to_contiguous(array, info=''):
    
    if not array.is_contiguous():
        # print(f"Warning: Torch tensor {info} is not contiguous, making it contiguous.")
        array = array.contiguous()
        
    return array

def array_to_float32_contiguous(array, info=''):
    """
    Convert the input array to float32 and ensure it has contiguous memory layout.
    Supports NumPy, Torch, and CuPy arrays.

    Parameters:
    - array: The input array, which can be a NumPy, Torch, or CuPy array.

    Returns:
    - array: The converted array in float32 and contiguous layout.
    """
    if isinstance(array, np.ndarray):
        # Check dtype for NumPy arrays
        if array.dtype != np.float32:
            array = array.astype(np.float32)
            print(f"Warning: NumPy array {info} is not float32, converting to float32.")
        
        # Check contiguous layout for NumPy arrays
        if not array.flags['C_CONTIGUOUS']:
            print(f"Warning: NumPy array {info} is not contiguous, making it contiguous.")
            array = np.ascontiguousarray(array)
            
        return array

    elif isinstance(array, torch.Tensor):
        # Check dtype for PyTorch tensors
        if array.dtype != torch.float32:
            array = array.to(torch.float32)
            print(f"Warning: Torch tensor {info} is not float32, converting to float32.")
        
        # Check contiguous layout for PyTorch tensors
        if not array.is_contiguous():
            print(f"Warning: Torch tensor {info} is not contiguous, making it contiguous.")
            array = array.contiguous()
            
        return array

    elif isinstance(array, cp.ndarray):
        # Check dtype for CuPy arrays
        if array.dtype != cp.float32:
            array = array.astype(cp.float32)
            print(f"Warning: CuPy array {info} is not float32, converting to float32.")
        
        # Check contiguous layout for CuPy arrays
        if not array.flags['C_CONTIGUOUS']:
            print(f"Warning: CuPy array {info} is not contiguous, making it contiguous.")
            array = cp.ascontiguousarray(array)
            
        return array

    else:
        raise TypeError("Unsupported array type. Supported types: np.ndarray, torch.Tensor, cp.ndarray")


def array_to_float32_contiguous_list(array_list, info=''):
    
    for i, array in enumerate(array_list):
        
        array_list[i] = array_to_float32_contiguous(array, info=f"{info}[{i}]");
    
    return array_list


def array_to_float32_contiguous_check(array, info=''):
    """
    Convert the input array to float32 and ensure it has contiguous memory layout.
    Supports NumPy, Torch, and CuPy arrays.

    Parameters:
    - array: The input array, which can be a NumPy, Torch, or CuPy array.

    Returns:
    - array: The converted array in float32 and contiguous layout.
    """
    if isinstance(array, np.ndarray):
        # Check dtype for NumPy arrays
        if array.dtype != np.float32:
            raise TypeError(f"Warning: NumPy array {info} is not float32, converting to float32.")
        
        # Check contiguous layout for NumPy arrays
        if not array.flags['C_CONTIGUOUS']:
            raise TypeError(f"Warning: NumPy array {info} is not contiguous, making it contiguous.")
            
        

    elif isinstance(array, torch.Tensor):
        # Check dtype for PyTorch tensors
        if array.dtype != torch.float32:
            raise TypeError(f"Warning: Torch tensor {info} is not float32, converting to float32.")
        
        # Check contiguous layout for PyTorch tensors
        if not array.is_contiguous():
            raise TypeError(f"Warning: Torch tensor {info} is not contiguous, making it contiguous.")
            
        

    elif isinstance(array, cp.ndarray):
        # Check dtype for CuPy arrays
        if array.dtype != cp.float32:
           raise TypeError(f"Warning: CuPy array {info} is not float32, converting to float32.")
        
        # Check contiguous layout for CuPy arrays
        if not array.flags['C_CONTIGUOUS']:
            raise TypeError(f"Warning: CuPy array {info} is not contiguous, making it contiguous.")
            
        

    else:
        raise TypeError("Unsupported array type. Supported types: np.ndarray, torch.Tensor, cp.ndarray")
        
        
def array_to_float32_contiguous_list_check(array_list, info=''):
    
    for i, array in enumerate(array_list):
        
        array_to_float32_contiguous_check(array, info=f"{info}[{i}]");



def array_to_int32(array, info=''):
    try:
        # 判断是否已经是 int32 类型
        if isinstance(array, (int, np.int32, cp.int32)) or (torch.is_tensor(array) and array.dtype == torch.int32):
            return array

        # 标量类型
        elif isinstance(array, (int, float)):
            print(f"Warning: {info}: scalar {array} of type {type(array).__name__} detected, converting to int32.")
            return np.int32(array)

        # NumPy 标量
        elif isinstance(array, np.generic):
            print(f"Warning: {info}: np.generic of type {type(array).__name__} detected, converting to int32 of np.")
            return np.int32(array)
        
        # NumPy 数组
        elif isinstance(array, np.ndarray):
            if array.dtype != np.int32:
                print(f"Warning: {info}: NumPy array with dtype {array.dtype} detected, converting to int32 of np.")
                return array.astype(np.int32)
            else:
                return array
        
        # CuPy 标量
        elif isinstance(array, cp.generic):
            print(f"Warning: {info}: CuPy generic of type {type(array).__name__} detected, converting to int32 of cp.")
            return cp.int32(array)

        # CuPy 数组
        elif isinstance(array, cp.ndarray):
            if array.dtype != cp.int32:
                print(f"Warning: {info}: CuPy array with dtype {array.dtype} detected, converting to int32 of cp.")
                return array.astype(cp.int32)
            else:
                return array
        
        # Torch 张量
        elif isinstance(array, torch.Tensor):
            if array.dtype != torch.int32:
                print(f"Warning: {info}: Torch tensor with dtype {array.dtype} detected, converting to int32 of torch.")
                return array.to(dtype=torch.int32)
            else:
                return array

        # 列表
        elif isinstance(array, list):
            print(f"Warning: {info}: list detected, converting to NumPy array with int32 dtype.")
            return np.array(array).astype(np.int32)

        # 其他类型
        else:
            print(f"Warning: {info}: unsupported type {type(array).__name__}, returning without conversion.")
            return array

    except Exception as e:
        print(f"Error: {info}: failed to convert {type(array).__name__} to int32: {e}")
        raise



def dict_arr_to_int32(data_dict):
    
    for name, value in data_dict.items():
        
        data_dict[name] = array_to_int32(value, info=str(name) )
    
    return data_dict

def dict_arr_to_float32_contiguous(data_dict):
    
    for name, value in data_dict.items():
        
        data_dict[name] = array_to_float32_contiguous(value, info=str(name) )

    return data_dict

def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return "No GPU available."

    num_gpus = torch.cuda.device_count()
    gpu_info = []

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 2) 
        max_memory_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 2)  
        memory_cached = torch.cuda.memory_reserved(i) / (1024 ** 2)  
        max_memory_cached = torch.cuda.max_memory_reserved(i) / (1024 ** 2) 

        gpu_info.append({
            "GPU": i,
            "Name": gpu_name,
            "Memory Allocated (MB)": memory_allocated,
            "Max Memory Allocated (MB)": max_memory_allocated,
            "Memory Cached (MB)": memory_cached,
            "Max Memory Cached (MB)": max_memory_cached,
        })

    return gpu_info






    

# gpu_memory_info = get_gpu_memory_info()
# for info in gpu_memory_info:
#     print(f"GPU {info['GPU']} ({info['Name']}):")
#     print(f"  Memory Allocated: {info['Memory Allocated (MB)']} MB")
#     print(f"  Max Memory Allocated: {info['Max Memory Allocated (MB)']} MB")
#     print(f"  Memory Cached: {info['Memory Cached (MB)']} MB")
#     print(f"  Max Memory Cached: {info['Max Memory Cached (MB)']} MB")
    
# ===================
# ===================
# how to use: hook_save in main function
# z_grad=[] ;                                       
# list for save gradient
# hook_func = lambda a: hook_save(z_grad,a)         
# define a function, z_grad is the defalut input, a is input of hook_func
# ref_arr_torch.register_hook(hook_func)            
# variable.register_hook means to the input is it's gradient
# mig_forward_pytorch.backward(mig_forward_pytorch);
#variable.backward
def hook_save(z_grad,grad):
    print(grad)
    z_grad.append(grad)
    # return z_grad
    



def calculate_snr(signal, signal2):
    
    module = WR.get_module_type(signal)
    
    # 计算信号的均方根值
    signal_power = module.mean(signal ** 2)
    
    # 计算噪声的均方根值
    noise_power = module.mean((signal - signal2) ** 2)
    
    # 防止除零错误
    if noise_power == 0:
        return float('inf')  # 当噪声为零时，信噪比无限大
    
    # 计算信噪比
    snr = 10 * module.log10(signal_power / noise_power)
    
    return snr.item()

snr = lambda x, y: calculate_snr(x, y);


def add_noise_with_snr(signal, snr_db):
    
    module = WR.get_module_type(signal)
    
    signal_power = module.mean(signal**2)
    
    # 计算信噪比（线性）
    snr_linear = 10 ** (snr_db / 10)
    
    # 计算噪声功率
    noise_power = signal_power / snr_linear
    
    # 生成具有指定噪声功率的噪声
    noise = module.randn_like(signal) * module.sqrt(noise_power)
    
    # 将噪声添加到信号中
    noisy_signal = signal + noise
    
    return noisy_signal



def torch_cal_angle_ref_2D(in_list, angle_start=0, angle_num=90, dangle=1.0, model_para=0):
    '''
    
    model_para=0: constant density; 
    model_para=1: vp-density;
    model_para=2: vp-impedance;
    '''
    
    if len(list(in_list[0].shape)) ==4:
        para1_arr = in_list[0][0, 0, :, :]
        para2_arr = in_list[1][0, 0, :, :]
    
    elif len(list(in_list[0].shape)) ==3:
        para1_arr = in_list[0][0, :, :]
        para2_arr = in_list[1][0, :, :]
    
    elif len(list(in_list[0].shape)) ==2:
        para1_arr = in_list[0][:, :]
        para2_arr = in_list[1][:, :]
    
    
    nx, nz = para1_arr.shape

    ############velocity
    vel1_3d = para1_arr.unsqueeze(0).expand(angle_num, -1, -1)
    
    vel1 = 1.0 * para1_arr[ :, 1:nz ];
    vel2 = 1.0 * para1_arr[ :, nz-1 ].contiguous().unsqueeze(-1);
    
    vel  = torch.cat( (vel1, vel2), axis=-1);
    
    # Add a new dimension at the beginning
    # Expand the array to the desired shape
    vel2_3d = vel.unsqueeze(0).expand(angle_num, -1, -1);
    
    
    #########
    gama1     = torch.arange(angle_start, angle_start + angle_num * dangle, dangle)/180.0 * torch.pi
    gama1     = gama1.to(para1_arr.device);
    
    gama1_3d  = gama1.unsqueeze(-1).unsqueeze(-1).expand(angle_num, nx, nz)
    
    radian 	  =  vel2_3d/vel1_3d * torch.sin( gama1_3d );
    
    radian    =  torch.clip(radian, -1.0, 1.0) ;
    
    gama2_3d  =  1.0 * ( 1.0 * torch.arcsin( radian )  );
    

    if   model_para==0:
        A      = 1.0 * vel2_3d * torch.cos( 1.0 * gama1_3d )
        B      = 1.0 * vel1_3d * torch.cos( 1.0 * gama2_3d )

        output = 1.0 * (A - B) / (A + B);
        
        mask = torch.isnan(output)
        output[mask] = 0; 
        
    elif model_para==1:
        den1_3d = para2_arr.unsqueeze(0).expand(angle_num, -1, -1);
        
        den1 = 1.0 * para2_arr[ :, 1:nz ];
        den2 = 1.0 * para2_arr[ :, nz-1 ].contiguous().unsqueeze(-1);
        
        den  = torch.cat( (den1, den2), axis=-1);
        
        # Add a new dimension at the beginning
        # Expand the array to the desired shape
        den2_3d = den.unsqueeze(0).expand(angle_num, -1, -1)
        

        A      = 1.0 * den2_3d * vel2_3d * torch.cos( 1.0 * gama1_3d )
        B      = 1.0 * den1_3d * vel1_3d * torch.cos( 1.0 * gama2_3d )

        output = 1.0 * (A - B) / (A + B);
        
        mask = torch.isnan(output)
        output[mask] = 0; 
        
    else:
        den1_3d = para2_arr.unsqueeze(0).expand(angle_num, -1, -1)
        
        den1 = 1.0 * para2_arr[ :, 1:nz ];
        den2 = 1.0 * para2_arr[ :, nz-1 ].contiguous().unsqueeze(-1);
        
        den  = torch.cat( (den1, den2), axis=-1);
        
        # Add a new dimension at the beginning
        # Expand the array to the desired shape
        den2_3d = den.unsqueeze(0).expand(angle_num, -1, -1)
        

        A      = 1.0 * den2_3d * torch.cos( 1.0 * gama1_3d ) ##now den is IP
        B      = 1.0 * den1_3d * torch.cos( 1.0 * gama2_3d ) ##now den is IP

        output = 1.0 * (A - B) / (A + B);
        
        mask = torch.isnan(output)
        output[mask] = 0; 
    
    if   len( list(output.shape) ) ==2:
        return output.reshape( [1, 1] +  list(output.shape) )
    elif len( list(output.shape) ) ==3:
        return output.reshape( [1] +  list(output.shape) )
    else:
        return output


class Torch_Cal_angle_ref_2D():
    def __init__(self, angle_start=0, angle_num=90, dangle=1.0, model_para=0):
        '''
        
        model_para=0: constant density; 
        model_para=1: vp-density;
        model_para=2: vp-impedance;
        '''
        
        self.angle_start    = angle_start
        self.angle_num      = angle_num
        self.dangle         = dangle
        self.model_para     = model_para

    def forward(self, in_list):
        
        if len(list(in_list[0].shape)) ==4:
            para1_arr = in_list[0][0, 0, :, :]
            
            if self.model_para>0:
                para2_arr = in_list[1][0, 0, :, :]
        
        elif len(list(in_list[0].shape)) ==3:
            para1_arr = in_list[0][0, :, :]
            
            if self.model_para>0:
                para2_arr = in_list[1][0, :, :]
        
        elif len(list(in_list[0].shape)) ==2:
            para1_arr = in_list[0][:, :]
            
            if self.model_para>0:
                para2_arr = in_list[1][:, :]
            
        nx, nz = para1_arr.shape
        
        ############velocity
        vel1_3d = para1_arr.unsqueeze(0).expand(self.angle_num, -1, -1)
        
        vel1 = 1.0 * para1_arr[ :, 1:nz ];
        vel2 = 1.0 * para1_arr[ :, nz-1 ].contiguous().unsqueeze(-1);
        
        vel  = torch.cat( (vel1, vel2), axis=-1);
        
        # Add a new dimension at the beginning
        # Expand the array to the desired shape
        vel2_3d = vel.unsqueeze(0).expand(self.angle_num, -1, -1);
        
        
        #########
        gama1     = torch.arange(self.angle_start, self.angle_start + self.angle_num * self.dangle, self.dangle)/180.0 * torch.pi
        gama1     = gama1.to(para1_arr.device);
        
        gama1_3d  = gama1.unsqueeze(-1).unsqueeze(-1).expand(self.angle_num, nx, nz)
        
        radian 	  =  vel2_3d/vel1_3d * torch.sin( gama1_3d );
        
        radian    =  torch.clip(radian, -1.0, 1.0) ;
        
        gama2_3d  =  1.0 * ( 1.0 * torch.arcsin( radian )  );
        

        ####constant density
        if   self.model_para==0:
            A      = 1.0 * vel2_3d * torch.cos( 1.0 * gama1_3d )
            B      = 1.0 * vel1_3d * torch.cos( 1.0 * gama2_3d )
            
            output = 1.0 * (A - B) / (A + B);
            
            mask = torch.isnan(output)
            output[mask] = 0; 

        ####velocity - density
        elif self.model_para==1:
            den1_3d = para2_arr.unsqueeze(0).expand(self.angle_num, -1, -1);
            
            den1 = 1.0 * para2_arr[ :, 1:nz ];
            den2 = 1.0 * para2_arr[ :, nz-1 ].contiguous().unsqueeze(-1);
            
            den  = torch.cat( (den1, den2), axis=-1);
            
            # Add a new dimension at the beginning
            # Expand the array to the desired shape
            den2_3d = den.unsqueeze(0).expand(self.angle_num, -1, -1)
            

            A      = 1.0 * den2_3d * vel2_3d * torch.cos( 1.0 * gama1_3d )
            B      = 1.0 * den1_3d * vel1_3d * torch.cos( 1.0 * gama2_3d )

            output = 1.0 * (A - B) / (A + B);
            
            mask = torch.isnan(output)
            output[mask] = 0; 
          
        ####velocity - impedance    
        else:
            den1_3d = para2_arr.unsqueeze(0).expand(self.angle_num, -1, -1)
            
            den1 = 1.0 * para2_arr[ :, 1:nz ];
            den2 = 1.0 * para2_arr[ :, nz-1 ].contiguous().unsqueeze(-1);
            
            den  = torch.cat( (den1, den2), axis=-1);
            
            # Add a new dimension at the beginning
            # Expand the array to the desired shape
            den2_3d = den.unsqueeze(0).expand(self.angle_num, -1, -1)
            

            A      = den2_3d * torch.cos( 1.0 * gama1_3d ) ##now den is IP
            B      = den1_3d * torch.cos( 1.0 * gama2_3d ) ##now den is IP

            output = 1.0 * (A - B) / (A + B);
            
            mask = torch.isnan(output)
            output[mask] = 0; 
    
    
        if   len( list(output.shape) ) ==2:
            return output.reshape( [1, 1] +  list(output.shape) )
        elif len( list(output.shape) ) ==3:
            return output.reshape( [1] +  list(output.shape) )
        else:
            return output
    

class Torch_identify():
    def __init__(self, para1):
        '''
        
        model_para=0: constant density; 
        model_para=1: vp-density;
        model_para=2: vp-impedance;
        '''
        
        self.para1    = para1

    def forward(self, in_list):
        
        if isinstance(in_list, list):
            output = [ tensor *1.0 for tensor in in_list]
        else:
            output = in_list *1.0

        return output
    



def ssim(img1, img2, C1=1e-4, C2=9e-4):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    
    Parameters:
        img1 (torch.Tensor): The first image tensor.
        img2 (torch.Tensor): The second image tensor.
        C1 (float): Constant to avoid division by zero in the computation.
        C2 (float): Constant to avoid division by zero in the computation.
        
    Returns:
        torch.Tensor: The SSIM index.
    """
    # Ensure that images are in [0, 1] range
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    
    # Convert to float
    img1 = img1.float()
    img2 = img2.float()
    
    # Compute mean, variance, and covariance
    mu1 = F.conv2d(img1, torch.ones(1, 1, 11, 11) / 121, padding=5)
    mu2 = F.conv2d(img2, torch.ones(1, 1, 11, 11) / 121, padding=5)
    
    sigma1_sq = F.conv2d(img1 * img1, torch.ones(1, 1, 11, 11) / 121, padding=5) - mu1 * mu1
    sigma2_sq = F.conv2d(img2 * img2, torch.ones(1, 1, 11, 11) / 121, padding=5) - mu2 * mu2
    sigma12 = F.conv2d(img1 * img2, torch.ones(1, 1, 11, 11) / 121, padding=5) - mu1 * mu2
    
    # Compute SSIM
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Return mean SSIM index
    return ssim_map.mean()

# # Example usage
# img1 = torch.rand(1, 1, 256, 256)  # Example image 1 (B, C, H, W)
# img2 = torch.rand(1, 1, 256, 256)  # Example image 2 (B, C, H, W)

# ssim_index = ssim(img1, img2)
# print(f"SSIM Index: {ssim_index.item()}")


def print_graph(grad_fn, level=0):
    indent = "  " * level  # 缩进显示层级
    if grad_fn is not None:
        # 显示当前操作的名称
        print(f"{indent}Function: {grad_fn.__class__.__name__}")
        
        # 检查是否有变量名
        if hasattr(grad_fn, 'variable') and grad_fn.variable is not None:
            print(f"{indent}Variable: {grad_fn.variable.name if hasattr(grad_fn.variable, 'name') else 'Unnamed'}")
        
        # 递归地显示下一层函数
        for i, next_fn in enumerate(grad_fn.next_functions):
            if next_fn[0] is not None:
                print(f"{indent}|--[Next Function {i}]")
                print_graph(next_fn[0], level + 1)
                
            
def loss_l2(obs, sys):
    
    res  = obs - sys ;
    loss = 0.5 * torch.sum( res * res )
    
    return loss

def loss_l1(obs, sys):
    
    res  = obs - sys ;
    loss = 0.5 * torch.sum(torch.abs( res ))
    
    return loss

def loss_reg_l2(sys):

    return 0.5 * torch.sum( sys * sys )




def NCC(data1, data2):
    """
    计算归一化的零延迟互相关系数 (Normalized Zero-Lag Cross-Correlation)

    Args:
        data1 (array-like): 第一组数据
        data2 (array-like): 第二组数据
    
    Returns:
        float: 归一化的零延迟互相关系数
    """
    # 转换为 NumPy 数组
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    # 检查数据长度是否一致
    if data1.shape != data2.shape:
        raise ValueError("Input data must have the same shape.")
    
    # 计算均值
    # mean1 = np.mean(data1)
    # mean2 = np.mean(data2)
    
    # 去均值
    centered_data1 = data1 
    centered_data2 = data2 
    
    # 计算分子 (协方差)
    numerator = np.sum(centered_data1 * centered_data2)
    
    # 计算分母 (标准差的乘积)
    std1 = np.sqrt(np.sum(centered_data1**2))
    std2 = np.sqrt(np.sum(centered_data2**2))
    denominator = std1 * std2
    
    # 防止分母为零
    if denominator == 0:
        raise ValueError("Denominator is zero. Cannot calculate NCC.")
    
    # 计算零延迟归一化互相关系数
    ncc = numerator / denominator
    
    return ncc


def linear_interpolate_nd(array, target_shape):
    """
    对输入的 N 维数组进行插值，以适配目标维度。
    参数:
        array: 输入数组，可以是 numpy.ndarray, cupy.ndarray, 或 torch.Tensor。
        target_shape: 目标形状的元组，用于指定插值后的维度大小。
    返回:
        插值后的数组，与输入数组类型相同。
    """
    # 检查输入和目标维度的长度是否一致
    if len(array.shape) != len(target_shape):
        raise ValueError("Input array and target shape must have the same number of dimensions.")

    # 获取模块类型（numpy, cupy, torch）
    module = WR.get_module_type(array)

    if module not in [np, cp, torch]:
        raise TypeError("Input array must be of type numpy.ndarray, cupy.ndarray, or torch.Tensor.")

    # 初始化目标数组
    target_array = module.zeros(target_shape, dtype=array.dtype)
    if module in [torch]:
        target_array = target_array.to(array.device);

    # 创建插值网格
    source_coords = [module.linspace(0, dim - 1, dim) for dim in array.shape]
    target_coords = [module.linspace(0, dim - 1, target_dim) for dim, target_dim in zip(array.shape, target_shape)]

    # 插值逻辑
    if module == np:
        # NumPy 或 CuPy 插值
        from scipy.interpolate import RegularGridInterpolator

        interpolator = RegularGridInterpolator(source_coords, array, bounds_error=False, fill_value=None)
        target_grids = module.meshgrid(*target_coords, indexing='ij')
        flat_grids  = module.stack([grid.ravel() for grid in target_grids], axis=-1)
        interpolated_values = interpolator(flat_grids).reshape(target_shape)
        target_array[:]     = interpolated_values
    
    elif module == cp:
        from scipy.interpolate import RegularGridInterpolator

        interpolator = RegularGridInterpolator([cp.asnumpy(c) for c in source_coords], cp.asnumpy(array), bounds_error=False, fill_value=None)
        target_grids = cp.meshgrid(*target_coords, indexing='ij')
        flat_grids = cp.stack([grid.ravel() for grid in target_grids], axis=-1)
        interpolated_values = cp.asarray(interpolator(flat_grids.get()).reshape(target_shape))
        target_array[:] = interpolated_values

    else:
        import torch.nn.functional as F
        input_tensor = array.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel
        scale_factors = [t / s for s, t in zip(array.shape, target_shape)]

        if len(array.shape) == 1:
            target_array = F.interpolate(input_tensor, size=target_shape, mode='linear', align_corners=True)
        elif len(array.shape) == 2:
            # 2D 插值
            target_array = F.interpolate(input_tensor, size=target_shape, mode='bilinear', align_corners=True)
        elif len(array.shape) == 3:
            # 3D 插值
            target_array = F.interpolate(input_tensor, size=target_shape, mode='trilinear', align_corners=True)
        else:
            raise ValueError("Torch interpolation only supports 1D, 2D, and 3D inputs.")

        # 移除 batch 和 channel 维度（如果不是 4D）
        if len(array.shape) < 4:
            target_array = target_array.squeeze(0).squeeze(0)

    return target_array


def linear_interpolate_nd_test():
    # 测试 NumPy
    print("Testing NumPy...")
    np_array = np.random.rand(4, 4)
    np_target_shape = (8, 8)
    np_result = linear_interpolate_nd(np_array, np_target_shape)
    print("NumPy Result Shape:", np_result.shape)

    # 测试 CuPy
    print("Testing CuPy...")
    cp_array = cp.random.rand(4, 4)
    cp_target_shape = (8, 8)
    cp_result = linear_interpolate_nd(cp_array, cp_target_shape)
    print("CuPy Result Shape:", cp_result.shape)

    # 测试 PyTorch 2D
    print("Testing PyTorch 2D...")
    torch_array_2d = torch.rand(4, 4, device="cuda")
    torch_target_shape_2d = (8, 8)
    torch_result_2d = linear_interpolate_nd(torch_array_2d, torch_target_shape_2d)
    print("PyTorch 2D Result Shape:", torch_result_2d.shape)

    # 测试 PyTorch 3D
    print("Testing PyTorch 3D...")
    torch_array_3d = torch.rand(4, 4, 4, device="cuda")
    torch_target_shape_3d = (8, 8, 8)
    torch_result_3d = linear_interpolate_nd(torch_array_3d, torch_target_shape_3d)
    print("PyTorch 3D Result Shape:", torch_result_3d.shape)
    
    
    
def linear_interpolate_sliding_sum(big_array2, 
                                        small_array1, 
                                        axis=0):
    """
     
    """
    if big_array2.ndim != small_array1.ndim:
       raise ValueError("big_array2 and small_array1 must have the same number of dimensions.")
    for i, (b_dim, s_dim) in enumerate(zip(big_array2.shape, small_array1.shape)):
        if i != axis and b_dim != s_dim:
            raise ValueError(f"Dimension mismatch at axis {i}: big_array2({b_dim}) != small_array1({s_dim})")

    module  = WR.get_module_type(small_array1)
    module2 = WR.get_module_type(big_array2)

    if module != module2:
        raise TypeError("small_array1 and big_array2 must be of the same type and one of numpy.ndarray, cupy.ndarray, or torch.Tensor.")

    # 计算第 1 维度的倍数
    original_size = small_array1.shape[axis]
    multiple_size = (big_array2.shape[axis] // original_size) + (0 if big_array2.shape[axis] % original_size == 0 else 1)
    target_size = original_size * multiple_size

    # 创建插值形状
    interpolate_shape       = list(big_array2.shape)
    interpolate_shape[axis] = target_size
    
    interpolate_arr          = linear_interpolate_nd( big_array2, interpolate_shape );
    
    
    output_arr               = sliding_sum(interpolate_arr, window_sizes=[multiple_size,], axes=[axis,], step_sizes=[multiple_size,])
    
    
    if output_arr.shape != small_array1.shape:
        raise TypeError("output_arr.shape !=small_array1.shape")
    
    return output_arr, interpolate_arr


def linear_interpolate_sliding_sum_test():
    
    big_array2   = torch.zeros((50,60,70));
    small_array1 = torch.zeros((15,60,70));
    axis=0
    
    output_arr, interpolate_arr   =  linear_interpolate_sliding_sum(big_array2, small_array1, axis=0);
    


def sliding_sum(array, window_sizes=[2, 2], axes=[0, 1], step_sizes=[1, 1]):
    """
    通用滑动叠加函数，适配 NumPy, CuPy 和 PyTorch，支持多维滑动和自定义步长。
    
    Args:
        array: 输入数组，可以是 NumPy, CuPy 或 PyTorch 张量。
        window_sizes: 滑动窗口大小列表，长度为 N。
        axes: 滑动的维度列表，与 window_sizes 一一对应。
        step_sizes: 滑动步长列表，与 window_sizes 和 axes 一一对应。
    
    Returns:
        滑动叠加后的数组，与输入类型一致。
    """
    # 自动识别模块类型
    if isinstance(array, torch.Tensor):
        module = torch
    elif isinstance(array, cp.ndarray):
        module = cp
    elif isinstance(array, np.ndarray):
        module = np
    else:
        raise TypeError("Unsupported array type. Use NumPy, CuPy, or PyTorch.")
    
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]
    if isinstance(axes, int):
        axes = [axes]
    if isinstance(step_sizes, int):
        step_sizes = [step_sizes]
    
    if not (len(window_sizes) == len(axes) == len(step_sizes)):
        raise ValueError("window_sizes, axes, and step_sizes must have the same length.")
    
    # 初始化 shape 和 strides
    shape = list(array.shape)
    strides = list(array.strides if module != torch else array.stride())
    
    # 更新 shape 和 strides
    new_shape = list(shape)
    new_strides = list(strides)
    
    for axis, window_size, step_size in zip(axes, window_sizes, step_sizes):
        if window_size <= 0 or window_size > shape[axis]:
            raise ValueError(f"Invalid window size for axis {axis}.")
        if step_size <= 0:
            raise ValueError(f"Step size must be positive for axis {axis}.")
        
        # 计算新的维度大小
        new_dim = (shape[axis] - window_size) // step_size + 1
        new_shape[axis] = new_dim
        new_shape.append(window_size)
        
        # 更新步长
        new_strides[axis] *= step_size
        new_strides.append(strides[axis])
    
    # 构造滑动窗口
    if module == torch:
        sliding_windows = array.as_strided(size=new_shape, stride=tuple(new_strides))
        result = sliding_windows
        for _ in range(len(axes)):
            result = result.sum(dim=-1)
    else:
        sliding_windows = module.lib.stride_tricks.as_strided(array, shape=new_shape, strides=tuple(new_strides))
        result = sliding_windows
        for _ in range(len(axes)):
            result = module.sum(result, axis=-1)
    
    return result




def sliding_sum_test():
    """
    测试 sliding_sum 函数，验证其在 NumPy、CuPy 和 PyTorch 中的正确性。
    """
    # 测试输入
    np_array = np.ones((45, 50, 60), )  
    cp_array = cp.array(np_array)             
    torch_array = torch.tensor(np_array)      

    
    window_sizes = [3, ]
    axes         = [2, ]
    step_sizes   = [3, ] 
    # NumPy 结果
    np_result = sliding_sum(np_array, window_sizes=window_sizes, axes=axes, step_sizes=step_sizes)

    # CuPy 结果
    cp_result = sliding_sum(cp_array, window_sizes=window_sizes, axes=axes, step_sizes=step_sizes)

    # PyTorch 结果
    torch_result = sliding_sum(torch_array, window_sizes=window_sizes, axes=axes, step_sizes=step_sizes)

    # 将结果转换为 NumPy 格式以进行对比
    cp_result_np    = cp.asnumpy(cp_result)
    torch_result_np = torch_result.numpy()

    # 打印结果
    print("NumPy Result:\n", np_result)
    print("CuPy Result:\n", cp_result_np)
    print("PyTorch Result:\n", torch_result_np)

    # 验证结果一致性
    assert np.allclose(np_result, cp_result_np), "NumPy and CuPy results differ!"
    assert np.allclose(np_result, torch_result_np), "NumPy and PyTorch results differ!"

    print("All tests passed successfully!")


