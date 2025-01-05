

import sys
import os
import random
import pickle
from filelock import FileLock, Timeout
import re
import types
import ast
import time
import struct
import mmap
import threading
from datetime import datetime
import inspect
import shutil
import gc
import math
import dill
import tokenize

import zipfile
import copy

import numpy as np
import cupy  as cp
import torch
from collections import OrderedDict

import plot_func as PF

#########
def mkdir_unique_dir(filename):
    original_filename = filename
    counter = 1
    
    # 检查路径是否已经存在
    while os.path.exists(filename):
        # 如果路径已存在，则添加一个数字后缀
        filename = f"{original_filename}_{counter}"
        counter += 1
    
    # 创建路径
    os.makedirs(filename)
    print(f"Directory created: {filename}")


def get_unique_dir(filename):
    original_filename = filename.rstrip("/")  # 去掉可能的结尾斜杠
    counter = 1
    
    while os.path.exists(filename):
        filename = f"{original_filename}_{counter}/"
        counter += 1

    return filename


def mkdir(filename):
    if not os.path.isdir(filename):
        #os.makedirs(filename);
        os.system("mkdir {}".format(filename));

def rename(filename,filename1):
    if os.path.exists(filename):
        #os.rename(filename,filename1);
        os.system("mv {}  {}".format(filename,filename1));
        
def cp_r(filename,filename1):  
    zzz = "cp -r {}  {}".format(filename,filename1);
    os.system(zzz);        

def rm_rf(filename):
    ###
    zzz = "rm -rf {} ".format(filename);
    os.system(zzz);

def move_txt_files_to_pwd(log_file="move_txt_files_to_pwd.txt"):
    # 获取当前路径
    
    current_path = os.getcwd()
    # 定义目标文件夹
    target_folder = os.path.join(current_path, 'txt')
    log_file = os.path.join(current_path, log_file)
    
    # 打开日志文件
    with open(log_file, "w") as log:
        # 如果 txt 文件夹不存在，则创建它
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            log.write("Created 'txt' folder.\n")

        # 获取当前路径下的所有 .txt 文件
        txt_files = [f for f in os.listdir(current_path) if f.endswith('.txt')]

        # 检查是否存在 .txt 文件
        if not txt_files:
            log.write("No .txt files found in the current directory.\n")
            return

        # 移动所有 .txt 文件到 txt 文件夹
        for file in txt_files:
            src = os.path.join(current_path, file)
            dest = os.path.join(target_folder, file)
            
            # 如果目标文件已存在，进行重命名
            if os.path.exists(dest):
                base, ext = os.path.splitext(file)
                i = 1
                new_dest = os.path.join(target_folder, f"{base}_{i}{ext}")
                while os.path.exists(new_dest):
                    i += 1
                    new_dest = os.path.join(target_folder, f"{base}_{i}{ext}")
                dest = new_dest  # 更新为重命名后的文件路径
                log.write(f"File '{file}' already exists, renaming to '{os.path.basename(dest)}'\n")
            
            shutil.move(src, dest)
            log.write(f"Moved '{file}' to 'txt' folder.\n")
        
        log.write("All .txt files have been moved to the 'txt' folder.\n")


def move_txt_files_to_pwd_old():
    # 获取当前路径
    current_path = os.getcwd()
    # 定义目标文件夹
    target_folder = os.path.join(current_path, 'txt')
    
    # 如果 txt 文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print("Created 'txt' folder.")

    # 获取当前路径下的所有 .txt 文件
    txt_files = [f for f in os.listdir(current_path) if f.endswith('.txt')]

    # 检查是否存在 .txt 文件
    if not txt_files:
        print("No .txt files found in the current directory.")
        return

    # 移动所有 .txt 文件到 txt 文件夹
    for file in txt_files:
        src = os.path.join(current_path, file)
        dest = os.path.join(target_folder, file)
        
        # 如果目标文件已存在，进行重命名
        if os.path.exists(dest):
            base, ext = os.path.splitext(file)
            i = 1
            new_dest = os.path.join(target_folder, f"{base}_{i}{ext}")
            while os.path.exists(new_dest):
                i += 1
                new_dest = os.path.join(target_folder, f"{base}_{i}{ext}")
            dest = new_dest  # 更新为重命名后的文件路径
            print(f"File '{file}' already exists, renaming to '{os.path.basename(dest)}'")
        
        shutil.move(src, dest)
        print(f"Moved '{file}' to 'txt' folder.")
        
    print("All .txt files have been moved to the 'txt' folder.")

def copy_files_by_extension(input_dir, output_path, zip_name="pyfunc.zip", extensions=['.py', '.cu', '.sh']):
    '''
    在输入路径中搜索指定后缀的文件，直接压缩到目标 .zip 文件中。

    参数：
        input_dir (str): 输入路径。
        output_path (str): 输出 .zip 文件所在的目录路径。
        zip_name (str): 压缩文件的名称。
        extensions (list or str): 要压缩的文件后缀，可以是字符串或列表。
    '''
    
    # 确保参数类型正确
    if not isinstance(output_path, str):
        raise TypeError("output_path must be a string.")
    if not isinstance(zip_name, str):
        raise TypeError("zip_name must be a string.")
        
        
    # 创建输出目录（如果不存在）
    os.makedirs(output_path, exist_ok=True)
    
    # 如果 extensions 是字符串，则转换为列表
    if isinstance(extensions, str):
        extensions = [extensions]
    
    # 确定压缩文件的完整路径
    zip_file_name = os.path.join(output_path, zip_name)
    
    # 创建 zip 文件
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历输入路径下的所有文件夹和文件
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                # 检查文件是否具有列表中的任意一个后缀
                if any(file.endswith(ext) for ext in extensions):
                    # 获取文件的完整路径
                    full_path = os.path.join(root, file)
                    # 在 zip 文件中保存的相对路径
                    arcname = os.path.relpath(full_path, input_dir)
                    # 将文件添加到 zip 文件
                    zipf.write(full_path, arcname)
    
    print(f"Compression completed: {zip_file_name}")



def copy_files_by_extension_old(input_dir, output_dir, extensions=['.py', '.cu', '.sh']):
    '''
    # 示例用法
    input_dir = '/path/to/input_dir'  # 输入路径
    output_dir = '/path/to/output_dir'  # 输出路径
    extensions = '.py'  # 后缀可以是字符串
    # 或者：extensions = ['.py', '.txt', '.md']  # 后缀可以是列表

    copy_files_by_extension(input_dir, output_dir, extensions)
    '''
    # 如果 extensions 是字符串，则转换为列表
    if isinstance(extensions, str):
        extensions = [extensions]
    
    # 遍历输入路径下的所有文件夹和文件
    for root, dirs, files in os.walk(input_dir):
        # 遍历每个文件
        for file in files:
            # 检查文件是否具有列表中的任意一个后缀
            if any(file.endswith(ext) for ext in extensions):
                # 获取文件的相对路径
                relative_path = os.path.relpath(root, input_dir)
                # 构建输出文件夹的路径
                dest_folder = os.path.join(output_dir, relative_path)
                # 如果输出文件夹不存在，则创建它
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)
                # 复制文件到对应的输出路径
                shutil.copy(os.path.join(root, file), os.path.join(dest_folder, file))


def replace_txt(directory, old_call, new_call, extensions=['.py', '.sh']):
    '''
    替换目录中所有指定扩展名文件中的特定内容。

    参数：
        directory (str): 要搜索的目录路径。
        old_call (str): 要替换的旧字符串。
        new_call (str): 替换为的新字符串。
        extensions (list): 要处理的文件扩展名列表，默认为 ['.py', '.sh']。
    '''
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否在指定范围内
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                # 打开文件并读取内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 替换内容
                new_content = content.replace(old_call, new_call)
                
                # 如果有变化，则写回文件
                if content != new_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Modified: {file_path}")


def join_with_dash(input_list):
    # Convert each argument to a string and join them with a dash
    return '-'.join(str(arg) for arg in input_list)


def delete_save_pro_folders(directory):
    '''
    删除指定路径下所有名为 save_pro 的文件夹及其内容。

    参数：
        directory (str): 要搜索的根目录路径。
    '''
    # 遍历目录下的所有文件夹
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            if dir_name == "save_pro":
                folder_path = os.path.join(root, dir_name)
                # 删除文件夹
                shutil.rmtree(folder_path)
                print(f"Deleted: {folder_path}")


#################bin operation
#################bin operation
#################bin operation
#################bin operation
#################bin operation
#################bin operation
#################bin operation
#################bin operation
def read_file(filename, x_data=0, shape_list=0, retry_delay=10, time_bool=False):
    """
    Reads data from a binary file into a provided tensor or numpy array.
    
    Parameters:
    - filename: str, the name of the file to read.
    - x_data: torch tensor or numpy array, the target array to store the data.
    - shape_list: list or tuple, the desired shape for the reshaped data.
    - retry_delay: int, the delay in seconds before retrying to acquire the lock.
    - time_bool: bool, whether to print the elapsed time for the reading operation.
    """
    
    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
    
    t_start = time.perf_counter()
    lock = FileLock(f"{filename}.lock")

    while True:  # Infinite retry loop
        try:
            # Try to acquire the lock with a timeout
            with lock.acquire(timeout=retry_delay):
                with open(filename, 'rb') as f:
                    arr_1d = np.frombuffer(f.read(), dtype=np.float32)
                break  # If the file is successfully read, exit the loop
        except Timeout:
            print(f"Waiting for the lock on {filename}... Retrying in {retry_delay} seconds.")
        except Exception as e:
            print(f"Error reading the file: {e}")
            return

    t_end = time.perf_counter()

    if time_bool:
        print(filename + " " + str((t_end - t_start)) + " s (read)")

    if torch.is_tensor(x_data):
        if shape_list != 0:
            x_data[:] = torch.from_numpy(arr_1d.reshape(shape_list)).to(x_data.device)
            # Ensure that we set x_data[:], not x_data, to maintain the reference
        else:
            x_data[:] = torch.from_numpy(arr_1d).to(x_data.device)
    else:
        if shape_list != 0:
            x_data[:] = arr_1d.reshape(shape_list)
        else:
            return arr_1d[:]

    return x_data

def bin_read(filename, x_data=0, shape_list=0, retry_delay=10, time_bool=False):
    
    """
    bin_read = lambda filename, x_data, shape_list, retry_delay, time_bool: read_file(filename, x_data, shape_list, retry_delay, time_bool);
    """
    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
        
    x_data = read_file(filename, x_data, shape_list, retry_delay, time_bool);
    
    return x_data


def read_file_improve(filename, x_data=0, shape_list=0, way=1, offset=0, time_bool=False):
    """
    filename: data name
    x_data: x_data=0
    shape_list: if shape_list==0, directly retrun array, in_arr = P.fread_file(file_list[i]); Otherwise, example: in_arr = np.zeros((nx,ny,nz),dtype=np.float32);    P.fread_file(in_file, in_arr, (nx,ny,nz));
    way: defalut 1, different way to read the data.
    offest:    uesless
    time_bool: True. output the time of read data
    #noted we must use x_data[:] rather than x_data =  1.0*np.array,  the first operation is set value, the second is to create a new array and the work domain is only the function.
    
    Example: in_arr = P.fread_file(file_list[i]);
    Example: in_arr      = np.zeros((nx,ny,nz),dtype=np.float32);    P.fread_file(in_file, in_arr, (nx,ny,nz));
    """
    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
        
    t_start = time.perf_counter();
    
    if way==0:
        with open(filename, 'rb') as f:
            arr_1d = np.fromfile(f, dtype=np.float32) 
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
        
    elif way==1:
        with open(filename, 'rb') as f:
            arr_1d = np.frombuffer(f.read(), dtype=np.float32)
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
        
        
    elif way==2:
        with open(filename, 'r+b') as f:
            # Memory-map the file using mmap.mmap()
            mmapped = mmap.mmap(f.fileno(), 0)
            # Create a 1D NumPy array from the memory-mapped buffer, using dtype=np.float32
            arr_1d = np.frombuffer(mmapped, dtype=np.float32)
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
    else:
        length=1
        for i in range(0, len(shape_list) ):
            length = length * shape_list[i];
            
        with open(filename, 'rb') as f:
            arr_1d = struct.unpack( 'f' * (length), f.read())
            # x_data[:] = 1.0*np.array(arr_1d, dtype=(np.float32)).reshape(shape_list) ;###it is note that c langauge is row first
    
    t_end = time.perf_counter();
    
    if time_bool:
        print(filename + " " +  str((t_end - t_start))   + " s (read)"  )
    
    if torch.is_tensor(x_data):
        if shape_list!=0:
            x_data[:] = torch.from_numpy( arr_1d.reshape(shape_list) ).to(x_data.device)  ;  #### we must set x_data[:], not x_data, there is the address
        else:
            x_data[:] = torch.from_numpy( arr_1d).to(x_data.device)
        
    else:
        if shape_list!=0:
            x_data[:] = arr_1d.reshape(shape_list) ;
        else:
            return arr_1d[:]
    

def read_file_list(filename, x_data, shape_list, way=0, thread_num=24, time=False):
    
    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
        
    nz_number = len(filename);
    
    if (nz_number%thread_num) ==0:
        number = nz_number//thread_num
    else:
        number = nz_number//thread_num + 1
    
    for i in range(0, number):
        file_list = filename[i*thread_num:(i+1)*thread_num]
        threads_list = []
        
        for t in range(0, thread_num):
            iz = i*thread_num + t
            
            if iz < len(filename):
                file = file_list[t]
                thr  = threading.Thread(target=(read_file), args=(file, x_data[:,:,:,iz], shape_list, 0) ); #fread_file(filename, x_data, shape_list, way=0)
                
                threads_list.append(thr)
                
        for thr in threads_list:
            thr.start()
        for thr in threads_list:
            thr.join();


def write_txt(filename, 
              output_some, 
              w_type='a+', 
              mkdir_path=False,
              print_bool=False):
    
    if mkdir_path:
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
    
    # Open the file and write data
    with open(filename, w_type) as fp:  # Using 'with' for better resource management
        print(output_some, file=fp);
    
    # Print to console if print_bool is True
    if print_bool:
        print(output_some)


def write_txt_old(filename, output_some, w_type='a+', print_bool=False):
    data=open(filename, w_type); ##'w+' 'a+'
    print(output_some, file=data)
    data.close()
    if print_bool:
        print(output_some);

def write_np_txt(filename, arr, delimiter='\n', fmt='%.5f', w_type="w+"):
    with open(filename, w_type) as f:
        np.savetxt(f, np.asarray(arr), delimiter=delimiter, fmt=fmt);

def write_array_txt(filename, arr, delimiter='\n', fmt='%.5f', w_type="a+"):
    
    if  isinstance(arr, cp.ndarray):
        arr2 = cp.asnumpy(arr)
    elif isinstance(arr, torch.Tensor):
        arr2 = arr.detach().cpu().numpy()
    else:
        arr2 = arr
    
    with open(filename, w_type) as f:
        np.savetxt(f, np.asarray(arr2), delimiter=delimiter, fmt=fmt);




def bin_name(input_arr):
    
    # if isinstance(input_arr, torch.Tensor):
    #     input_arr = input_arr.detach().cpu().numpy()
        
    shape = list(input_arr.shape)
    file=""
    for i, j in enumerate(shape): 
        if i < len(shape)-1:
            file = file + str(j) + "-"
        else:
            file = file + str(j)
    return file


def write_file(filename, x_data, time_bool=False):
    '''
    write an array for [nz, ny, nx], for x first demension
    write an array for [nz, ny, nx], for x first demension
    write an array for [nz, ny, nx], for x first demension
    '''
    if isinstance(x_data, torch.Tensor):
        x_data = x_data.detach().cpu().numpy()
        
    if isinstance(x_data, cp.ndarray):
        x_data = cp.asnumpy(x_data)
    
    t_start = time.perf_counter();
    
    if len(filename)!=0:
        with open(filename, 'wb') as f:
            x_data.astype(np.float32).tofile(f)###it is note that c langauge is row first
    else:
        filename = bin_name(x_data);
        with open(filename, 'wb') as f:
            x_data.astype(np.float32).tofile(f)###it is note that c langauge is row first
    
    
    t_end = time.perf_counter();
    if time_bool:
        print(filename + " " + str((t_end - t_start))    + " s (write)"    )


'''array'''
'''array'''
'''array'''
def unravel_index(indices, shape, module):
    """
    Generalized unravel_index for NumPy, CuPy, and PyTorch.
    
    Parameters:
    - indices: Linear indices (int, list, or tensor/array).
    - shape: Shape of the array or tensor.
    - module: The module to use ('np', 'cp', or 'torch').
    
    Returns:
    - tuple: Multi-dimensional indices as per the specified module.
    """
    if module == np:
        return np.unravel_index(indices, shape)
    elif module == cp:
        return cp.unravel_index(indices, shape)
    elif module == torch:
        if isinstance(indices, int):  # Handle scalar indices
            indices = torch.tensor([indices], dtype=torch.long)
        indices = indices.long()  # Ensure the correct dtype
        shape = torch.tensor(shape, dtype=torch.long, device=indices.device)
        
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = indices // dim
        coord = tuple(reversed(coord))
        
        if len(coord) == 1:  # Return as scalar if single dimension
            return coord[0]
        return coord
    else:
        raise ValueError("Unsupported module. Choose 'np', 'cp', or 'torch'.")


def array_conti(in_arr):

    return in_arr.flags['C_CONTIGUOUS']

def array_info(in_arr, file="array_info", print_bool=False):

    module  = get_module_type(in_arr)

    # Get the flat index of max, min, and abs max values
    max_idx     = module.argmax(in_arr)
    abs_max_idx = module.argmax(module.abs(in_arr))
    min_idx     = module.argmin(in_arr)
    abs_min_idx = module.argmin(module.abs(in_arr))

    # Convert the flat index to N-dimensional coordinates using unravel_index
    max_coords      = unravel_index(max_idx, in_arr.shape, module=module)
    abs_max_coords  = unravel_index(abs_max_idx, in_arr.shape, module=module)
    min_coords      = unravel_index(min_idx, in_arr.shape, module=module)
    abs_min_coords  = unravel_index(abs_min_idx, in_arr.shape, module=module)

    # Formatting the output strings
    file0 = "\n{} allocate on {}\n".format(file, module.__name__)
    file1 = "{} shape is {}\n".format(file, in_arr.shape)
    file2 = "{} max is {} at index {}\n".format(file, module.max(in_arr), max_coords)
    file3 = "{} max abs is {} at index {}\n".format(file, module.max(module.abs(in_arr)), abs_max_coords)
    file4 = "{} min is {} at index {}\n".format(file, module.min(in_arr), min_coords)
    file5 = "{} min abs is {} at index {}\n".format(file, module.min(module.abs(in_arr)), abs_min_coords)
    file6 = "{} mean is {}\n".format(file, module.mean(in_arr))
    file66 = "{} mean of abs is {}\n".format(file, module.mean(module.abs(in_arr)))
    file7 = "{} std is {}\n".format(file, module.std(in_arr))
    file77 = "{} std of abs is {}\n".format(file, module.std(module.abs(in_arr)))
    # Handle optional attributes such as device and flags
    try:
        file8 = "device is {}\n".format(in_arr.device)
    except AttributeError:
        file8 = "there is no device\n"

    try:
        file9 = "dtype is {}\n".format(in_arr.dtype)
    except AttributeError:
        file9 = "there is no dtype\n"

    try:
        file10 = "in_arr.flags['C_CONTIGUOUS'] is {}\n".format(in_arr.flags['C_CONTIGUOUS'])
    except AttributeError:
        file10 = "there is no .flags\n"
        
    try:
        file11 = "array_memory(in_arr) is {}\n".format(  array_memory(in_arr) )
    except AttributeError:
        file11 = "there is no array_memory\n"

    # Print if requested
    if print_bool:
        print(file0 + file1 + file2 + file3 + file4 + file5 + file6 + file7 + file8 + file9 + file10 + file11)

    # Return the concatenated string
    return file0 + file1 + file2 + file3 + file4 + file5 + file6 + file66 + file7+ file77 + file8 + file9 + file10 + file11


def array_memory(arr):
    """
    Calculate the memory usage of the array and convert it to MB.
    Supports NumPy, CuPy, and PyTorch arrays/tensors.
    
    :param arr: The array (NumPy, CuPy, or PyTorch tensor)
    :return: The memory usage in MB
    """
    # Check the type of array and calculate its memory size
    if isinstance(arr, np.ndarray):
        total_size = arr.nbytes
    elif isinstance(arr, cp.ndarray):
        total_size = arr.nbytes
    elif isinstance(arr, torch.Tensor):
        total_size = arr.element_size() * arr.nelement()  # Calculate memory size for PyTorch tensor
    else:
        raise TypeError("Unsupported array type. Please use NumPy, CuPy, or PyTorch arrays.")
    
    total_size_mb = total_size / (1024 ** 2)  # Convert bytes to MB
    return total_size_mb

def array_squeeze(arr):
    
    if isinstance(arr, np.ndarray):
        return np.squeeze(arr)  # NumPy
    elif isinstance(arr, torch.Tensor):
        return arr.squeeze()    # PyTorch
    elif isinstance(arr, cp.ndarray):
        return cp.squeeze(arr)  # CuPy
    else:
        raise TypeError("Unsupported array type. Please use NumPy, PyTorch, or CuPy arrays.")




'''compile        cupy  '''
'''compile        cupy  '''
'''compile        cupy  '''
'''compile        cupy  '''
'''compile        cupy  '''
def cuda_record_time_dict_old(time_dict, key, start_event, end_event):
    # ensure the end of event 
    end_event.synchronize()
    # start_event to end_event ( ms )
    elapsed_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    elapsed_time_s  = elapsed_time_ms / 1000.0  ###s
   
    if key in time_dict:
        time_dict[key] += elapsed_time_s
    else:
        time_dict[key]  = elapsed_time_s


def cuda_record_time_dict(time_dict, key, start_event, end_event, cpu_start_time=None, cpu_end_time=None):
    # ensure the end of event 
    end_event.synchronize()
    # start_event to end_event (GPU time in ms)
    elapsed_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    elapsed_time_s  = elapsed_time_ms / 1000.0  # GPU time in seconds

    # Update GPU time in time_dict
    if key in time_dict:
        time_dict[key] += elapsed_time_s
    else:
        time_dict[key]  = elapsed_time_s
    
    # Optionally record CPU time if cpu_start_time is provided
    if cpu_start_time is not None and cpu_end_time is not None:
        # CPU time in seconds
        elapsed_time = cpu_end_time - cpu_start_time  
        
        if key+'-cpu_time' in time_dict:
            time_dict[key+'-cpu_time'] += elapsed_time_s
        else:
            time_dict[key+'-cpu_time']  = elapsed_time_s
        



def compile_cupy_kernels(cuda_code=None, 
                         ptx_file=None, 
                         log_file="cupy_kernels.txt", 
                         print_bool=False
                         ):
    """
    https://docs.cupy.dev/en/stable/reference/generated/cupy.RawModule.html
    
    https://www.sidmittal.eu/blogs/gpu-articles/cuPyLoadPTX.html
    
    Compiles CUDA code or loads a PTX file using CuPy and returns a dictionary of kernel functions.

    Parameters:
    cuda_code (str): The CUDA code as a string (optional).
    ptx_file (str): Path to the PTX file (optional).

    Returns:
    dict: A dictionary where keys are kernel function names and values are compiled CuPy kernel functions.
    
    class cupy.RawModule(unicode code=None, *, unicode path=None, tuple options=(), unicode backend=u'nvrtc', bool translate_cucomplex=False, bool enable_cooperative_groups=False, name_expressions=None, bool jitify=False)
    

User-defined custom module.

This class can be used to either compile raw CUDA sources or load CUDA modules (*.cubin, *.ptx). This class is useful when a number of CUDA kernels in the same source need to be retrieved.

For the former case, the CUDA source code is compiled when any method is called. For the latter case, an existing CUDA binary (*.cubin) or a PTX file can be loaded by providing its path.

CUDA kernels in a RawModule can be retrieved by calling get_function(), which will return an instance of RawKernel. (Same as in RawKernel, the generated binary is also cached.)
    """
    if ptx_file:

        cupy_module = cp.RawModule(path=ptx_file)

        # Extract function names from PTX by looking for ".entry"
        with open(ptx_file, 'r') as file:
            ptx_file_open = file.read()
       
            function_names = re.findall(r'\.entry\s+([_a-zA-Z0-9]+)\s*\(', ptx_file_open, re.DOTALL)


    elif cuda_code:
        # Compile the CUDA code if no PTX file is provided
        cupy_module = cp.RawModule(code=cuda_code)

        # Extract function names from the CUDA code using regex
        function_names = re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)', cuda_code)
    
    else:
        raise ValueError("Either 'cuda_code' or 'ptx_file' must be provided.")

    # Create a dictionary to store the kernel functions
    wave_kernel_dict = {}
    for func in function_names:
        wave_kernel_dict[func] = cupy_module.get_function(func)

    # Optionally print the available kernel functions
    output_some="Available kernel functions:{}".format( list(wave_kernel_dict.keys()) );
    write_txt(log_file, output_some, print_bool=print_bool)
    
    return wave_kernel_dict




def compile_cupy_kernels_code(cuda_code):
    """
    Compiles CUDA code using CuPy and returns a dictionary of kernel functions.

    Parameters:
    cuda_code (str): The CUDA code as a string.

    Returns:
    dict: A dictionary where keys are kernel function names and values are compiled CuPy kernel functions.
    """
    # Compile the CUDA code with CuPy
    cupy_wave_kernel = cp.RawModule(code=cuda_code)

    # Use regular expressions to find all kernel function names
    function_names = re.findall(r'extern\s+"C"\s+__global__\s+void\s+(\w+)', cuda_code)

    # Create a dictionary to store the kernel functions
    wave_kernel_dict = {}
    for func in function_names:
        wave_kernel_dict[func] = cupy_wave_kernel.get_function(func)

    # Optionally print the available kernel functions
    print("Available kernel functions:", list(wave_kernel_dict.keys()))

    return wave_kernel_dict



###
def get_module_type(arr):
    
    if isinstance(arr, list):
        raise  TypeError("get_module_type is error, the input is list")
    
    if isinstance(arr, np.ndarray):
        return np
    elif isinstance(arr, cp.ndarray):
        return cp
    elif isinstance(arr, torch.Tensor):
        return torch
    else:
        raise  TypeError("get_module_type is error for np cp torch")
        
        
def get_device_ini(gpu_id=0):
    '''
    return device, cp_device
    '''
    if gpu_id == 0:
        cp.cuda.Device().use()  # 默认, 因为可以在 sbatch 中运行代码
        device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        
    else:
        cp.cuda.Device(gpu_id).use()
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    cp_device = cp.cuda.Device()
    print("cp_device1:", cp_device)
    print("torchdevice1:", device)
    
    return device, cp_device

# device, cp_device = get_device_ini(0)


def get_seed(seed=100):
    # Python built-in random module
    random.seed(seed)
    
    # Python hash seed (for reproducible hashing)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # NumPy global random generator (for parallel or newer APIs)
    np.random.default_rng(seed)
    
    # PyTorch random seeds
    torch.manual_seed(seed)  # For CPU
    torch.cuda.manual_seed(seed)  # For the current GPU device
    torch.cuda.manual_seed_all(seed)  # For all available GPUs
    
    # CuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for full reproducibility
    torch.backends.cudnn.enabled = True
    
    # Force PyTorch to use deterministic algorithms where available
    if hasattr(torch, 'use_deterministic_algorithms'):
        # torch.use_deterministic_algorithms(True)
        torch.use_deterministic_algorithms(False)
    else:
        print( torch.__version__ )



def get_operator_info(operator, full_code=False, lines_number=6):
    """
    Parameters:
        operator (function): obtain full code and name of an operator
    Returns:
        if full_code:
            return operator_info
        else:
            if operator_info["name"] == None:
                 return operator_info["definition"].rstrip("\n");
            else:
                if operator_info["name"] == "<lambda>":
                    return operator_info["definition"].rstrip("\n");
                else:
                    return operator_info["definition"].rstrip("\n");
    """
    operator_info = {
        "name": operator.__name__ if hasattr(operator, "__name__") else None,
        "definition": None
    }
    
    try:
        # 尝试使用 inspect 获取源代码
        source_code = inspect.getsource(operator)
        operator_info["definition"] = source_code
    except (OSError, TypeError, tokenize.TokenError):
        # 如果是 lambda 或者内置函数，尝试用 dill 获取源代码
        try:
            source_code = dill.source.getsource(operator)
            operator_info["definition"] = source_code
        except Exception:
            operator_info["definition"] = "Source code not available."
    
    if full_code:
        return operator_info
    else:
        definition = operator_info["definition"]
        
        if definition is not None:
            lines = definition.splitlines()
            return "\n".join(lines[:lines_number]) + ("..." if len(lines) > lines_number else "")
        
        return definition
        
        # if operator_info["name"] == None:
        #      return operator_info["definition"].rstrip("\n");
        # else:
        #     if operator_info["name"] == "<lambda>":
        #         return operator_info["definition"].rstrip("\n");
        #     else:
        #         return operator_info["definition"].rstrip("\n");
        




##################
##################
##################
##################class  
##################
##################
##################
class class_:
    pass

def class_tmp():
    def __init__(self, 
                 
                 dims=[1, 1, 1],
                 
                 dims_interval=[1, 1, 1],

                 log_file="class_tmp.txt"
                ):
        
        self.readme     = {}; ##I can record something, when I save it
        
        self.name       = self.__class__.__name__
        
        self.log_file   = log_file

        # Initialize dimensions and intervals
        self.dims = list( map(np.int32, dims) )
        self.dims_interval = list( map(np.float32, dims_interval) )
        
        
        descriptions = {
                        'dims': "operator dimension",
                        
                        }
        
        ##step1    ##recording calss ini value     WR.
        class_dict_ini_log(log_file, self.__class__, w_type="w");  
        
        ##step2    # Populate the dict with variable names, descriptions, and values WR.
        self.dict                   = {}
        class_dict_description(self, descriptions);

        ##step3    ##record the final log WR.
        class_dict_log_file(log_file, self.dict, w_type="a")
        
class Class_Placeholder:
        """Simple placeholder class to use when a class definition is missing."""
        def __init__(self, *args, **kwargs):
            pass        

def class_write(instance_func, 
                filename='data_dict.pkl', 
                deepcopy=False,
                func_time=True):
    """
    Save the class instance_func to a file, converting CuPy arrays to NumPy for saving.
    Ignore any non-pickleable attributes like module and lambda functions.
    """
    if func_time:
        t_start = time.perf_counter();
    
    
    # 创建深拷贝的实例，避免修改原始对象
    # 上述代码会修改原始类实例的属性
    if deepcopy:
        instance = copy.deepcopy(instance_func)
    else:
        instance  =              instance_func
        
    # 遍历属性，将 CuPy 和 PyTorch 数据转换为 NumPy
    for name, value in instance.__dict__.items():
        print(f"Attribute: {name}, Type: {type(value)}")
        
        if isinstance(value, cp.ndarray):
            print(f"Converting CuPy array {name} to NumPy.\n")
            instance.__dict__[name] = cp.asnumpy(value)
        elif isinstance(value, torch.Tensor):
            print(f"Converting PyTorch tensor {name} to NumPy.\n")
            instance.__dict__[name] = value.to('cpu').detach().numpy()
        elif callable(value):  # 忽略所有函数和方法
            print(f"Ignoring function {name} during saving.\n")
            instance.__dict__[name] = None
        elif isinstance(value, (type(cp), type(np), type(torch))):  
            print(f"Ignoring module {name} during saving.\n")
            instance.__dict__[name] = None
        else:
            print(f"Keeping attribute {name} as is.\n")


    # 保存对象
    with open(filename, 'wb') as f:
        pickle.dump(instance, f)
        print('Successfully saved the instance, converting arrays to NumPy.\n')
        
    
    if func_time:
        t_end = time.perf_counter();
        print("Time for write {}: {} s\n".format(filename, t_end-t_start) );

def class_read(filename, convert_to_cupy=False, retry_delay=10, func_time=True):
    """
    Load the class instance from a file. Optionally convert NumPy arrays back to CuPy.

    Parameters:
    - filename: The name of the file to load the instance from.
    - convert_to_cupy: Whether to convert NumPy arrays to CuPy (default is False).
    
    Returns:
    - The loaded class instance.
    """
    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
    
    lock_filename = filename + ".lock"  # Define the lock file name
    lock = FileLock(lock_filename)  # Create a FileLock object
    
    if func_time:
        t_start = time.perf_counter();
        
    while True:  # Infinite retry loop
           try:
               # Try to acquire the lock
               with lock.acquire(timeout=retry_delay):  
                   with open(filename, 'rb') as f:
                       instance = pickle.load(f)
                   break  # Exit the loop after successfully reading the file
           except Timeout:
               # Log and retry after a delay
               print(f"Task is waiting: File {filename} is locked. Retrying in {retry_delay} seconds...")
           except Exception as e:
               raise RuntimeError(f"Error while reading file {filename}: {e}")
    
    # Optionally convert NumPy arrays back to CuPy
    if convert_to_cupy:
        for name, value in instance.__dict__.items():
            if isinstance(value, np.ndarray):
                instance.__dict__[name] = cp.asarray(value)
    
    if func_time:
        t_end = time.perf_counter();
        print("Time for read {}: {} s".format(filename, t_end-t_start) );
        
    return instance

def class_read_old(filename, convert_to_cupy=False, func_time=True):
    """
    Load the class instance from a file. Optionally convert NumPy arrays back to CuPy.

    Parameters:
    - filename: The name of the file to load the instance from.
    - convert_to_cupy: Whether to convert NumPy arrays to CuPy (default is False).
    
    Returns:
    - The loaded class instance.
    """

    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
        
    if func_time:
        t_start = time.perf_counter()

    with open(filename, 'rb') as f:
        try:
            instance = pickle.load(f)
        except ModuleNotFoundError as e:
            print(f"Warning: {e}. Using placeholder class for missing module.")
            instance = Class_Placeholder()

    # Optionally convert NumPy arrays back to CuPy
    if convert_to_cupy and hasattr(instance, '__dict__'):
        for name, value in instance.__dict__.items():
            if isinstance(value, np.ndarray):
                instance.__dict__[name] = cp.asarray(value)
    
    if func_time:
        t_end = time.perf_counter()
        print(f"Time for read {filename}: {t_end - t_start:.3f} s")
    
    return instance







##################
##################
##################
##################dict    
##################
##################
##################



def dict_empty_keys( keys = ['p', 'vx', 'vz', 'vy', 'p_dt', 'p_dt_dt'] ):
    
    empty_dict = {key: None for key in keys}
    
    return empty_dict

def dict_write_as_txt(filename='data_dict.txt', 
                      data_dict={}, 
                      w_type="a+", 
                      convert_to_list=True, 
                      print_bool=False):
   
    with open(filename, w_type) as file:
        for key, value in data_dict.items():
            
            if convert_to_list:
                # Check if the value is a CuPy array
                if isinstance(value, cp.ndarray):
                    value = cp.asnumpy(value).tolist()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()  # Convert NumPy array to list  
            
            file.write(f"Key: {key}, Value: {value}\n\n")
            if print_bool:
                print(f"Key: {key}, Value: {value}\n\n");
            
            

def dict_str_key_as_txt(filename='data_dict.txt', data_dict={}, w_type="w+", convert_to_list=True):
   '''
   write only str into log_file
   '''
   with open(filename, w_type) as file:
       for key, value in data_dict.items():
            if isinstance(key, str):
                file.write("{}: {}\n".format(key, value));
                

def dict_write(filename='data_dict.npz', data_dict={}, np_or_pickle=True, func_time=True):

    if func_time:
        t_start = time.perf_counter()
        

    for name, value in data_dict.items():
        if   isinstance(value, cp.ndarray):
            data_dict[name] = cp.asnumpy(value)
        elif isinstance(value, torch.Tensor):  
            data_dict[name] = value.to('cpu').detach().numpy();
        
    # check the key keys is_or_not  str, 
    keys_are_strings = all(isinstance(k, str) for k in data_dict.keys())

    if np_or_pickle:
        if not keys_are_strings:
            data_dict2 = {str(k): v for k, v in data_dict.items()}
            np.savez(filename, **data_dict2);
        else:
            data_dict2 = data_dict
            np.savez(filename, **data_dict2);
    
    else:
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)
            
    if func_time:
        t_end = time.perf_counter()
        print(f"Time for dict_write {filename}: {t_end - t_start:.3f} s")


def dict_read(filename='data_dict.npz', np_or_pickle=True, retry_delay=10, func_time=True):
    """
    Safely loads a dictionary from a file, supporting both NumPy (.npz) and Pickle (.pkl) formats.
    Includes file locking to avoid conflicts in multi-task environments and infinite retry functionality.

    Parameters:
    - filename: The file name to load (default: 'data_dict.npz').
    - np_or_pickle: Whether to use NumPy for loading the file (True: .npz, False: .pkl).
    - retry_delay: Time (in seconds) to wait before retrying if the file is locked (default: 10 seconds).
    - func_time: Whether to measure the file loading time (default: True).

    Returns:
    - The loaded dictionary.
    """
    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
        
    lock_filename = filename + ".lock"  # Path to the lock file
    lock = FileLock(lock_filename)

    if func_time:
        t_start = time.perf_counter()

    while True:  # Infinite retry loop
        try:
            # Attempt to acquire the file lock
            with lock.acquire(timeout=retry_delay):
                if np_or_pickle:
                    # Use NumPy to load a .npz file
                    loaded_data = np.load(filename)
                    loaded_dict = {ast.literal_eval(key): loaded_data[key] for key in loaded_data.files}
                else:
                    # Use Pickle to load a .pkl file
                    with open(filename, 'rb') as f:
                        loaded_dict = pickle.load(f)
                break  # Successfully loaded the file, exit the loop
        except Timeout:
            print(f"File {filename} is locked. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            raise RuntimeError(f"Error occurred while reading file {filename}: {e}")

    if func_time:
        t_end = time.perf_counter()
        print(f"Time for dict_read {filename}: {t_end - t_start:.3f} s")

    return loaded_dict


def dict_write_pkl(filename='data_dict.pkl', data_dict={}, func_time=True):
    """
    将数据字典保存到指定文件。支持 CuPy 数组、PyTorch 张量、NumPy 数组，以及包含这些类型的列表。
    对于纯 Python 列表和纯数字，直接保存。
    """
    
    if func_time:
        t_start = time.perf_counter()
    
    processed_dict = {}

    #data_dict.items() 是迭代的副本, 提供了字典的键值对，并不会直接修改原始字典。
    for name, value in data_dict.items():
        if isinstance(value, cp.ndarray):
            processed_dict[name] = cp.asnumpy(value)
        elif isinstance(value, torch.Tensor):
            processed_dict[name] = value.to('cpu').detach().numpy()
        elif isinstance(value, np.ndarray):
            processed_dict[name] = value
        elif isinstance(value, list):
            # 处理列表中的每个元素
            processed_dict[name] = [
                cp.asnumpy(item) if isinstance(item, cp.ndarray) else
                item.to('cpu').detach().numpy() if isinstance(item, torch.Tensor) else
                item if isinstance(item, (np.ndarray, int, float)) else item
                for item in value
            ]
        elif isinstance(value, (int, float)):
            processed_dict[name] = value
        else:
            processed_dict[name] = value  # 其他类型保持不变

    with open(filename, 'wb') as f:
        pickle.dump(processed_dict, f)
        
    if func_time:
        t_end = time.perf_counter()
        print(f"Time for dict_write_pkl {filename}: {t_end - t_start:.3f} s")


def dict_read_pkl(filename='data_dict.pkl', map_location='cpu', keep_torch_device=False, retry_delay=10, func_time=True):
    """
    Loads a data dictionary from the specified file. Attempts to use `pickle.load` first, and if that fails, falls back to using `torch.load` with the specified `map_location`.
    Supports NumPy arrays, CuPy arrays, PyTorch tensors, and their lists.
    
    Parameters:
    - filename: str, the name of the file to load.
    - map_location: str or torch.device, the device to load the tensors to.
    - keep_torch_device: bool, whether to keep the original device of the tensors.
    - retry_delay: int, the delay in seconds before retrying to acquire the lock.
    - func_time: bool, whether to print the elapsed time for the loading operation.
    
    Returns:
    - loaded_dict: dict, the loaded data dictionary with appropriate data types.
    """
    
    if not os.path.isfile(filename):
        print(f"file {filename} not exist")
        sys.exit(1) 
        
    lock_filename = filename + ".lock"  # Path to the lock file
    lock = FileLock(lock_filename)
    
    if func_time:
        t_start = time.perf_counter()
        
    while True:  # Infinite retry loop
        try:
            with lock.acquire(timeout=retry_delay):  # Try to acquire the lock, wait for retry_delay seconds
                with open(filename, 'rb') as f:
                    try:
                        # Attempt to use pickle.load first
                        loaded_dict = pickle.load(f)
                    except Exception as e:
                        print(f"pickle.load failed, attempting torch.load: {e}")
                        f.seek(0)  # Reset file pointer
                        try:
                            loaded_dict = torch.load(f, map_location=map_location)
                        except Exception as torch_e:
                            raise RuntimeError(f"torch.load also failed, unable to load file: {torch_e}")
                break  # Exit loop after successful load
        except Timeout:
            print(f"File {filename} is locked, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except Exception as e:
            raise RuntimeError(f"Error occurred while loading file {filename}: {e}")

    def process_item(item):
        """Helper function: processes each item based on its data type"""
        if isinstance(item, torch.Tensor) and not keep_torch_device:
            return item.to(map_location)
        elif isinstance(item, cp.ndarray):
            return cp.asnumpy(item)
        return item

    # Process each item in the loaded data
    for name, value in loaded_dict.items():
        if isinstance(value, list):
            loaded_dict[name] = [process_item(item) for item in value]
        else:
            loaded_dict[name] = process_item(value)
            
    if func_time:
        t_end = time.perf_counter()
        print(f"Time for dict_read_pkl {filename}: {t_end - t_start:.3f} s")

    return loaded_dict



def dict_data_sxsysz_as_array(obs_shot_dict, time_bool=True):
    
    if time_bool:
        t_start = time.perf_counter();
        
        
    
    sorted_dict = OrderedDict(sorted(obs_shot_dict.items(), key=lambda item: item[0]));
    
    sx_list = []
    sy_list = []
    sz_list = []
    
    total_arr = next(iter(sorted_dict.values()));
    total_arr = total_arr[np.newaxis, ...]
    
    for ishot, (key, value) in enumerate(sorted_dict.items()):
        sx, sy, sz = key
        sx_list.append(sx)
        sy_list.append(sy)
        sz_list.append(sz)
        
        if ishot >=1:
            data_arr  = value[np.newaxis, ...]
            
            total_arr = np.concatenate([total_arr, data_arr], axis=0);
    
    if time_bool:
        t_end = time.perf_counter();
        print( "dict_data_sxsysz_as_array:"  + str( (t_end - t_start) )+ " s"     )
    
    
    return np.ascontiguousarray(total_arr), np.array(sx_list), np.array(sy_list), np.array(sz_list)
    

def dict_obs_imshow(obs_shot_dict, output_path, write_bin=False, plot_min_scale=0.5, plot_max_scale=0.5, d1=0.01, d2=0.001):
    
    for ishot, (s_key, obs_shot_d) in enumerate( obs_shot_dict.items() ):
     
        sx, sy, sz  = s_key
        
        sxsysz_name = "sx-" + str(sx) + "-sy-" + str(sy) + "-sz-" + str(sz);
        
        plot_min     = plot_min_scale * obs_shot_d.min().item();
        plot_max     = plot_max_scale * obs_shot_d.max().item();
        
        prefix_name  = output_path + "obs-" +  sxsysz_name + "-" +  bin_name(  obs_shot_d.T  );
        data_name    = prefix_name + ".bin";
        eps_name     = prefix_name + ".eps";
        
        PF.imshow( obs_shot_d, d1=d1, d2=d2, ylabel='Times(s)', vmin=plot_min, vmax=plot_max, output_name=eps_name);
        if write_bin:
            write_file( data_name, obs_shot_d.T );


def dict_obs_sort(obs_shot_dict, sort_order=True):
    
    obs_shot_dict_sorted = dict(
       sorted( obs_shot_dict.items(), key=lambda item: item[0], reverse=not sort_order )
   )
        
    return obs_shot_dict_sorted

def dict_value_to_list(obs_shot_dict):
    
    obs_shot_dict_list = [value for key, value in obs_shot_dict.items()];
    
    return obs_shot_dict_list

def dict_arr_to_list(obs_shot_dict):
    
    obs_shot_dict_list = [value for key, value in obs_shot_dict.items()];
    
    return obs_shot_dict_list

def dict_to_cupy(obs_shot_dict):

    for key, value in obs_shot_dict.items():
        obs_shot_dict[key]  = cp.asarray(value) ;


def class_dict_ini_log(log_file, cls, w_type="w"):
    """
    Logs the initial parameter values (and their defaults) for the constructor (__init__)
    of the given class.

    Parameters:
    - log_file: The file where initial parameter values should be logged.
    - cls: The class whose __init__ method's signature will be inspected.
    """
    if len(log_file) != 0:
        # Get the signature of the __init__ method of the class
        signature = inspect.signature(cls.__init__)
        params    = signature.parameters
        
        # Open the log file in write mode
        with open(log_file, w_type) as log:
            
            log.write("Class name: {}\n".format(cls.__name__))
            
            for param_name, param in params.items():
                # Format the default parameter in formation
                file_content = "default para {}: {}\n\n".format(param_name, param)
                log.write(file_content)


def class_dict_description(instance, descriptions=None):
    """
    A global function to populate a dictionary with variable names, descriptions, and values
    for any class instance.
    
    Parameters:
    - instance: The class instance whose attributes will be recorded in a dict.
    - descriptions: A dictionary containing the descriptions of certain attributes.
                    If not provided, 'No description available' will be used.
    """
    if descriptions is None:
        descriptions = {}

    # Initialize the instance's dict attribute if it doesn't exist
    if not hasattr(instance, 'dict'):
        instance.dict = {}
    
    # Iterate over the instance's attributes and populate the dict
    for name, value in instance.__dict__.items():
        if name != 'dict':  # Skip the dict attribute itself
            
            if isinstance(value, (cp.ndarray, np.ndarray)):
                # If it's an array, store its metadata (shape, dtype, size)
                try:
                    array_info_file = array_info(value, name)
                except:
                    array_info_file = "There is no array_info";
                    
                instance.dict[name] = {
                    'des': descriptions.get(name, 'No description available'),
                    'val': array_info_file,
                    'dtype': value.dtype,
                } #torch.Tensor
            
            elif callable(value):
                instance.dict[name] = {
                    'des': descriptions.get(name, 'No description available'),
                    'val': "function",
                    'dtype': "function",
                }
            
            else:
                try:
                    dtype = value.dtype
                except AttributeError:
                    dtype = "There is no dtype"
                
                # If it's not an array, store the actual value
                instance.dict[name] = {
                    'des': descriptions.get(name, 'No description available'),
                    'val': value,
                    'dtype': dtype,
                }


def class_dict_log_file(log_file, data_dict, w_type="a"):
    """
    A global function to log the contents of a dictionary to a file.

    Parameters:
    - log_file: The name of the file where the dictionary content will be logged.
    - data_dict: The dictionary containing the attributes to log. Each item in the
                 dictionary should be structured as {'val': value, 'des': description}.
    """
    if len(log_file) != 0:
        with open(log_file, w_type) as log:
            for key, value in data_dict.items():
                
                file_content = "    key:{}\n    des:{}\n    val:{}\n    dtype:{}\n\n\n".format(key, value['des'], value['val'], value['dtype'])
                log.write(file_content)





def dict_list_element_types_check(func_dict, log_file="log.txt", w_type='a'):

    if    isinstance(func_dict, dict):
        input_dict = func_dict;
        
    elif isinstance(func_dict, list):
        input_dict = list_to_dict(func_dict);

    
    with open(log_file, w_type) as fopen:

        fopen.write("dict_list_element_types_check\n")    ;    

        if isinstance(input_dict, dict):
            for key, item in input_dict.items():
                
                # NumPy array
                if isinstance(item, np.ndarray):
                    fopen.write("Key: {}, Type: NumPy Array, Dtype: {}\n".format(key, item.dtype))
                    fopen.write("Key: {}, item.flags['C_CONTIGUOUS']: {}\n".format(key, item.flags['C_CONTIGUOUS']))
                    fopen.write("Key: {}, item.shape(): {}\n\n".format(key, item.shape))
                
                # CuPy array
                elif isinstance(item, cp.ndarray):
                    fopen.write("Key: {}, Type: CuPy Array, Dtype: {}\n".format(key, item.dtype))
                    fopen.write("Key: {}, item.flags['C_CONTIGUOUS']: {}\n".format(key, item.flags['C_CONTIGUOUS']))
                    fopen.write("Key: {}, item.shape(): {}\n\n".format(key, item.shape))
                
                
                # NumPy float and int types
                elif isinstance(item, np.generic):
                    fopen.write("Key: {}, item:{}, Type: numpy.dtype:{}\n\n".format(key, item,  item.dtype))
                    
                
                # CuPy float and int types
                elif isinstance(item, cp.generic):
                    fopen.write("Key: {}, item:{}, Type: CuPy.dtype:{}\n\n".format(key, item,  item.dtype))
                

                # Python native float
                elif isinstance(item, float):
                    fopen.write("Key: {}, item:{}, Type: Python float64\n\n".format(key, item))
                
                # Python native int
                elif isinstance(item, int):
                    fopen.write("Key: {}, item:{}, Type: Python int64\n\n".format(key, item))
                
                # Nested dictionary
                elif isinstance(item, dict):
                    fopen.write("Key: {}, item: {}, Python dict\n\n".format(key, item))
                    fopen.write("Contents of the nested dict:\n\n")
                    dict_list_element_types_check(item, log_file, w_type)
                
                # Python list
                elif isinstance(item, list):
                    fopen.write("Key: {}, item:{}, Type: Python list, Length: {}\n\n".format(key, item, len(item)))
                    for idx, sub_item in enumerate(item):
                        fopen.write("List item {}:, Value:{}, Type: {}\n\n".format(idx, sub_item, type(sub_item).__name__))
                
                
                # Python function
                elif isinstance(item, types.FunctionType):
                    fopen.write("Key: {}, item:{}, Type: Python function\n\n".format(key, item))
                
                
                
                
                
                elif isinstance(item, str):
                    fopen.write("Key: {}, item: '{}', Type: Python str\n".format(key, item))

                elif isinstance(item, tuple):
                    fopen.write("Key: {}, item: {}, Type: Python tuple, Length: {}\n".format(key, item, len(item)))
                
                elif isinstance(item, set):
                    fopen.write("Key: {}, item: {}, Type: Python set, Length: {}\n".format(key, item, len(item)))
                
                elif isinstance(item, bool):
                    fopen.write("Key: {}, item: {}, Type: Python bool\n".format(key, item))
    
                elif isinstance(item, complex):
                    fopen.write("Key: {}, item: {}, Type: Python complex\n".format(key, item))
    
                
                # Other types
                else:
                    fopen.write("Key: {}, item:{}, Type: {}\n".format(key, item, type(item).__name__))
                    
                    
                    
                    
###########check                        
###########check                        
###########check                        
###########check                        
###########check                      
def check_nan_and_exit(array):
    """Check if a CuPy array contains any NaN values and exit the program if it does."""
    moudle = get_module_type(array)
    
    if moudle.any(moudle.isnan(array)):
        print("Array contains NaN values. Exiting the program.")
        sys.exit(1)
    else:
        print("Array does not contain NaN values.")


def check_numeric_list(ray_para_list):
    for item in ray_para_list:
        if not isinstance(item, (int, float)):
            raise TypeError(f"List contains a non-numeric element: {item}")

def check_contiguous(array):
    if isinstance(array, cp.ndarray):
        return {
            "c_contiguous": array.flags.c_contiguous,
            "f_contiguous": array.flags.f_contiguous,
        }
    elif isinstance(array, np.ndarray):
        return {
            "c_contiguous": array.flags.c_contiguous,
            "f_contiguous": array.flags.f_contiguous,
        }
    elif isinstance(array, torch.Tensor):
        return {
            "c_contiguous": array.is_contiguous(),
            "f_contiguous": False,  # Torch doesn't distinguish C/F order
        }
    else:
        raise TypeError("Unsupported array type")


def normalize_list(input_list, fabs_bool=True, module=np):
    """
    
    """
    normalized_list = []
    
    for wavelet in input_list:
        # Compute the maximum value of the wavelet
        if fabs_bool:
            max_value = module.max(module.abs(wavelet))
        else:
            max_value = module.max( (wavelet) )
        
        # Normalize the wavelet by its maximum value
        if max_value != 0:  # Avoid division by zero
            normalized_wavelet = wavelet / max_value
        else:
            normalized_wavelet = wavelet  # If the max is zero, keep the array unchanged
        
        # Append the normalized wavelet to the list
        normalized_list.append(normalized_wavelet)
    
    return normalized_list


def fft1D_and_magnitude(wavelet_list, length=10, axis=0, module=np):
    """
    Perform FFT on each signal in the list and compute the magnitude of the FFT result.
    
    Parameters:
    wavelet_list (list of cupy.ndarray): List of wavelet arrays to process.
    
    Returns:
    list of cupy.ndarray: List of magnitudes of FFT results for each wavelet.
    """
    magnitude_list = []
    
    for wavelet in wavelet_list:
        
        if wavelet.shape[0]<length:
            wavelet2 = module.pad(wavelet, ((0, length - wavelet.shape[0]), (0, 0)), mode='constant', constant_values=0)
        else:
            wavelet2 = wavelet
        
        # Perform FFT on the wavelet2
        fft_result = module.fft.fft(wavelet2, axis=axis)
        
        # Compute the magnitude (absolute value of the complex FFT result)
        magnitude = module.abs(fft_result)
        
        # Append the magnitude to the list
        magnitude_list.append(magnitude)
    
    return magnitude_list




#########some operation on list
#########some operation on list
#########some operation on list
#########some operation on list
#########some operation on list
def list_l2_norm_square(array_list):
    module = get_module_type(array_list[0]);
    return sum( 0.5 * (module.sum(array * array)).item() for array in array_list  );

def list_l1_norm(array_list):
    module = get_module_type(array_list[0]);
    return sum(  module.sum( module.abs(array) ).item() for array in array_list  );

def list_arr_sum(array_list):
    module = get_module_type(array_list[0]);
    return sum(  module.sum( array ).item() for array in array_list  );

def list_arr_math(array_list1, array_list2, math_type="add"):
    
    module = get_module_type(array_list1[0]);
    
    if   math_type in ["add"]:
        return [ arr1+ arr2  for arr1, arr2 in zip( array_list1, array_list2 )]
    elif math_type in ["sub"]:
        return [ arr1 - arr2 for arr1, arr2 in zip( array_list1, array_list2 )]
    elif math_type in ["mul"]:
        return [ arr1 * arr2 for arr1, arr2 in zip( array_list1, array_list2 )]
    elif math_type in ["div"]:
        return [ arr1 / arr2 for arr1, arr2 in zip( array_list1, array_list2 )]
        

                    
def list_to_txt(filename, data_list, w_type="w+"):
    """
    Write a list to a txt file.

    Parameters:
    data_list (list): The list to write to the file.
    filename (str): The name of the file to write the list to.
    """
    with open(filename, w_type) as file:
        for item in data_list:
            file.write(f"{item}\n");
            
def list_to_dict(input_list):
    result_dict = {} 

    if isinstance(input_list, list):
        
        for i, var_value in enumerate(input_list):
            
            result_dict[i] = var_value;
    
    return result_dict 

def list_py_files_and_functions(output_file="py_functions.txt"):
    # Get all paths that have been added to sys.path
    paths = sys.path
    
    # Open output file
    with open(output_file, 'w') as file:
        for path in paths:
            if os.path.exists(path):
                # Iterate through all files in the directory
                for py_file in os.listdir(path):
                    if py_file.endswith('.py'):
                        file_path = os.path.join(path, py_file)
                        file.write(f"File: {file_path}\n")
                        file.write("-" * 40 + "\n")
                        
                        # Extract module name
                        module_name = py_file[:-3]  # Strip .py extension
                        if module_name in sys.modules:
                            module = sys.modules[module_name]
                        else:
                            try:
                                # Import the module if not already loaded
                                module = __import__(module_name)
                            except Exception as e:
                                file.write(f"Error loading module: {e}\n")
                                continue
                        
                        # Inspect the functions in the module
                        functions = inspect.getmembers(module, inspect.isfunction)
                        for func in functions:
                            file.write(f"Function: {func[0]}\n")
                        file.write("\n")
            else:
                file.write(f"Path not found: {path}\n")
                file.write("-" * 40 + "\n")    
    
    
def list_op_list_to_txt(forward_op_list, 
                        log_file="file7.txt", 
                        mark_file="", 
                        w_type='a+'):
    
    with open(log_file, w_type) as fp:

        fp.write("{}={}\n".format(mark_file, [func.__name__ for func in forward_op_list]))
        for i, op_forward in enumerate(forward_op_list):
            
            operator_info = get_operator_info(op_forward, full_code=True)
        
            fp.write(f"{mark_file} [{i}]: {operator_info}\n\n\n");
       


def list_arr_shape(inv_para_list):
    """
    obtain inv_para_list 
    if list，
    if numpy/cupy array torch tensor
    """
    
    if isinstance(inv_para_list, list):
        inv_para_shape = []
        for idx, para in enumerate( inv_para_list ):
            try:
                inv_para_shape.append( list(para.shape) )
            except:
                inv_para_shape.append( f'list[{idx}], there is no shape' );
        
    elif isinstance(inv_para_list, (np.ndarray, cp.ndarray, torch.Tensor)):
        inv_para_shape = list(inv_para_list.shape)
    else:
        raise TypeError(f"Unsupported type: {type(inv_para_list)}. Must be list, numpy/cupy array, or torch tensor.")
    
    return inv_para_shape

def list_element_to_length(in_list, length):
    
    if not isinstance(in_list, list):
       raise TypeError(f"Expected a list, but got {type(in_list).__name__}")
    
       if length <= len(in_list): 
           return in_list 
    
    in_list.extend([in_list[-1]] * (length - len(in_list)))
    return in_list


def list_list_element_idx( list_list, idx=0):
    '''
    This function aims to get the list of list, element
    for each list
    '''
    element_list = []
    
    for list_tmp in list_list:
        
        element_list.append( list_tmp[idx]   );
        
    return element_list
    
def list_list_element_idx_test( list_list, idx=0):
    
    
    list_list = [[1,2,3], [5,6,7]];
    
    
    print(  list_list_element_idx( list_list, idx=0) )
    
    print(  list_list_element_idx( list_list, idx=2) )
    

#######################
def free_memory():

    # Free GPU memory in PyTorch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Free GPU memory in CuPy
    cp.get_default_memory_pool().free_all_blocks()
    
    # Run garbage collection to free up RAM
    gc.collect()  
######

def func_info(print_bool=False):
    # 获取当前帧信息
    frame = inspect.currentframe().f_back
    # 获取文件路径和函数名称
    file_path = inspect.getfile(frame)
    function_name = inspect.getframeinfo(frame).function
    
    if print_bool:
        print(f"Entering function '{function_name}' in file '{file_path}'")
    
    return file_path + " : " + function_name 

def randn_like_func(x):
    
    module = get_module_type(x)
    
    try:
        # Try module's direct randn_like method
        return module.randn_like(x)
    except AttributeError:
        try:
            # Fallback for modules without randn_like but with random.randn
            return module.random.randn(*x.shape).astype(x.dtype)
        except AttributeError:
            # Raise an error if no compatible method is found
            raise TypeError(f"Unsupported module: {module.__name__}")


def test_randn_like():
    # NumPy test
    np_array = np.ones((3, 3), dtype=np.float32)
    np_result = randn_like_func(np_array)
    print("NumPy Test:")
    print(np_result)
    print("Shape:", np_result.shape, "| Dtype:", np_result.dtype)

    # CuPy test
    cp_array = cp.ones((3, 3), dtype=cp.float32)
    cp_result = randn_like_func(cp_array)
    print("\nCuPy Test:")
    print(cp_result)
    print("Shape:", cp_result.shape, "| Dtype:", cp_result.dtype)

    # PyTorch test
    torch_tensor = torch.ones((3, 3), dtype=torch.float32)
    torch_result = randn_like_func(torch_tensor)
    print("\nPyTorch Test:")
    print(torch_result)
    print("Shape:", torch_result.shape, "| Dtype:", torch_result.dtype)


def dot_test(forward, adjoint, x, 
                        use_x=False,
                        tolerance=1e-3, 
                        output_bool=True, 
                        description='', 
                        log_file="dot_test.txt"):
    """
    if there is non-linear relation, I will use it to check dimension size for debug code
    
    Tests the dot product consistency between forward and adjoint operations.
        
    Args:
        forward (callable): The forward operator.
        adjoint (callable): The adjoint operator.
        x_list (list): List of input tensors/vectors.
        tolerance (float): Tolerance for checking equality.
        output_bool (bool): Whether to print the results.
        
    Returns:
        tuple: (x1x2_value, y1y2_value) as lists of dot products.
    """
    
    for_info = get_operator_info(forward, full_code=False);
    adj_info = get_operator_info(adjoint, full_code=False);
    
    write_txt(log_file, f"   forward: {for_info}\n", print_bool=True);
    write_txt(log_file, f"   adjoint: {adj_info}\n", print_bool=True);
    
    
    '''Ensure input validity'''
    x_list = x
    if isinstance(x_list, list):
        module = get_module_type(x_list[0])
    elif isinstance(x_list, (np.ndarray, cp.ndarray, torch.Tensor)):
        module = get_module_type(x_list)
    else:
        raise TypeError("   Unsupported type for x_list: {}".format(type(x_list)))
    
    
    start_time = time.time();
    
    '''Generate random test inputs'''
    if not use_x:
        if  isinstance(x_list, list):
            x1_list = [ randn_like_func(x) for x in x_list ]
        else:
            x1_list = randn_like_func(x_list)
    else:
        '''link address'''
        x1_list = x_list;
    
    
    '''apply forward operator'''
    y1_list = forward(x1_list)
    
    
    if  isinstance(y1_list, list):
        y2_list = [ randn_like_func(y) for y in y1_list ]
    else:
        y2_list = randn_like_func(y1_list)
    
    
    '''apply adjoint operator'''
    x2_list = adjoint(y2_list)
    
    end_time   = time.time();
    
    
    '''Compute dot products'''
    if  isinstance(x1_list, list):
        x1x2_value = [ module.sum(x1 * x2).item() for x1, x2 in zip(x1_list, x2_list) ]
    else:
        x1x2_value = module.sum(x1_list * x2_list).item()
    
    
    if  isinstance(y1_list, list):
        y1y2_value = [ module.sum(y1 * y2).item() for y1, y2 in zip(y1_list, y2_list) ]
    else:
        y1y2_value = module.sum(y1_list * y2_list).item()
    
    
    '''Sum up dot products for overall comparison'''
    try:
        x1x2_sum = sum(x1x2_value)
    except:
        x1x2_sum = x1x2_value
    try:
        y1y2_sum = sum(y1y2_value)
    except:
        y1y2_sum = y1y2_value
    
    
    '''Print and check results'''
    
    write_txt(log_file, f"   {description} x1x2_sum: {x1x2_sum:.6f}", print_bool=True);
    write_txt(log_file, f"   {description} y1y2_sum: {y1y2_sum:.6f}", print_bool=True);
    write_txt(log_file, f"   {description} x1x2_value: {x1x2_value}", print_bool=True);
    write_txt(log_file, f"   {description} y1y2_value: {y1y2_value}", print_bool=True);
    
    if np.allclose(x1x2_sum, y1y2_sum, rtol=tolerance):
        if output_bool:
            write_txt(log_file, f"   {description} Dot product test passed, times:{end_time-start_time}\n\n", print_bool=True);
        
        return True
    
    else:
        if output_bool:
            write_txt(log_file, f"   Warning: {description} Dot product test did not pass, times:{end_time-start_time}\n\n", print_bool=True);
        
        return False

def define_operator_from_operator_list(in_operator_list, a_to_b_bool=True):
    
    if a_to_b_bool:
        '''Apply operators in forward direction (0, 1, 2, ...)'''
        composite_operator = lambda x: x
        for operator in in_operator_list:
            composite_operator = (  lambda prev_op, op: lambda x: op(prev_op(x))  )(composite_operator, operator)
    else:
        '''Apply operators in reverse direction (n-1, n-2, ...)'''
        composite_operator = lambda x: x
        for operator in reversed(in_operator_list):
            composite_operator = (  lambda prev_op, op: lambda x: op(prev_op(x))  )(composite_operator, operator)

    return composite_operator

def define_apply_operator_on_operator_list(forward_adjoint, in_operator_list, a_to_b_bool=True):
    
    ou_operator_list=[];
    
    for idx, operator in enumerate(  in_operator_list  ):
        
        if a_to_b_bool:
            # op = lambda x : forward_adjoint(  operator(x)  );
            op = lambda x, op=operator: forward_adjoint(  op(x)  )
        else:
            # op = lambda x : operator(  forward_adjoint(x)  );
            op = lambda x, op=operator:  op(  forward_adjoint(x)  )
        
        ou_operator_list.append(  op  )
        
    return ou_operator_list

def define_apply_operator_list_on_operator_list(forward_adjoint_list, in_operator_list, a_to_b_bool=True):
    
    
    ou_operator_list1=[];
    
    for idx,forward_adjoint in enumerate(  forward_adjoint_list  ):
        
        ou_operator_list2=[];
        
        for idx, in_operator in enumerate(  in_operator_list  ):
            
            if a_to_b_bool:
                # op = lambda x :forward_adjoint(  in_operator(x)  );
                op = lambda x, op=in_operator :forward_adjoint(  op(x)  );
            else:
                # op = lambda x : in_operator( forward_adjoint(x)  );
                op = lambda x, op=in_operator : op( forward_adjoint(x)  );
            
            ou_operator_list2.append(  op  )
            
        ou_operator_list1.append( ou_operator_list2 )
        
    return ou_operator_list1


def generate_unique_integers(start, end, count, module=np):
    """
    Generate unique random integers using NumPy, CuPy, or PyTorch.
    
    Parameters:
    - start (int): The start of the range (inclusive).
    - end (int): The end of the range (exclusive).
    - count (int): The number of unique integers to generate.
    - module: The module to use: 'np' (NumPy), 'cp' (CuPy), or 'torch' (PyTorch).
    
    Returns:
    - array-like: An array of unique integers (NumPy, CuPy, or PyTorch).
    """
    range_size = end - start
    if count > range_size:
        print(f"generate_unique_integers, Count ({count}) is greater than range size ({range_size}). Adjusting count to {range_size}.")
        count = range_size  # Adjust count to the maximum possible value
    
    if module == np:
        result = np.random.choice(range(start, end), size=count, replace=False)
        return result.astype(int)  # Ensure result is a NumPy integer array
    elif module == cp:
        result = cp.random.choice(cp.arange(start, end), size=count, replace=False)
        return result.astype(cp.int32)  # Ensure result is a CuPy integer array
    elif module == torch:
        result = torch.randperm(range_size)[:count] + start
        return result.to(dtype=torch.int32)  # Ensure result is a PyTorch integer tensor
    else:
        raise ValueError("Unsupported module. Choose 'np', 'cp', or 'torch'.")

# x = generate_unique_integers(0, 10, 0, module =np)








def validate_lists(func):
    """装饰器：对函数参数中带有类型注解为 list 的参数进行检查"""
    def wrapper(*args, **kwargs):
        sig         = inspect.signature(func)
        bound_args  = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # 检查类型注解为 list 的参数
        for param_name, param_value in bound_args.arguments.items():
            param_annotation = sig.parameters[param_name].annotation
            if param_annotation == list and not isinstance(param_value, list):
                raise ValueError(f"Parameter '{param_name}' must be a list, got {type(param_value).__name__}.")
        
        # 执行原始函数
        return func(*args, **kwargs)
    return wrapper


def print_func_info(print_bool=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取当前函数的帧信息
            frame = inspect.currentframe().f_back
            # 获取文件路径和函数名称
            file_path = inspect.getfile(frame)
            function_name = func.__name__
            
            # 打印函数信息
            print(f"Entering function '{function_name}' in file '{file_path}'")
            
            
            result = func(*args, **kwargs)
            
            return result
        return wrapper
    return decorator
