import torch
import subprocess
import os
import shutil
import numpy as np
from sys import platform 
from utils.logger import PythonLogger 
logger = PythonLogger().logger 

def get_splitter():
    if platform.startswith("linux") or platform.startswith("darwin"):
        return '/'
    else:
        return "\\"

def check_hardware():
    # Need to use torch to check gpu
    # Check whether hardware has gpu, if there is not gpu, model will predict using cpu
    
    cuda_avail = torch.cuda.is_available()
    
    if cuda_avail:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f'GPU Count: {device_count}, GPU Name: {device_name}')
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"GPU Available: {cuda_avail}, Device: {device}")
    return cuda_avail

def seed_everything(seed=43):
    '''
      Make PyTorch deterministic.
    '''    
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

'''
def display_output(no_detection_images, sign_stamp_count):
    print()
    print('########## Report ###########')
    print('PDFS that does not have signs or stamps')
    for count, value in enumerate(no_detection_images):
        print(f'{count + 1}: {value}')
        
    print()
    print('PDFS that have signs and stamps')
    for count, (key, value) in enumerate(sign_stamp_count.items()):
        print(f"{count+1}: {key}: {value}")

def allowed_file(filename):
    ALLOWED_EXTENSIONS = set(['pdf', 'PDF'])
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS    

'''

# Paths    
def get_preprocess_image_paths():
    paths = []
    
    for dirname, _, filenames in os.walk(get_preprocess_folder_path()):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            
    return paths

def get_user_upload_pdf_paths():
    paths = []
    
    for dirname, _, filenames in os.walk(get_user_uploads_paths()):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            
    return paths

def get_pred_img_paths():
    paths = []
    splitter = get_splitter()
    
    for dirname, _, filenames in os.walk(get_pred_folder_path()):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))

    new_paths = sorted([path.split(splitter)[-1] for path in paths])
    return new_paths      

def get_pred_folder_path():
    return r"/app/static/uploads/results"

def get_weight_path():
    return r"/app/weights/trial6-best.onnx"

def get_user_uploads_paths():
    path = r'/app/user-uploads'    
    os.makedirs(path, exist_ok=True)
    return path

def clean_up():
    '''
       Delete
       1: user-upload pdf folder
       2: preprocessed_imgs folder 
    '''

    user_upload_dir = get_user_uploads_paths()
    preprocess_imgs_dir = get_preprocess_folder_path()
    pred_imgs_dir = get_pred_folder_path()

    shutil.rmtree(user_upload_dir)
    shutil.rmtree(preprocess_imgs_dir)
    #shutil.rmtree(pred_imgs_dir)
