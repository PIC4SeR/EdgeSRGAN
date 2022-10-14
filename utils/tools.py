import subprocess as sp

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import joblib
import yaml
import pprint

pprint.sorted = lambda x, key=None: x


# ---------------------------------------
#  Images
# ---------------------------------------

def load_image(path):
    return np.array(Image.open(path))

def plot_sample(lr, sr):
    
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

        
# ---------------------------------------
#  Logger
# ---------------------------------------

class Logger(object):
    
    def __init__(self, file):
        
        self.file = file
        self.pp = pprint.PrettyPrinter(depth=2)
        
    def save_log(self, text):
        
        if type(text) is dict:
            text = self.pp.pformat(text)
            
        print(text)
        with open(self.file, 'a') as f:
            f.write(text + '\n')
            
            
# ---------------------------------------
#  Yaml
# ---------------------------------------

def read_yaml(path):

    stream = open(path, 'r')
    dictionary = yaml.safe_load(stream)
    return dictionary


def opt_save_callback(study, trial):
    
    joblib.dump(study, f"grid_search/study_srgan_small_{study.study_name}.pkl")
    
    
# ---------------------------------------
#  GPU
# ---------------------------------------

def get_free_gpu(ids, req):
    
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    
    for i in ids:
        if memory_free_values[i] > req:
            return i
    return -1
