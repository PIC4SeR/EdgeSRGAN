import os
from datetime import datetime
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf 

from utils.train import Trainer
from utils.hp_search import HPSearcher
from utils.tools import read_yaml, Logger, get_free_gpu



# CONFIG 
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml', type=str, help='Config path', required=False)    
args = parser.parse_args()
config = read_yaml(args.config)


# GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
i = get_free_gpu([0], config['GPU_MEMORY'])
tf.config.experimental.set_visible_devices(gpus[i], 'GPU')
tf.config.experimental.set_memory_growth(gpus[i], True)


# PATHS and LOGGER
for entry in ['WEIGHTS_DIR','LOG_DIR','HP_SEARCH_DIR']:
    if not os.path.exists(config[entry]):
        os.mkdir(config[entry])
        
now = datetime.now().strftime("%y%m%d%H%M%S")
logger = Logger(config['LOG_DIR'] + '_' + now + '.txt')

    
# TRAINING
if config['HP_SEARCH']:
    searcher = HPSearcher(config=config, logger=logger, teacher=None, trial=None)
    searcher.hp_search()

else:
    if config['VERBOSE']:
        logger.save_log(config)
        
    trainer = Trainer(config=config, logger=logger, teacher=None, trial=None)
    res = trainer.train()
    print(f"Best {config[config['MODE']]['METRIC']}: {res}")

    if config['MODE'] == 'PSNR':
        pre_train_weights = os.path.join(config['WEIGHTS_DIR'], f'pre_generator_{config["MODEL_NAME"]}.h5')
        trainer.generator.save_weights(pre_train_weights)
    
    
    elif config['MODE'] == 'GAN':
        gan_gen_weights = os.path.join(config['WEIGHTS_DIR'], f'gan_generator_{config["MODEL_NAME"]}.h5')
        gan_disc_weights = os.path.join(config['WEIGHTS_DIR'], f'gan_discriminator_{config["MODEL_NAME"]}.h5')
        trainer.generator.save_weights(gan_gen_weights)
        # trainer.discriminator.save_weights(gan_disc_weights)
        