import os
import argparse
import joblib

import optuna

from utils.train import Trainer



class Tester:
      
    def __init__(self, config, logger=None, trainer=None):
        
        self.config = config
        self.logger = logger
        self.trainer = trainer
        
   
   
            