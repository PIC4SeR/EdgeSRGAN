import os
import argparse
import joblib

import optuna

from utils.train import Trainer



class HPSearcher:
      
    def __init__(self, config, logger=None, teacher=None, trial=None):
        
        self.config = config
        self.logger = logger
        self.teacher = teacher
        self.trial = trial
        
        
    def get_random_hps(self):
        
        self.config[self.config['MODE']]['LR'] = self.trial.suggest_categorical("LR", [1e-4, 5e-5])
        self.config[self.config['MODE']]['LAMBDA'] = self.trial.suggest_categorical("LAMBDA", [0.1, 0.25])
        self.config[self.config['MODE']]['FEAT_WEIGHT'] = self.trial.suggest_categorical("FEAT_WEIGHT", [0.01, 0.1, 1, 10])
        self.config[self.config['MODE']]['TASK_WEIGHT'] = self.trial.suggest_categorical("TASK_WEIGHT", [0.01, 0.1, 1, 10])
        #self.config[self.config['MODE']]['SCHEDULER'] = self.trial.suggest_categorical("SCHEDULER", ['Step'])
        #self.config[self.config['MODE']]['ALPHA'] = self.trial.suggest_categorical("ALPHA", [0.001])
        #self.config[self.config['MODE']]['GP_WEIGHT'] = self.trial.suggest_categorical("GP_WEIGHT", [0.0])
        #self.config[self.config['MODEL_SIZE']]['DIST_LAYERS'] = self.trial.suggest_categorical("DIST_LAYERS", [(1,2,4),(1,2,3)])
        #self.config[self.config['TEACHER_SIZE']]['DIST_LAYERS'] = self.trial.suggest_categorical("TEACH_LAYERS", [(2,5,8),(1,4,8)])
        
        #self.config[self.config['MODE']]['STEPS'] = int(self.trial.suggest_categorical("STEPS", [200000]))
        
        #self.config[self.config['MODE']]['DISC_ACT'] = self.trial.suggest_categorical("DISC_ACT", ['sigmoid'])
        #self.config[self.config['MODEL_SIZE']]['DISC_BOTTLENECK'] = self.trial.suggest_categorical("DISC_BOTTLENECK", ['Flatten'])
        #self.config[self.config['MODE']]['DISC_OPT'] = self.trial.suggest_categorical("DISC_OPT", ['Adam'])
        #self.config[self.config['MODEL_SIZE']]['DISC_HEAD'] = self.trial.suggest_categorical("DISC_HEAD",[256,512])    
        #self.config[self.config['MODE']]['OUT_LOSS'] = self.trial.suggest_categorical("OUT_LOSS", ['MAE'])
        
        if self.config['VERBOSE']:
            self.logger.save_log(self.config[self.config['MODE']])
    
    
    def objective(self, trial):
        
        name = 'gridsearch_' + str(trial.datetime_start) + str(trial.number)
        self.trial = trial 
        
        self.get_random_hps()
        trainer = Trainer(config=self.config, logger=self.logger, teacher=self.teacher, trial=self.trial)

        
        pre_train_weights = os.path.join(self.config['WEIGHTS_DIR'], f'{self.config["PSNR_MODEL"]}.h5')
        trainer.generator.load_weights(pre_train_weights)
        
        metr = trainer.train()
        trainer.generator.load_weights(os.path.join(self.config['WEIGHTS_DIR'], f'best_{self.config["MODEL_NAME"]}.h5'))
        trainer.generator.save_weights(os.path.join(self.config['WEIGHTS_DIR'], f'gan_generator_{name}.h5'))

        return metr
    
    
    def hp_search(self):

        direction = 'maximize' if self.config[self.config['MODE']]['METRIC'] == 'PSNR' else 'minimize'
        self.study = optuna.create_study(study_name=f'hp_search_{self.config["HP_SEARCH_NAME"]}', 
                                         direction=direction, 
                                         pruner=optuna.pruners.MedianPruner())
        
        if os.path.exists(f'{self.config["HP_SEARCH_DIR"]}/hp_search_{self.config["HP_SEARCH_NAME"]}.pkl'): 
            study_old = joblib.load(f'{self.config["HP_SEARCH_DIR"]}/hp_search_{self.config["HP_SEARCH_NAME"]}.pkl')
            self.study.add_trials(study_old.get_trials())
            print('Study resumed!')
        
        save_callback = SaveCallback(self.config['HP_SEARCH_DIR'])
        self.study.optimize(lambda trial: self.objective(trial), n_trials=self.config['N_TRIALS'],
                            callbacks=[save_callback])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")
        self.logger.save_log("Best trial:")
        self.logger.save_log(f"  Value: {self.study.best_trial.value}")
        self.logger.save_log("  Params: ")
        for key, value in self.study.best_trial.params.items():
            self.logger.save_log(f"    {key}: {value}")

        return self.study
    
    
    
class SaveCallback:
    
    def __init__(self, directory):
        self.directory = directory

    def __call__(self, study, trial):
        joblib.dump(study, os.path.join(self.directory, f'{study.study_name}.pkl'))