import os
import time

import math
import numpy as np

import tensorflow as tf
import torch
import optuna

from utils.data import DIV2K
from utils.srgan import generator, discriminator, vgg_22, vgg_54, AccumulateNorm
from utils.srgan_tools import evaluate, get_margin

import lpips
      
class Trainer:
    
    def __init__(self, config, logger=None, teacher=None, trial=None):

        self.config = config
        self.logger = logger

        self.teacher = teacher
        self.trial = trial
                
        if self.config['MODE'] == 'TEST':
            self.lrate = 0.0    
            self.get_models()
            self.valid_ds = None
            return
        
        self.L = self.config[self.config['MODE']]['LAMBDA']
        self.W = self.config[self.config['MODE']]['FEAT_WEIGHT']
        self.T = self.config[self.config['MODE']]['TASK_WEIGHT']
        
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
        if self.config[self.config['MODE']]['OUT_LOSS'] == 'MSE':
            self.out_loss = tf.keras.losses.MeanSquaredError()
        else:
            self.out_loss = tf.keras.losses.MeanAbsoluteError()
        
        self.kl = tf.keras.losses.KLDivergence()
        self.gen_mean = tf.keras.metrics.Mean()
        self.disc_mean = tf.keras.metrics.Mean()
        
        self.prep = tf.keras.applications.vgg19.preprocess_input
        
        self.get_dataset()
        self.get_scheduler()            
        self.get_models()
    
        if self.config['MODE'] == 'PSNR':
            self.train_step = self.train_step_psnr
        else:
            self.train_step = self.train_step_gan
    
        signature_lr = [self.config[self.config['MODE']]['BATCH_SIZE'], self.config['PATCH_SIZE'][0] // self.config['SCALE'], 
                        self.config['PATCH_SIZE'][1] // self.config['SCALE'], self.config['CHANNELS']]
        signature_hr = [self.config[self.config['MODE']]['BATCH_SIZE'], self.config['PATCH_SIZE'][0], 
                        self.config['PATCH_SIZE'][1], self.config['CHANNELS']]

        self.train_step = tf.function(input_signature=[tf.TensorSpec(shape=signature_lr, dtype=tf.uint8),
                                                       tf.TensorSpec(shape=signature_hr, dtype=tf.uint8)])(self.train_step)
        
        if self.config[self.config['MODE']]['METRIC'] == 'LPIPS':
            with torch.no_grad():
                self.lpips = lpips.LPIPS(net='alex').cuda()
    
    
    def get_dataset(self):
        
        div2k_train = DIV2K(scale=self.config['SCALE'], resolution=tuple(self.config['PATCH_SIZE']),
                            subset='train', downgrade='bicubic', data_dir=self.config['DATA_DIR'])
        div2k_valid = DIV2K(scale=self.config['SCALE'], subset='valid', downgrade='bicubic', data_dir=self.config['DATA_DIR'])

        self.train_ds = div2k_train.dataset(batch_size=self.config[self.config['MODE']]['BATCH_SIZE'],
                                            rand_transform=self.config['TRANSFORMS'], rand_jpeg=self.config['RANDOM_JPEG'])
        self.valid_ds = div2k_valid.dataset(batch_size=1, rand_crop=True, rand_transform=False, repeat_count=1, cache=True)
    
    
    def get_models(self):
        
        self.generator = generator(scale=self.config['SCALE'],
                         num_filters=self.config[self.config['MODEL_SIZE']]['FILTERS'], 
                         num_res_blocks=self.config[self.config['MODEL_SIZE']]['BLOCKS'],
                         batch_norm=self.config[self.config['MODEL_SIZE']]['BATCH_NORM'],
                         activation=self.config[self.config['MODEL_SIZE']]['ACTIVATION'],
                         upsampling=self.config[self.config['MODEL_SIZE']]['UPSAMPLING'])
        
        self.gen_optim = tf.keras.optimizers.Adam(learning_rate=self.lrate)

        if self.config['MODE'] == 'PSNR':
            self.perc_model = None
            self.discriminator = None
        elif self.config['MODE'] == 'GAN':
            self.discriminator = discriminator(shape=(self.config['PATCH_SIZE'][0],self.config['PATCH_SIZE'][1], 
                                                      self.config['CHANNELS']),
                                               num_filters=self.config[self.config['MODEL_SIZE']]['FILTERS'],
                                               bottleneck=self.config[self.config['MODEL_SIZE']]['DISC_BOTTLENECK'],
                                               head=self.config[self.config['MODEL_SIZE']]['DISC_HEAD'], 
                                               activation=self.config[self.config['MODEL_SIZE']]['DISC_ACT'],
                                               big=self.config[self.config['MODEL_SIZE']]['BIG_DISC'])
            
            if self.config[self.config['MODE']]['DISC_OPT'] == 'Adam':
                self.disc_optim = tf.keras.optimizers.Adam(learning_rate=self.lrate)
            elif self.config[self.config['MODE']]['DISC_OPT'] == 'RMSprop':
                self.disc_optim = tf.keras.optimizers.RMSprop(learning_rate=self.lrate)
                
            if self.config[self.config['MODE']]['PERC_MODEL'] == 'VGG22':
                self.perc_model = vgg_22()
            elif self.config[self.config['MODE']]['PERC_MODEL'] == 'VGG54':
                self.perc_model = vgg_54()
                
        if self.config['DISTILLATION'] != 'None' and self.config['MODE'] != 'TEST':
            self.teacher = generator(scale=self.config['SCALE'],
                                     num_filters=self.config[self.config['TEACHER_SIZE']]['FILTERS'], 
                                     num_res_blocks=self.config[self.config['TEACHER_SIZE']]['BLOCKS'],
                                     batch_norm=self.config[self.config['TEACHER_SIZE']]['BATCH_NORM'],
                                     activation=self.config[self.config['TEACHER_SIZE']]['ACTIVATION'],
                                     upsampling=self.config[self.config['TEACHER_SIZE']]['UPSAMPLING'])
            
            teacher_weights = os.path.join(self.config['WEIGHTS_DIR'], self.config['TEACHER_NAME'])
            self.teacher.load_weights(teacher_weights)
            
            if self.config['DISTILLATION'] == 'BRKD':
                self.get_head()
                
            elif self.config['DISTILLATION'] != 'FAKD':
                self.get_connectors()
                self.get_margins()
                

    def get_scheduler(self):
        if self.config[self.config['MODE']]['SCHEDULER'] == 'Exponential':
            self.lrate = tf.keras.optimizers.schedules.ExponentialDecay(float(self.config[self.config['MODE']]['LR']), 
                                                                        self.config['STEPS'], 
                                                                        self.config['EXP_DECAY'])
        elif self.config[self.config['MODE']]['SCHEDULER'] == 'Linear':
            self.lrate = tf.keras.optimizers.schedules.InverseTimeDecay(float(self.config[self.config['MODE']]['LR']), 
                                                                        self.config['STEPS'], 
                                                                        self.config['LIN_DECAY'])
        elif self.config[self.config['MODE']]['SCHEDULER'] == 'Constant':
            self.lrate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([0],
                                                                              [float(self.config[self.config['MODE']]['LR']),
                                                                               float(self.config[self.config['MODE']]['LR'])])
        elif self.config[self.config['MODE']]['SCHEDULER'] == 'Step':
            self.lrate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([self.config[self.config['MODE']]['STEPS']/2],
                                                                              [float(self.config[self.config['MODE']]['LR']),
                                                                               float(self.config[self.config['MODE']]['LR'])/10])
    
    def get_head(self):
        
        self.head = generator(scale=self.config['SCALE'],
                              num_filters=self.config[self.config['TEACHER_SIZE']]['FILTERS'], 
                              num_res_blocks=self.config[self.config['TEACHER_SIZE']]['BLOCKS'],
                              batch_norm=self.config[self.config['TEACHER_SIZE']]['BATCH_NORM'],
                              activation=self.config[self.config['TEACHER_SIZE']]['ACTIVATION'],
                              upsampling=self.config[self.config['TEACHER_SIZE']]['UPSAMPLING'],
                              head_only=True)
        self.head.set_weights(self.teacher.get_weights()[-4:])
            
    
    def get_connectors(self):

        self.connectors = []
        for i in range(len(self.config[self.config['TEACHER_SIZE']]['DIST_LAYERS'])):

            model = tf.keras.Sequential()
            model.add(tf.keras.Input(shape=(self.config['PATCH_SIZE'][0] // self.config['SCALE'],
                                            self.config['PATCH_SIZE'][0] // self.config['SCALE'],
                                            self.config[self.config['MODEL_SIZE']]['FILTERS'])))
            n = math.sqrt(2.0 / self.config[self.config['TEACHER_SIZE']]['FILTERS'])
            model.add(tf.keras.layers.Conv2D(self.config[self.config['TEACHER_SIZE']]['FILTERS'],
                                             kernel_size=1, strides=1, use_bias=False,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=n)))
            model.add(tf.keras.layers.BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001))

            self.connectors.append(model)

                      
    def get_margins(self):
        
        if self.config['DISTILLATION'] == 'ABKD':
            self.margins = [1.0 for _ in self.config[self.config['TEACHER_SIZE']]['DIST_LAYERS']] 
        else:
            batch_norms = [l.get_weights() for l in self.teacher.layers if isinstance(l, AccumulateNorm)]
            batch_norms = [batch_norms[i] for i in self.config[self.config['TEACHER_SIZE']]['DIST_LAYERS']]             
            means = [bn[0] for bn in batch_norms]
            stds = [tf.math.sqrt(bn[1]) for bn in batch_norms]
            self.margins = [get_margin(m,s) for (m,s) in zip(means, stds)]
                
                
    def log_results(self, step, gen_loss, metric, disc_loss, now):
        
        log = f"{step}/{self.config[self.config['MODE']]['STEPS']}: LR {self.lrate(step):.2e}, "
        if self.config['MODE'] == 'PSNR':
            log += f"Loss {gen_loss:.4f}, "
        else:
            log += f"Perc {gen_loss:.4f}, Disc {disc_loss:.4f}, "
        log += f"{self.config[self.config['MODE']]['METRIC']} {metric:.4f} ({time.perf_counter() - now:.1f}s)"
        
        self.logger.save_log(log)
    

    def train(self):
        
        if self.config['MODE'] == 'GAN':
            pre_train_weights = os.path.join(self.config['WEIGHTS_DIR'], f'{self.config["PSNR_MODEL"]}.h5')
            self.generator.load_weights(pre_train_weights)
        
        now = time.perf_counter()
        steps = self.config[self.config['MODE']]['STEPS']

        for step, (lr, hr) in enumerate(self.train_ds.take(steps), 1):
            gl, dl = self.train_step(lr, hr)
            self.gen_mean(gl)
            self.disc_mean(dl)

            if step % self.config[self.config['MODE']]['EVAL_EVERY'] == 0:
                metr = self.evaluate(self.valid_ds.take(len(self.valid_ds)))  
                if self.should_save(metr):
                    self.best_metr = metr
                    self.generator.save_weights(os.path.join(self.config['WEIGHTS_DIR'], 
                                                             f'best_{self.config["MODEL_NAME"]}.h5'))
                    self.logger.save_log('Saved ->')
                    
                self.log_results(step, self.gen_mean.result(), metr, self.disc_mean.result(), now)
                
                if self.trial is not None:
                    self.trial.report(metr, step)
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                self.gen_mean.reset_states()
                self.disc_mean.reset_states()
                
                now = time.perf_counter()
        
        return self.best_metr if self.config['CHECKPOINT'] else metr
                                                
                                    
    def train_step_psnr(self, lr, hr):
        
        with tf.GradientTape() as gen_tape:
            
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)
            sr, feats_s = self.generator(lr, training=True)
            
            loss = self.out_loss(hr, sr)
            
            if self.config['DISTILLATION'] != 'None':
                sr_t, feats_t = self.teacher(lr, training=False)
        
                feats_t = tf.gather(feats_t,[self.config[self.config['TEACHER_SIZE']]['DIST_LAYERS']])
                feats_s = tf.gather(feats_s,[self.config[self.config['MODEL_SIZE']]['DIST_LAYERS']])

                if self.config['DISTILLATION'] == 'FAKD':
                    feat_kd_loss = self.fakd_loss(feats_t, feats_s)
                    out_kd_loss = self.out_loss(sr_t, sr)
                    loss += feat_kd_loss + 0.5 * out_kd_loss
                       
                elif self.config['DISTILLATION'] == 'COKD':
                    feat_kd_loss = 0
                    for i in range(len(self.config[self.config['MODEL_SIZE']]['DIST_LAYERS'])):
                        sf = self.connectors[i](feats_s[0,i])
                        loss += self.cokd_loss(sf, feats_t[0,i], self.margins[i] / 2 ** (len(self.connectors) - i - 1)) 
                
                elif self.config['DISTILLATION'] == 'ABKD':
                    for i in range(len(self.config[self.config['MODEL_SIZE']]['DIST_LAYERS'])):
                        sf = self.connectors[i](feats_s[0,i])
                        loss += self.abkd_loss(sf, feats_t[0,i], self.margins[i] / 2 ** (len(self.connectors) - i - 1)) 
                        
                elif self.config['DISTILLATION'] == 'BRKD':
                    task_kd_loss = self.brkd_loss(self.head(feats_s[0][0], training=False),sr_t) # task-specific loss
                    out_kd_loss = self.out_loss(sr_t, sr)
                    feat_kd_loss = self.fakd_loss(feats_t, feats_s) # feature loss
                    loss += self.W * feat_kd_loss 
                    loss += self.L * out_kd_loss 
                    loss += self.T * task_kd_loss
                    
        gen_grads = gen_tape.gradient(loss, self.generator.trainable_variables)
        self.gen_optim.apply_gradients(zip(gen_grads, self.generator.trainable_variables))

        return loss, 0
                

    def train_step_gan(self, lr, hr):
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            
            lr = tf.cast(lr, tf.float32)
            hr = tf.cast(hr, tf.float32)

            sr, feats_s= self.generator(lr, training=True)

            hr_disc = self.discriminator(hr, training=True)
            sr_disc = self.discriminator(sr, training=True)

            cont_loss = self.content_loss(hr, sr)
            gen_adv_loss = self.generator_adversarial_loss(sr_disc)
            
            perc_loss = cont_loss + self.config[self.config['MODE']]['ALPHA'] * gen_adv_loss
            
            disc_adv_loss = self.discriminator_adversarial_loss(hr_disc, sr_disc)
            disc_reg = self.gradient_penalty(hr, sr)
            
            disc_loss = disc_adv_loss + self.config[self.config['MODE']]['GP_WEIGHT'] * disc_reg
            
            if self.config['DISTILLATION'] != 'None':
                sr_t, feats_t = self.teacher(lr, training=False)
                out_kd_loss = self.out_loss(sr_t, sr)
                
                feats_t = tf.gather(feats_t,[self.config[self.config['TEACHER_SIZE']]['DIST_LAYERS']])
                feats_s = tf.gather(feats_s,[self.config[self.config['MODEL_SIZE']]['DIST_LAYERS']])
                
                # print(feats_t.shape, feats_s.shape)
                
                feats_t = tf.squeeze(feats_t)
                feats_s = tf.squeeze(feats_s)
                
                # print(feats_t.shape, feats_s.shape)
                
                if self.config['DISTILLATION'] == 'FAKD':
                    feat_kd_loss = self.fakd_loss(feats_t, feats_s)
                    perc_loss += self.W * feat_kd_loss + self.L * out_kd_loss
                    
                elif self.config['DISTILLATION'] == 'COKD':
                    for i in range(len(self.config[self.config['MODEL_SIZE']]['DIST_LAYERS'])):
                        sf = self.connectors[i](feats_s[i])
                        perc_loss += self.cokd_loss(sf, feats_t[i], self.margins[i] / 2 ** (len(self.connectors) - i - 1)) 
                        perc_loss += self.L * out_kd_loss

                elif self.config['DISTILLATION'] == 'ABKD':
                    for i in range(len(self.config[self.config['MODEL_SIZE']]['DIST_LAYERS'])):
                        sf = self.connectors[i](feats_s[i])
                        perc_loss += self.abkd_loss(sf, feats_t[i], self.margins[i] / 2 ** (len(self.connectors) - i - 1)) 
                        perc_loss += self.L * out_kd_loss
        
                elif self.config['DISTILLATION'] == 'BRKD':
                    task_kd_loss = self.brkd_loss(self.head(feats_s[0][None], training=False),sr_t) # task-specific loss
                    feat_kd_loss = self.fakd_loss(feats_t, feats_s) # feature loss
                    perc_loss += self.W * feat_kd_loss + self.L * out_kd_loss + self.T * task_kd_loss
            
            
        gen_grads = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.gen_optim.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        self.disc_optim.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))

        return perc_loss, disc_loss

    
    def content_loss(self, hr, sr):
        
        sr = self.prep(sr)
        hr = self.prep(hr)
        sr_features = self.perc_model(sr, training=False) / 12.75
        hr_features = self.perc_model(hr, training=False) / 12.75
        
        return self.out_loss(hr_features, sr_features)

    
    def generator_adversarial_loss(self, sr_disc):
        
        return self.bce(tf.ones_like(sr_disc), sr_disc) # Discriminator must output 1 for SR images

    
    def discriminator_adversarial_loss(self, hr_disc, sr_disc):
        
        hr_loss = self.bce(tf.ones_like(hr_disc), hr_disc) # Discriminator must output 1 for HR images
        sr_loss = self.bce(tf.zeros_like(sr_disc), sr_disc) # Discriminator must output 0 for SR images
        return hr_loss + sr_loss


    def gradient_penalty(self, hr, sr):

        eps = tf.random.uniform(np.array([sr.shape[0],1,1,1]), 0.0, 1.0, dtype=tf.float32)
        x_hat = eps * hr + (1 - eps) * sr
        
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminator(x_hat)
            
        grads = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(grads**2, axis=[1,2]))
        d_reg = tf.reduce_mean((ddx-1.0)**2)
        
        return d_reg

    
    def fakd_loss(self, ft, fs):

        #fs_n = tf.linalg.normalize(fs)[0]
        fs_n = fs / tf.repeat(tf.norm(fs, ord='euclidean', axis=-1)[...,None], 
                              self.config[self.config['MODEL_SIZE']]['FILTERS'], 
                              axis=-1)
        fs_r = tf.reshape(fs_n, [len(self.config[self.config['MODEL_SIZE']]['DIST_LAYERS']),
                                 self.config[self.config['MODE']]['BATCH_SIZE'],
                                 self.config['PATCH_SIZE'][0] // self.config['SCALE'] * self.config['PATCH_SIZE'][1] // self.config['SCALE'],
                                 self.config[self.config['MODEL_SIZE']]['FILTERS']])
        fs_T = tf.transpose(fs_r, [0,1,3,2])
        a_s = tf.linalg.matmul(fs_r, fs_T)

        ft_n = tf.linalg.normalize(ft)[0]
        ft_r = tf.reshape(ft_n, [len(self.config[self.config['TEACHER_SIZE']]['DIST_LAYERS']),
                                 self.config[self.config['MODE']]['BATCH_SIZE'],
                                 self.config['PATCH_SIZE'][0] // self.config['SCALE'] * self.config['PATCH_SIZE'][1] // self.config['SCALE'],
                                 self.config[self.config['TEACHER_SIZE']]['FILTERS']])
        ft_T = tf.transpose(ft_r, [0,1,3,2])
        a_t = tf.linalg.matmul(ft_r, ft_T)

        d = a_t - a_s
        norm = tf.math.reduce_mean(tf.math.abs(d), axis=[1,2,3])
        loss = tf.reduce_sum(norm)        
        #norm = tf.math.reduce_sum(tf.math.abs(d), axis=[0,2,3])
        #loss = tf.reduce_mean(norm)

        return loss
        
        
    def cokd_loss(self, source, target, margin):

        target = tf.math.maximum(target, margin)

        loss = tf.math.subtract(source,target)
        loss = tf.math.square(loss)

        tt = ((source > target) | (target > 0.0))
        tt = tf.cast(tt, dtype=tf.float32)
        loss = loss * tt

        return tf.math.reduce_sum(loss) / self.config[self.config['MODE']]['BATCH_SIZE'] / 1000.0
    
    
    def abkd_loss(self, source, target, margin):
        
        t1 = ((source > -margin) & (target <= 0))
        t1 = (source + margin) ** 2 * tf.cast(t1, dtype=tf.float32)

        t2 = ((source <= margin) & (target > 0))
        t2 = (source - margin) ** 2 * tf.cast(t1, dtype=tf.float32)

        loss = tf.math.abs(t1 + t2)
        return tf.math.reduce_sum(loss) / self.config[self.config['MODE']]['BATCH_SIZE'] / 1000.0

    
    def brkd_loss(self, source, target):
        nmse = tf.math.subtract(source,target)
        nmse = tf.math.square(nmse)
        nmse = tf.math.reduce_sum(nmse, axis=[0,2,3])
        loss = tf.reduce_mean(nmse) / tf.norm(target)
        return loss
    
    
    def evaluate(self, dataset):
        if self.config[self.config['MODE']]['METRIC'] == 'LPIPS':
            return evaluate(self.generator, dataset, metric=self.config[self.config['MODE']]['METRIC'], perc=self.lpips)
        else:
            return evaluate(self.generator, dataset, metric=self.config[self.config['MODE']]['METRIC'])
    
                                    
    def should_save(self, m):
        
        if not self.config['CHECKPOINT']:
            return False
        elif not hasattr(self, 'best_metr'):
            return True
        elif m > self.best_metr:
            return self.config[self.config['MODE']]['METRIC'] in ['PSNR', 'SSIM']
        else:
            return self.config[self.config['MODE']]['METRIC'] in ['NIQE', 'ERQA', 'LPIPS']
        