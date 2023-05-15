#!/bin/bash

for model in 'gan_generator_small_ablation_3_batchnorm_new.h5'
do
    for type in 'gan'
    do  
        for DATASET in 'Set5' 'Set14' 'B100' 'Manga109' 'Urban100'
        do    
            LOGFILE=logs/eval_${DATASET}.txt
            echo '' >> $LOGFILE
            echo $model $type $DATASET >> $LOGFILE
            python eval.py --dataset_hr /media/Datasets/super-resolution/${DATASET}/HR \
                           --dataset_lr /media/Datasets/super-resolution/${DATASET}/LR_bicubic \
                           --model ${model} --train_type $type --logfile $LOGFILE
        done
    done
done