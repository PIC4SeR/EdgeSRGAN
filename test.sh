#!/bin/bash

# i=0
# for type in 'PSNR' 'GAN'
#     do
#     model="ESRGAN"
#     echo '' >> result_log.txt
#     echo $model $type $i >> result_log.txt
#     echo '' >> result_log.txt
#     for DATASET in 'Set5' 'Set14' 'B100' 'Manga109' 'Urban100'
#     do    python eval.py --dataset_hr /media/Datasets/super-resolution/${DATASET}/HR \
#                         --dataset_lr ../ESRGAN/results/${type}/${DATASET} --no-predict \
#                         --model ${model} --train_type $type --excel_row $i
#     done
#     ((i=i+1))
# done
# i=8
# for model in 'pix' #'tiny' 'small' 'base' 'pix'
# do
#     for type in 'pre' 'gan'
#     do
#         echo '' >> result_log.txt
#         echo $model $type $i >> result_log.txt
#         echo '' >> result_log.txt
#         for DATASET in 'Set5' 'Set14' 'B100' 'Manga109' 'Urban100'
#         do    
#             python eval.py --dataset_hr /media/Datasets/super-resolution/${DATASET}/HR \
#                            --dataset_lr /media/Datasets/super-resolution/${DATASET}/LR_bicubic \
#                            --model ${model} --train_type $type --excel_row $i            
#         done
#         ((i=i+1))
#     done
# done

# for type in 'pre' 'gan'
#     do
#     model="SRGAN"
#     echo '' >> result_log.txt
#     echo $model $type $i >> result_log.txt
#     echo '' >> result_log.txt
#     for DATASET in 'Set5' 'Set14' 'B100' 'Manga109' 'Urban100'
#     do    python eval.py --dataset_hr /media/Datasets/super-resolution/${DATASET}/HR \
#                          --dataset_lr ../super-resolution/results/${type}/${DATASET} --no-predict \
#                          --model ${model} --train_type $type --excel_row $i
#     done
#     ((i=i+1))
# done

i=96
for model in 'gan_generator_gridsearch_2022-10-05_23:53:52.1912986.h5' #'tiny' 'small' 'base' 'pix'
do
    for type in 'pre' #'pre' 'gan'
    do
        echo '' >> result_log.txt
        echo $model $type $i >> result_log.txt
        echo '' >> result_log.txt
        for DATASET in 'Set5' 'Set14' 'B100' 'Manga109' 'Urban100'
        do    
            python eval.py --dataset_hr /media/Datasets/super-resolution/${DATASET}/HR \
                           --dataset_lr /media/Datasets/super-resolution/${DATASET}/LR_bicubic \
                           --model ${model} --train_type $type --excel_row $i          
        done
        ((i=i+1))
    done
done



# model='new_model.h5'
# type='gan'
# echo '' >> result_log.txt
# echo $model $type >> result_log.txt
# echo '' >> result_log.txt
# python eval.py --dataset_hr /home/simone/sr-edge/dataset/div2k/images/DIV2K_valid_HR \
#                --dataset_lr /home/simone/sr-edge/dataset/div2k/images/DIV2K_valid_LR_bicubic \
#                --model ${model} --train_type $type --no-excel          


# i=16
# type="visual"
# model="AGD"
# echo '' >> result_log.txt
# echo $model $type $i >> result_log.txt
# echo '' >> result_log.txt
# for DATASET in 'Set5' 'Set14' 'B100' 'Manga109' 'Urban100'
# do    python eval.py --dataset_hr /media/Datasets/super-resolution/${DATASET}/HR \
#                      --dataset_lr ../AGD/AGD_SR/search/results/${type}/${DATASET} --no-predict \
#                      --model ${model} --train_type $type --excel_row $i
# done