gpu_id=9

# network=RN50
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_${network}.txt
# network=RN101
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_${network}.txt
# network=RN50x4
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_${network}.txt
# network=RN50x16
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_${network}.txt

# GPU memory issue
# network=RN50x64
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_${network}.txt

# network=ViT-B/32
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_ViT-B_32.txt
# network=ViT-B/16
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_ViT-B_16.txt
# network=ViT-L/14
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_ViT-L_14.txt
# network=ViT-L/14@336px
# python zero_train_20samples.py ${network} | tee exps/zero_train/zero_train_ViT-L_14_336px.txt



network=resnet18
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 500 | tee exps/zero_train/full_samples/zero_train_${network}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 20 | tee exps/zero_train/20_samples/zero_train_${network}.txt
network=resnet50
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 500 | tee exps/zero_train/full_samples/zero_train_${network}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 20 | tee exps/zero_train/20_samples/zero_train_${network}.txt

network=resnet18_fc
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 500 | tee exps/zero_train/full_samples/zero_train_${network}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 20 | tee exps/zero_train/20_samples/zero_train_${network}.txt
network=resnet50_fc
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 500 | tee exps/zero_train/full_samples/zero_train_${network}.txt
CUDA_VISIBLE_DEVICES=${gpu_id} python zero_train.py ${network} 20 | tee exps/zero_train/20_samples/zero_train_${network}.txt