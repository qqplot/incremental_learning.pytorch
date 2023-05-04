network=RN50
python zero_train.py ${network} | tee logs/zero_train_${network}.txt
network=RN101
python zero_train.py ${network} | tee logs/zero_train_${network}.txt
network=RN50x4
python zero_train.py ${network} | tee logs/zero_train_${network}.txt
network=RN50x16
python zero_train.py ${network} | tee logs/zero_train_${network}.txt

# GPU memory issue
# network=RN50x64
# python zero_train.py ${network} | tee logs/zero_train_${network}.txt

network=ViT-B/32
python zero_train.py ${network} | tee logs/zero_train_ViT-B_32.txt
network=ViT-B/16
python zero_train.py ${network} | tee logs/zero_train_ViT-B_16.txt
network=ViT-L/14
python zero_train.py ${network} | tee logs/zero_train_ViT-L_14.txt
network=ViT-L/14@336px
python zero_train.py ${network} | tee logs/zero_train_ViT-L_14_336px.txt
