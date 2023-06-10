gpu_id=3
lr=0.01
epoch=20
wd=0.01

ft_lr=0.05
ft_ep=20

k_orth=5

backbone=vit
increment=50

for increment in 10 5 2 1
do

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m inclearn \
    --options options/pretrained/pretrained_${backbone}_cifar100.yaml options/data/cifar100_1order.yaml \
    --initial-increment 50 --increment ${increment} --fixed-memory --device 0 \
    --lr ${lr} --epochs ${epoch} --weight-decay ${wd} --ft_lr ${ft_lr} --ft_ep ${ft_ep} --k_orth ${k_orth} \
    --label full_inc${increment}_ep${epoch}_lr${lr}_wd${wd}-ft_ep${ft_ep}_lr${ft_lr}-k${k_orth} \
    --data-path data --full_track

done
