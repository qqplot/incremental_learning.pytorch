gpu_id=9
lr=0.01
epoch=100
wd=0.01

ft_lr=0.05
ft_ep=20

backbone=res
increment=10

CUDA_VISIBLE_DEVICES=${gpu_id} python3 -m inclearn \
    --options options/pretrained/pretrained_${backbone}_cifar100.yaml options/data/cifar100_1order.yaml \
    --initial-increment 50 --increment ${increment} --fixed-memory --device 0 \
    --lr ${lr} --epochs ${epoch} --weight-decay ${wd} --ft_lr ${ft_lr} --ft_ep ${ft_ep} \
    --label cifar_${backbone}_inc_${increment} \
    --data-path data