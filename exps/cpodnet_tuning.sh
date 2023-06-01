device_id=4
increment=50

base=vit

# for lr in 1 0.1 0.01
for lr in 0.001 0.0001 0.00001
do
for wd in 0.00005
do

# python3 -minclearn --options options/cpodnet/cpodnet_cnn_cifar100_${base}.yaml options/data/cifar100_1order.yaml \
#     --initial-increment 50 --increment ${increment} --fixed-memory \
#     --device ${device_id} --label tuning_cpodnet_${base}_inc_${increment}_lr_${lr}_wd_${wd} \
#     --data-path data

CUDA_VISIBLE_DEVICES=${device_id} python3 -minclearn --options options/cpodnet/cpodnet_cnn_cifar100_${base}.yaml options/data/cifar100_1order.yaml \
    --initial-increment 50 --increment ${increment} --fixed-memory \
    --label tuning_cpodnet_${base}_inc_${increment}_lr_${lr}_wd_${wd} \
    --data-path data

done
done
