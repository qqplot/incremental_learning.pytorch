device_id=3


for method in nme #cnn
do
for increment in 50 #10 5 2 1
do
for lr in 0.1 0.05 0.01 0.005 0.001
do
for wd in 0.0005 0.0001
do

python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_resnet50.yaml options/data/cifar100_1order.yaml \
    --initial-increment 50 --increment ${increment} --fixed-memory \
    --device ${device_id} --lr ${lr} --weight-decay ${wd} \
    --label tuning_cifar_${method}_inc_${increment}_lr_${lr}_wd_${wd} \
    --data-path data

done
done
done
done