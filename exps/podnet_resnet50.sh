device_id=9

# method=nme
method=cnn

# for increment in 10 1
# do

# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_resnet50.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label resnet50_cifar_${method}_inc_${increment} --data-path data --save-model task

# done


for increment in 5 2
do

python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_resnet50.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label resnet50_cifar_${method}_inc_${increment} --data-path data --save-model task

done