device_id=2

# method=nme

# increment=10
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task

# increment=5
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task

# increment=2
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task

# increment=1
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
# python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task


method=cnn

increment=10
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task

increment=5
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task

increment=2
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task

increment=1
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task
python3 -m inclearn --options options/podnet/podnet_${method}_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml --initial-increment 50 --increment ${increment} --fixed-memory --device ${device_id} --label cifar_${method}_inc_${increment} --data-path data --save-model task

