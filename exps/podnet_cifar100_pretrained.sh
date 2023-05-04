python3 -m inclearn --options options/podnet/podnet_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --fixed-memory \
    --device 9 --label podnet_cnn_cifar100_5steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_5steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_cnn_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --fixed-memory \
    --device 9 --label podnet_cnn_cifar100_5steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 10 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_5steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 5 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_10steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 5 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_10steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 2 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_25steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 2 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_25steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 1 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_50steps \
    --data-path data

python3 -m inclearn --options options/podnet/podnet_nme_cifar100_pretrained.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 1 --fixed-memory \
    --device 9 --label podnet_nme_cifar100_50steps \
    --data-path data

