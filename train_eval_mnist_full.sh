#!/bin/bash
for run in 0 1 2
do
    for mtd in 'WRM' 'FRM'
    do
        echo "Running with --mtd $mtd at run $run"
        python train_mnist_models_torch_full.py --mtd $mtd --run $run
        echo "Attack at run $run"        
        python eval_mnist_models_torch.py --mtd $mtd --full 1 --run $run
    done
done
echo "Final save plot"
python plot_torch.py --full 1