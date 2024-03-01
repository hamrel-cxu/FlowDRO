# FlowDRO
Official implementation of "Flow-based Distributionally Robust Optimization", which is published in *IEEE Journal on Selected Areas in Information Theory---Information-Theoretic Methods for Trustworthy and Reliable Machine Learning*.

Please direct implementation inquiries to cxu310@gatech.edu.

## Pre-requisites
```
pip install -r requirements.txt
```

## Usage
We provide the complete code to reproduce the training of robust classifier on MNIST digits classification, where our FRM is compared against WRM in Fig 6(a-b) of https://arxiv.org/pdf/2310.19253.pdf.

* To train on binary MNIST digits
```
bash train_eval_mnist_binary.sh
```

* To train on full MNIST digits
```
bash train_eval_mnist_full.sh
```

## Citation
```
@ARTICLE{xu2024flow,
    author={Xu, Chen and Lee, Jonghyeok and Cheng, Xiuyuan and Xie, Yao},
    journal={IEEE Journal on Selected Areas in Information Theory}, 
    title={Flow-Based Distributionally Robust Optimization}, 
    year={2024},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/JSAIT.2024.3370699}
}
```

## Illustration

Our flow models yield gradual and meaningful changes from the nominal distribution $P$ to the least favorable distribution $Q$, from which adversarial samples can be obtained.

**Illustration on MNIST**
<p align="center">
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/MNIST_tsne.png" width="600" height="450"/>
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/MNIST_traj.png" width="200"  height="450"/>
</p>

**Illustration on CIFAR10**
<p align="center">
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/Cifar10_traj.png" width="800" height="450" />
</p>