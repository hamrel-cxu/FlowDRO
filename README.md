# FlowDRO
Official implementation of "[Flow-based Distributionally Robust Optimization](https://arxiv.org/pdf/2310.19253.pdf)", which is published in *IEEE Journal on Selected Areas in Information Theory---Information-Theoretic Methods for Trustworthy and Reliable Machine Learning*.

Please direct implementation inquiries to cxu310@gatech.edu.

## Pre-requisites
```
pip install -r requirements.txt
```

## Usage
We provide the complete code to reproduce the training of robust classifier on MNIST digits classification, where our FRM is compared against WRM as shown below (also see Fig 6 of https://arxiv.org/pdf/2310.19253.pdf). In this case, lower prediction errors are better. 

* To train and test on binary MNIST digits
```
bash train_eval_mnist_binary.sh
```

* To train and test on full MNIST digits
```
bash train_eval_mnist_full.sh
```

The left figure shows results on binary MNIST digits and the right figures shows results on full MNIST digits.

<p align="center">
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/binary_mnist.png" width="400" height="200"/>
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/full_mnist.png" width="400"  height="200"/>
</p>


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

Our trained flow models also yield continuous and meaningful changes from samples in the nominal distribution $P$ to those in the least favorable distribution $Q^*$.

**On MNIST**

The left figure visualizes the perturbation trajectories from digits 0 to 8 under 2D T-SNE embedding; in this case, digits 8 are adversarial samples based on digits 0, which are the nominal samples.

The right figure shows the corresponding trajectory in pixel space, along with the corresponding movement between original and perturbed images over integration steps of the continuous-time flow model.
<p align="center">
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/MNIST_tsne.png" width="450" height="450"/>
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/MNIST_traj.png" width="200"  height="450"/>
</p>

**Oon CIFAR10**

The trajectory in pixel space from nominal samples (top row) to adversarial samples (bottom row) via our flow model. Captions on the top and bottom indicate predictions by a pre-trained image classifiers on these samples.
<p align="center">
  <img src="https://github.com/hamrel-cxu/FlowDRO/blob/main/figs/Cifar10_traj.png" width="600" height="450" />
</p>
