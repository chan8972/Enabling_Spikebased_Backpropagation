# Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures

This repo contains the code associated with [Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures](https://www.frontiersin.org/articles/10.3389/fnins.2020.00119/abstract). This code has most recently been tested with Python 2.7.15 and Pytorch 0.3.1.



# Introduction

Spiking Neural Networks (SNNs) have recently emerged as a prominent neural computing paradigm. However, the typical shallow SNN architectures have limited capacity for expressing complex representations, while training deep SNNs using input spikes has not been successful so far. Diverse methods have been proposed to get around this issue such as converting off-line trained deep Artificial Neural Networks (ANNs) to SNNs. However, the ANN-SNN conversion scheme fails to capture the temporal dynamics of a spiking system. On the other hand, it is still a difficult problem to directly train deep SNNs using input spike events due to the discontinuous, non-differentiable nature of the spike generation function. To overcome this problem, we propose an approximate derivative method that accounts for the leaky behavior of LIF neurons. This method enables training deep convolutional SNNs directly (with input spike events) using spike-based backpropagation. Our experiments show the effectiveness of the proposed spike-based learning strategy on deep networks (VGG and Residual architectures) by achieving the best classification accuracies in MNIST, SVHN and CIFAR-10 datasets compared to other SNNs trained with a spike-based learning. Moreover, we analyze sparse event-based computations to demonstrate the efficacy of the proposed SNN training method for inference operation in the spiking domain.


# Testing and Training
The pretrained models are attached [__**here**__](https://www.dropbox.com/sh/vvq9afkq90refka/AAAIEnyBZ_wO7eM510GCyZ8ta?dl=0). The basic syntax is:
```python cifar10_ResNet11.py --resume model_bestT1_cifar10_r11.pth.tar --evaluate```
This will evaluate the model on ResNet 11.

To train a new model from scratch, the basic syntax is:
```python cifar10_ResNet11.py```


## Citations

If you find this code useful in your research, please consider citing:

```
@article{lee2020enabling,
  doi={10.3389/fnins.2020.00119}
  title={Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures},
  author={Lee, Chankyu and Sarwar, Syed Shakib and Panda, Priyadarshini and Srinivasan, Gopalakrishnan and Roy, Kaushik},
  journal={Frontiers in Neuroscience},
  volume={14},
  pages={119},
  year={2020},
  url={https://www.frontiersin.org/article/10.3389/fnins.2020.00119},
  publisher={Frontiers in Neuroscience}
}
```


## Authors

Chankyu Lee*, Syed Shakib Sarwar*, Priyadarshini Panda, Gopalakrishnan Srinivasan, and Kaushik Roy, (* Equal contributors)
