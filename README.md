# Detecting Advesarial images given as input to the CNN classifier
Adversarial images can fool the classification network(i.e. force it to misclassify an image) and thus can lead to disastrous results e.g. In case of An autonomous car, a "STOP" sign might not be classified as "Stop". In this project we have tried to develop a robust classification system which can detect whether the input image given to the CNN is a real or adversarial. If the image is real only then the system should consider it for any action else discard it. 

# Requirements
Pytorch, Python3, matplotlib

# Description
adv_mnist_digit.ipynb, adv_mnist_fashion.ipynb, adv_cifar.ipynb  are the main python notebooks. These include Adversarial examples generation and adversarial example detection method for corresponding datasets (MNIST, fashion MNIST, CIFAR10)

# References
https://medium.com/@ml.at.berkeley/tricking-neural-networks-create-your-own-adversarial-examples-a61eb7620fd8
https://github.com/dangeng/Simple_Adversarial_Examples
