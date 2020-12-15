## AUTOENCODER

An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal “noise” (Wikipedia).

![alt text](https://www.researchgate.net/publication/318204554/figure/fig1/AS:512595149770752@1499223615487/Autoencoder-architecture.png)

Pytorch is the library used. First tests are done with MNIST dataset.

### Variational Autoencoder

A Variational Autoencoder (VAE) create a distribution in its latent space, no a vector. This means that the latent space is continuous. Thus, it can be a generative model just sampling from the known latent space distribution.

<p align="center">
<img src="https://miro.medium.com/max/3080/1*82EghOQR2Z5uuwUjFiVV2A.png" width="300">
  </p>

It's based on variational interference, and it's loss function is the Lower Varitional Bound, or Evidence Lower Bound (ELBO). This ELBO is the sum of two losses: Reconstruction Loss and KL-Divergence. 
Reconstruction Loss is the difference between the original data and the reconstructed data. These loss is directly related with the distribution P(x|x̂) (being x the original data, and x̂ the reconstructed data) is Gaussian, this loss would be MSE; if the distribution is Bernoulli, the loss would be similar to Cross-Entropy.
The second part of ELBO is the KL-Divergence, which is a measure of the dissimilarity of two distributions. This time it's between the distribution q(z|x) and our known distribution (let's say Gaussian). Thus, we want to minimize the difference between the real distribution of the encoded data, and the distribution we created in our latent space.

At the end, this means that the loss function of a VAE is (the difference between the reconstructed data and the original data) + (difference between real unknown distribution and the latent space distribution that we create).

#### Reparametrization trick

Neural Networks are completely deterministic, that's why backpropagation can be applied. So, how can we perfom backpropagation in a stochastic architecture. That's the reparametrization trick. Let's say our distribution is Gaussian. If we sample from our distribution, the result would be X = nu + ((random_number) * std). 
Reparametrization trick consists on isolate the stochastic part of the VAE in just the random number, but keep as deterministic variables, the nu (mean) and the std (standard deviation).
