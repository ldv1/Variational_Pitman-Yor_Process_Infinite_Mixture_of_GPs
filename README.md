# Variational Pitman-Yor Process Infinite Mixture of GPs

This is a slightly different infinite mixture of GPs than the one described by Sun and Xu in
[Variational Inference for Infinite Mixtures of Gaussian Processes With Applications to Traffic Flow Prediction](https://www.researchgate.net/publication/224204579_Variational_Inference_for_Infinite_Mixtures_of_Gaussian_Processes_With_Applications_to_Traffic_Flow_Prediction) based partly on the PYP-GP regression model of
Chatzis and Demiris in [Nonparametric Mixtures of Gaussian Processes With Power-Law Behavior](https://www.researchgate.net/publication/260501688_Nonparametric_Mixtures_of_Gaussian_Processes_With_Power-Law_Behavior).

Some useful tricks are borrowed from Titsias and Lazaro-Gredilla in [Spike and Slab Variational Inference for Multi-Task and Multiple Kernel Learning](https://papers.nips.cc/paper/4305-spike-and-slab-variational-inference-for-multi-task-and-multiple-kernel-learning).

## Code
The model is created, initialized and trained in `imgpTrain`.
Without proper initialization, variational methods can be easily trapped into local optima. So it pays off to initialize the model in a sensible way to overcome this issue. Various initialization strategies are implemented.

The lower bound is computed in `imgpLowerBound'.

Prediction of the mean and variance at unseen inputs is done in `imgpPredict`.

## Usage
`demimgp_sine` illustrates how it works for the sine function with varying frequency. Results are compared against the vanilla GP equipped with the squared exponential covariance function.

## Dependencies
The code should run on any non-ancient version of Matlab/Octave.

All third-party libraries are included:
* The implementation builds upon the package [GPML](http://www.gaussianprocess.org/gpml/code/matlab/doc/) of Rasmussen and Nickisch
when it goes to building the covariance matrices and optimizing the hyperparameters of the kernels. An old version of GPML can be found under `misc_toolbox/gpml`. The code is very unlikely to work with any recent GPML library.
* The routines for Gaussian Mixture Model (GMM) clustering are part of [NETLAB](http://www.aston.ac.uk/eas/research/groups/ncrg/resources/netlab/) developed by Nabney. They can be found under `misc_toolbox/netlab`.
* The Matlab/Octave routine `psin` written by Godfrey computes any polygamma function. It is part of the package [Special Functions math library](http://fr.mathworks.com/matlabcentral/fileexchange/978-special-functions-math-library?requestedDomain=www.mathworks.com). The file psin.m can be found under `misc_toolbox/`.

## Acknowledgment
It is no coincidence that our implementation is reminiscent of the code developed by Titsias and Lazaro-Gredilla.
Although their model was quite different,
our bottom algorithm shares the same structure (EM updates),
and so we could adopt and adapt the structure of their code.
We are indebted to Titsias and Lazaro-Gredilla for putting [the full version of their code online](https://github.com/melihkandemir/eventdetector/tree/master/varsparse_mtmkl), to Godfrey for the Matlab / Octave routine to compute the polygamma functions and to Nabney for the Matlab Toolbox NETLAB.

## License
All third-party libraries are subject to their own license.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.
