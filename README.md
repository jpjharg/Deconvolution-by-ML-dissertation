#### Final year project to deconvolute data using machine learning methods to aid with research into fermi Surfaces. 

* Initial steps: studying denoising methods with the Richardson-Lucy algorithm
  *   An iterateive formula reqiring the measured and PSF to remove noise from simple functions, but is prone to amplifying noise and creating artefacts
* Progressed to developing simple neural networks to manipulate similarly to the RL algorithm to try and achieve better results
  * Started with a MLP which produced poor results, then moved on to a CNN which produced far more accurate results
* Currently combining the neural network with the RL algorithm to achieve the best results
