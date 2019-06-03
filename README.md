# POPQORN: An Algorithm to Quantify Robustness of Recurrent Neural Networks

*POPQORN* (**P**ropagated-**o**ut**p**ut **Q**uantified R**o**bustness for **RN**Ns) is a general algorithm to quantify robustness of recurrent neural networks (RNNs), including vanilla RNNs, LSTMs, and GRUs. It provides certified lower bound of minimum adversarial distortion for RNNs by bounding the non-linear activations in RNNs with linear functions. POPQORN is

* **Novel** - it is a general framework, which is, to the best of our knowledge, the **first** work to provide a quantified robustness  evaluation with guarantees for RNNs.
* **Effective** - it can handle complicated RNN structures besides vanilla RNNs, including LSTMs and GRUs that contain challenging coupled nonlinearities.
* **Versatile** - it can be widely applied in applications including but not limited to computer vision, natural language processing, and speech recognition. Its robustness quantification on individual steps in the input sequence can lead to new insights. 

This repo intends to release code for our work:


Ching-Yun Ko\*, Zhaoyang Lyu\*, Tsui-Wei Weng, Luca Daniel, Ngai Wong and Dahua Lin,"POPQORN: Quantifying Robustness of Recurrent Neural Networks", ICML 2019

\* Equal contribution

Setup
----------------------------------------------------------------

The code is tested with python 3.7.3, Pytorch 1.1.0 and CUDA 9.0. Run the following
to install pytorch and its CUDA toolkit:

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

Then clone this repository:

```
git clone https://github.com/ZhaoyangLyu/POPQORN.git
cd POPQORN
```

Experiment 1: Quantify Robustness for Vanilla RNN
---------------------------------------------------------------
All experiments of this part will be conducted in the folder `vanilla_rnn`.

```
cd vanilla_rnn
```

### Step 1: Train an RNN Mnist Classifier

We first train an RNN to classify images in the Mnist datatset. We will cut each image into several slices and feed them to the RNN. Each slice will be considered as the input of the RNN at each time step. By default, we cut each 28 × 28 image into 7 slices. Each slice is of shape 4 × 28. As for the RNN, we set the hidden size to 64 and use **tanh** nonlineararity. Run the following command to train the RNN:

```
python train_rnn_mnist_classifier.py
```

Run
```
python train_rnn_mnist_classifier.py --cuda
```

if you want to enable gpu usage.

The trained model will be save as `POPQORN/models/mnist_classifier/rnn_7_64_tanh/rnn`.

You can train more different models by changing the number of slices to cut each mnist image into, the hidden size and nonlinearity of the RNN.


### Step 2: Compute Certified Bound

For each input sequence, we will compute a certified bound within which the true label output value is larger than the target label output value, namely, the targeted attack won't succeed. Note that in this experiment we have assumed perturbations to be uniform across input frames. 
Run
```
python bound_vanilla_rnn.py
```

to compute the certified bound. By default, this file will read the pretrained model `POPQORN/models/mnist_classifier/rnn_7_64_tanh/rnn`, randomly sample 100 images, randomly choose target labels , and then compute certifed bounds for them (only for those images that are correctlly classified by the pretrained model). Each certified bound is the radius of the *l-p* (2-norm by default) ball centered at the original image slice. 

The script will print the minimum, mean, maximum and standard deviation of the computed certified bounds at the end. The complete result will be saved to `POPQORN/models/mnist_classifier/rnn_7_64_tanh/2_norm_bound/certified_bound`.

You can compute bounds for different models by setting the values of the arguments **work_dir** and **model_name**. Also remember to set **hidden_size**, **time_step** and **activation** accordingly to make them consistent with your specifed pretrained model. 

Experiment 2: Quantify Robustness for LSTM (Coming soon...)
------------------------------------------------------------