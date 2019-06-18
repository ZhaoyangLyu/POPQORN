# POPQORN: An Algorithm to Quantify Robustness of Recurrent Neural Networks

*POPQORN* (**P**ropagated-**o**ut**p**ut **Q**uantified R**o**bustness for **RN**Ns) is a general algorithm to quantify robustness of recurrent neural networks (RNNs), including vanilla RNNs, LSTMs, and GRUs. It provides certified lower bound of minimum adversarial distortion for RNNs by bounding the non-linear activations in RNNs with linear functions. POPQORN is

* **Novel** - it is a general framework, which is, to the best of our knowledge, the **first** work to provide a quantified robustness  evaluation with guarantees for RNNs.
* **Effective** - it can handle complicated RNN structures besides vanilla RNNs, including LSTMs and GRUs that contain challenging coupled nonlinearities.
* **Versatile** - it can be widely applied in applications including but not limited to computer vision, natural language processing, and speech recognition. Its robustness quantification on individual steps in the input sequence can lead to new insights. 

This repo intends to release code for our work:


Ching-Yun Ko\*, Zhaoyang Lyu\*, Tsui-Wei Weng, Luca Daniel, Ngai Wong and Dahua Lin, ["POPQORN: Quantifying Robustness of Recurrent Neural Networks"](https://arxiv.org/abs/1905.07387), ICML 2019

\* Equal contribution

Updates
----------------------------------------------------------------

- Jun 7, 2019: Initial release. Release the codes for quantifying robustness of vanilla RNNs and LSTMs.
- To do: Add descriptions for experimental results.


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

Experiment 2: Quantify Robustness for LSTM
------------------------------------------------------------
All experiments of this part will be conducted in the folder `lstm`.
```
cd lstm
```
### Step 1: Train an LSTM News Title Classifier

cd to the folder `NewsTitleClassification`.

We will train an LSTM classifier on TagMyNews dataset. TagMyNews is a dataset consisting of 32,567 English news items grouped into 7 categories: Sport, Business, U.S., Health, Sci&Tech, World, and Entertainment. Each news item has a news title and a short description. We train an LSTM to classify the news items into the 7 categories according to their news titles.

The news items have been preprocessed following the instructions in the github repo https://github.com/isthegeek/News-Classification. The processed data is stored in the file `test_data_pytorch.csv` and `train_data_pytorch.csv`.

Before training the LSTM, run the following commands to install the dependencies.

```
pip install torchtext
pip install spacy
python -m spacy download en
```

Then run
```
python train_model.py
```
to train an LSTM classifier.

Run
```
python train_model.py --cuda
```
if you want to enable gpu usage.

The script will download the pretrained word embedding vector from “glove.6B.100d". By default, the trained LSTM will be saved as `POPQORN/models/news_title_classifier/lstm_hidden_size_32/lstm`.

You can train more different LSTMs by changing the hidden size of the LSTM.

### Step 2: Compute Certified Bound

Next, for each news title, we will compute the untargeted POPQORN bound on every individual word, and call the words with minimal bounds sensitive words.​ 

```
cd ..
```
back to the folder `lstm`.

Run
```
python bound_news_classification_lstm.py
```
to compute the certified bound. By default, this file will read the pretrained model `POPQORN/models/news_title_classifier/lstm_hidden_size_32/lstm`, randomly sample 128 news titles, and then compute certified *l-2* balls for them (only for those news that are correctlly classified by the pretrained model). The script will save useful information to the diretory `POPQORN/models/news_title_classifier/lstm_hidden_size_32/2_norm`.
The sampled news titles will be saved as `input`. The complete log of the computation process will be saved as `logs.txt` and the computed bound will be saved as `bound_eps`. 

By default, the script will use 2D planes to bound the 2D nonlinear activations in the LSTM. You can also use 1D lines or constants to bound the 2D nonlinear activations by running the following:
```
python bound_news_classification_lstm.py --use-1D-line
```
or
```
python bound_news_classification_lstm.py --use-constant
```

You can also compute bounds for different models by setting the values of the arguments **work_dir** and **model_name**. Also remember to set **hidden_size** accordingly to make them consistent with your specifed pretrained model. 
