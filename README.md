# POPCORN: An Algorithm to Certify the Robustness of Recurrent Neural Networks

*POPCORN* (**P**ropagated-**o**ut**p**ut **C**ertified R**o**bustness for **RN**Ns) is a general algorithm to certify robustness of recurrent neural networks (RNNs), including vanilla RNNs, LSTMs, and GRUs. It provides certified bounds for RNNs by bounding the non-linear activations in RNNs with linear functions. POPCORN is

* **Novel** - it is a general framework, which is, to the best of our knowledge, the \textbf{first} work to provide a certified robustness evaluation for RNNs with robustness guarantees.
* **Effective** - it can handle complicated RNN structures besides vanilla RNNs, including LSTMs and GRUs that contain challenging coupled nonlinearities.
* **Versatile** - it can be widely applied in applications including but not limited to computer vision, natural language processing, and speech recognition. Its robustness certification on individual steps in the input sequence can lead to new insights. 

This repo intends to release code for our work:


Ching-Yun Ko\*, Zhaoyang Lyu\*, Tsui-Wei Weng, Luca Daniel, Ngai Wong and Dahua Lin,"POPCORN: Certifying Robustness of Recurrent Neural Networks", ICML 2019

\* Equal contribution

We are still working on refactoring our code and it's expected to be released in June 2019.