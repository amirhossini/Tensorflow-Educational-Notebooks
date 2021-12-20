# Tensorflow Education Notebooks
__A collection of useful Tensorflow workflows and working examples accompanied with occassional mathematical & statistical concepts for more advanced topics__ <br>

_Parts of this collection can prove useful for preparation for TensorFlow Developer Certification Exam._ 

Amir Hossini <br>
2021 <br>
![image](https://user-images.githubusercontent.com/91625030/144555506-cdfd55e5-9fe7-478f-a7c6-1926229218ca.png)
_________________________________________________________________________________________________________________

[__Simple_DNN_and_CNN__](https://github.com/amirhossini/Tensorflow-Educational-Notebooks/tree/main/00-Simple_DNN_and_CNN) is a collection of simple Keras code and Tensorflow workflows for regression by DNN and image classification by CNN. DNN Regression example gives a workflow for dealing with tabular data, including useful helper functions for data preparation. The example of image classification by CNN involves vanilla Keras architectures with a workflow for utilization of (1) ImageDataGenerators for in-memory image augmentation and (2) callback function template for early stopping. <br>

[  CNN with Transfer Learning  ](https://github.com/amirhossini/Tensorflow-Educational-Notebooks/tree/main/01-CNN_w_TransferLearning) is a collection of Keras code and Tensorflow workflows for in-memory image augmentation as well as transfer learning for image classification. The pre-trained model _InceptionV3_ with custom pre-trained weights is used in transfer learning set-up. <br>

[  NLP_LSTM-GRU-CNN  ](https://github.com/amirhossini/Tensorflow-Educational-Notebooks/tree/main/02-NLP_LSTM-GRU-CNN) is a collection involving various different examples in application of deep learning models for Natural Language Processing (NLP). The provided workflows include (1) an example of processing a JSON data file for sarcasm classification and feeding it into a LSTM model with Embedding layer; (2) an example of Transfer Learning based on GloVe; and (3) IMDB movie reviews sentiment classification with examples of DNN and bi-directional LSTM with Embedding layers. <br>

[  TimeSeries_CNN-LSTM  ](https://github.com/amirhossini/Tensorflow-Educational-Notebooks/tree/main/03-TimeSeries_CNN-LSTM) is a collection involving application of deep learning models for time series forecasting on Sunspot data set, which involves the application of stacked LSTM and dense layer architecture as well as example of learning rate optimization through the introduction of a learning rate scheduler.  A helper function for creation of windowed dataset based on window size and batch size is also given. <br>

[  Tensorflow_Prob_Distributions  ](https://github.com/amirhossini/Tensorflow-Educational-Notebooks/tree/main/21-TensorFlow_Prob_Distributions) is a collection of intorductory material to probability distributions (univariate, multivariate) as well as examples of creation of multivariate distributions from univariates using _Independent_ distribution and shifting batch dimensions to event dimensions using _reinterpreted_batch_ndims_, all available in _tensorflow-probability_ library. A good example of training trainable variables in Tensorflow-probability in the context of Navie Bayes & Logistic Regression classifiers is also provided.

[__Bayesian_NN__](https://github.com/amirhossini/Tensorflow-Educational-Notebooks/tree/main/22-Bayesian_NN) is a collection of theory and working examples on maximum likelihood estimation and Bayesian Neural Networks, respectively. The working examples include Bayesian Neural Networks for linear and nonlinear regression as well as 1D Convolutional Neural Networks in a classification context, where aleatoric and epistemic uncertainties are addressed. 
