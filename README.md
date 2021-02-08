# CNN_Relation_Extraction
It contains the source code of my master's dissertation on the implementation of Extraction of relationships from unstructured data based on deep learning and distant supervision.

In this project, we provide our implementations of a low dimensional Convolutional Neural Network (CNN) for Relation_Extraction based in [deeplearning4nlp-tutorial](https://github.com/UKPLab/deeplearning4nlp-tutorial/tree/master/2017-07_Seminar/Session%203%20-%20Relation%20CNN/code), however we use dataset created by distant supervision.

![What is this](my-model-cnn.png)

The CNN architecture was implemented inspired by [Nguyen et al. 2015] and [Zeng et al., 2014].
We use a widely used dataset developed by Riedel 2010, available in [[Lin et al., 2016](https://github.com/thunlp/NRE)]. This dataset was generated by aligning Freebase relationships with the NYT corpus.

The 01-preprocess_freebase.py file format [Lin et al., 2016](https://github.com/thunlp/NRE) dataset.

# Reference
[[Lin et al., 2016](http://www.aclweb.org/anthology/P16-1200)] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. "Neural Relation Extraction with Selective Attention over Instances."

[[Nguyen et al. 2015](http://www.aclweb.org/anthology/W15-1506)] Nguyen, Thien Huu, and Ralph Grishman. "Relation Extraction: Perspective from Convolutional Neural Networks."

[[Zeng et al., 2014](http://www.aclweb.org/anthology/C14-1220)] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. "Relation classification via convolutional deep neural network."

[[Zeng et al.,2015](http://www.aclweb.org/anthology/D15-1203)] Daojian Zeng,Kang Liu,Yubo Chen,and Jun Zhao. "Distant supervision for relation extraction via piecewise convolutional neural networks."

# Environment settings
We used Python 3.5, Keras and TensorFlow, in Anaconda plataform with Ubuntu 16.04.
Keras with Python 3.6 not worked for me.

We recommend the Virtualenv installation.

Before installing Keras, please install one of its backend engines: TensorFlow, Theano, or CNTK. We recommend the TensorFlow backend.

01. Install [TensorFlow](https://www.tensorflow.org/install/)
02. Install [Keras](https://keras.io/#installation)
