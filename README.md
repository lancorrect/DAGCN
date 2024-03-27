# DAGCN

The implementation of "DAGCN: Distance-based and Aspect-oriented Graph Convolutional Network for Aspect-based Sentiment Analysis" accepted by NAACL 2024.

# Requirements

To install requirements, run `pip install -r requirements.txt`.

# Preparation

1. Download and unzip Glove vectors (`glove.840B.300d.zip`) from https://nlp.stanford.edu/projects/glove/ and put it into `./glove` directory.
2. Prepare vocabulary with `sh build_vocab.sh`

# Training

To train the DAGCN model, run:

`sh run.sh`

# Credits

The code and datasets in this repository are based on [DualGCN](https://github.com/CCChenhao997/DualGCN-ABSA) and [SSEGCN](https://github.com/zhangzheng1997/SSEGCN-ABSA).