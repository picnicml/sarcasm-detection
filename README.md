## Sarcasm Detection with Tensorflow and doddle-model
We embed 1 million Reddit comments with a [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175) and then use the encoded data to train a logistic regression classifier that detects sarcasm.

Steps:
- download the data from from [Kaggle](https://www.kaggle.com/danofer/sarcasm) and unpack into the root of this repository (directory name is `sarcasm`)
- run `pip install -r requirements.tx` to install Python dependencies (preferably in a virtual environment)
- run `python save_hub_module.py` to download a pretrained universal sentence encoder model from Tensorflow Hub
- run `sbt "runMain io.picnicml.doddlemodel.sarcasm.EmbedDataset.class sarcasm/train-balanced-sarcasm.csv sarcasm/train-balanced-sarcasm-embedded.csv"` to embed the text data
- run `sbt "runMain io.picnicml.doddlemodel.sarcasm.TrainClassifier.class sarcasm/train-balanced-sarcasm-embedded.csv logreg.model"` to train a classifier
