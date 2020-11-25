# MULA: A Just-In-Time Multi-Labeling System for Issue Reports

## How to install app 

You can install MULA Label Bot on this [link](https://github.com/apps/mula-label-bot). 

To install, you need to sign in your GitHub account, and click 'install' button on this page. And then, you can select the repository you want to install. 

MULA Label Bot will listen opened issue events from your repository, give Just-In-Time multi-labeling service.


## Project structure

```
MULA-data
│   README.md    
│
└───raw_data
│   │   issue_data.csv
│   │   figure_6.csv
│   │   figure_7.xlsx
│   │   figure_8.txt
│   │   RQ2_full_metric_data.csv
│   │   feedback_data.csv
```

### benchmark set

This folder contains our benchmark set `issue_data.csv` , which contains 81601 entries.

In each entry, we present metadata of an issue, which are 'url', 'id', 'title', 'body', 'label':
- 'url' is the url of an issue on GitHub. 
- 'id' is the identification number of an issue.
- 'title' is the title of an issue.
- 'body' is the body of an issue.
- 'label' is the original label of an issue.

We also record the augmented labels of each issue, which contain 'bug', 'enhancement', 'question', 'ui', 'design', 'database', 'client', 'server', 'document', 'security' and 'performance'. 

In each cell of those label columns: 
- If the number is 0, it means this issue doesn't have this label, or its synonym originally, and after label augmentation, we still can't apply this label to this issue with enough confidence. 
- If the number is 1, it means this issue has this label originally. 
- If the number is 2, it means this issue doesn't have this label originally, but after label augmentation, we apply this label to this issue with enough confidence.

### raw data for experiment

- `figure_6.csv` records the raw data for Figure 6, that is, the F1 values comparision between MULA and the other methods in our paper.
- `figure_7.xlsx ` records the raw data for Figure 7, that is, the f1 values for MULA and the other methods in our paper, running on the 10 subsets.
- `figure_8.txt` contains the raw data for Figure 8, that is, the output threshold adjustment for two labels, 'bug' and 'document', in our paper.
- `RQ2_full_metric_data.csv` is the file containing all the metric values from RQ2.



### user feedback 

`feedback_data.csv`, we present the feedback data we collected. In each entry of this csv file, you can see the issue title, issue body, predicted label or unpredicted label, likes and dislikes, confidence score, and its label status of each MULA prediction comment.


## Configurations to repeat the experiment

### RQ1

We use fasttext python toolkit from fasttext.cc as our core model. For the experiment without GloVe, you can set your fasttext parameters as following in experiments of RQ1.

`lr=0.1, epoch=100, bucket=2000000, wordNgrams=5, dim=50, loss='ova', minCount=100, maxn=3, minn=3`

For the experiment with GloVe, please refer to this public [link](http://nlp.stanford.edu/data/glove.6B.zip), and choose the `glove.6B.50d.txt` file in this zip file as pre-trained vector data. In fasttext, please add a new parameter `pretrainedVectors="glove.6B.50d.txt"` to parameters above. Other parameters remain the same.

### RQ2

In RQ2, please use those parameters:

`lr=0.1, epoch=100, bucket=2000000, wordNgrams=5, dim=50, loss='ova', minCount=100, maxn=3, minn=3`

### RQ3

In this RQ, we compare efficiency of several methods. And we will show parameters configuration of those methods. 

We used toolkit from `sklearn` and `skmultilearn` in first three method. 

- BR + RandomForest

We just use the default configuration from `sklearn.ensemble` of `sklearn` package.

Our used classifier:

```
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
classifier = BinaryRelevance(
    classifier = RandomForestClassifier(),
    require_dense = [False, True]
)
```

- BR+SVM

```
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
classifier = BinaryRelevance(
    classifier = SVC(kernel='linear',decision_function_shape='ovo'),
    require_dense = [False, True]
)
```

- BR+kNN

```
from skmultilearn.adapt import BRkNNaClassifier
classifier = BRkNNaClassifier()
```

- textCNN

For textCNN, we use `keras` as our toolkit. Following are key parameters of textCNN.

```
EVALUATE_EVERY: 1000
NUM_EPOCHS: 100
BATCH_SIZE: 1024
CHECKPOINT_EVERY: 1000
DECAY_RATE: 0.95
DECAY_STEPS: 5000
LEARNING_RATE: 0.001
PAD_SEQ_LEN: 200
EMBEDDING_DIM: 50
FILTER_SIZES: 3,4,5
NUM_FILTERS: 128
FC_HIDDEN_SIZE: 1024
DROPOUT_KEEP_PROB: 0.5
THRESHOLD: 0.5
``` 

- textRNN

For textRNN, we use `keras` as our toolkit. Following are key parameters of textRNN.

```
EVALUATE_EVERY: 1000
NUM_EPOCHS: 100
BATCH_SIZE: 1024
CHECKPOINT_EVERY: 1000
DECAY_RATE: 0.95
DECAY_STEPS: 5000
LEARNING_RATE: 0.001
PAD_SEQ_LEN: 200
EMBEDDING_DIM: 50
LSTM_HIDDEN_SIZE: 256
FC_HIDDEN_SIZE: 1024
DROPOUT_KEEP_PROB: 0.5
THRESHOLD: 0.5
``` 

- MULA

```
lr=0.1, epoch=100, bucket=2000000, wordNgrams=5, dim=50, loss='ova', minCount=100, maxn=3, minn=3
```

