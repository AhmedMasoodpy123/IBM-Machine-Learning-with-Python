# IBM-Machine-Learning-with-Python

In this notebook, IBM challenged participants to demonstrate the effective use of all classification algorithms learnt throughout the course. 
Aim was to develop classifiers with a test accuracy of > 70% based on various metrics i.e. Jaccard Similarity Score and F1 Score as well as examine train and test data in jupyter notebook.

***Installations and libraries required for Classification algorithms, Data visualisation and preprocessing***
```
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
!conda install -c anaconda seaborn -y
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.externals.six import StringIO 
import pydotplus 
import matplotlib.image as mpimg 
from sklearn import tree
```

