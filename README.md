# Module-14-Algorithmic-trading
Machine Learning Trading Bot

As a Financial advisor of a top advisory firms, task of this challenge is to enhance the existing trading signals with machine learning algorithms that can adapt to new data.

First step is to import the following libraries 

import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report

next step establish a baseline performance 
plot below is a cumulative return plot of the baseline performance which displays the 'Actual Returns' vs 'Strategy Returns'

![image](https://user-images.githubusercontent.com/91399431/208250299-68fc41db-30bc-447f-9a27-2567ddbe7a18.png)

In this section the baseline model is adjusted with a 24 months window of training data 
plot below is a cumulative return plot of the adjusted baseline performance with 24 months of training data which displays the 'Actual Returns' vs 'Strategy Returns'

![image](https://user-images.githubusercontent.com/91399431/208250541-3ee4a5f2-8775-45b9-8785-413cbc68988d.png)

By increasing the training data from 3 months period to 24 months period, there is an increase in the classification report score for our model.

In this section the baseline model is adjusted with a 24 months window of training data and a SMA short window of 30 days and 90 days long window
plot below is a cumulative return plot which displays the 'Actual Returns' vs 'Strategy Returns'

![image](https://user-images.githubusercontent.com/91399431/208252238-d41d2956-fd0e-4b87-a40a-a9e22562562e.png)

Finally, plot of a new model (Logistic Regression) to evaluate its performance

![image](https://user-images.githubusercontent.com/91399431/208256131-9321f0ae-772a-46ba-ad95-08ac9ec1a8c1.png)
