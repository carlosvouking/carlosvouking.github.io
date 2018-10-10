---
published: true
---


Getting to the end of our trip, I will be building a classifer to predict the survival status of the passengers in the test file. I am facing a binary classification problem(0 or 1 for 'survived' or 'not survived' respectively). 
The model will learn the patterns of data in the training set and apply them when presented data from the test set.

## **(I)- Building predictive models for the titanic trip**
***


```python
# Calling the necessary libraries
import numpy as np
import pandas as pd
import os
```

#### As data need to be fed in the classifier to train it; I obviously need to import them in the notebook.

### **1- Importing the processed data into the notebook.**


```python
# set the paths to the processed data.
processed_data_path = os.path.join(os.path.pardir, 'data', 'processed') # 'data' and 'processed' subfolder initiated when installing the cookiecutter 
write_train_path = os.path.join(processed_data_path, 'train.csv')  # path to the processed trainning file.
write_test_path = os.path.join(processed_data_path, 'test.csv') # path to the processed test file.
```


```python
# reading the data...don't forget default parameters
df_train = pd.read_csv(write_train_path, index_col='PassengerId')
df_test = pd.read_csv(write_test_path, index_col='PassengerId')
```


```python
# A quick glance at the training data
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 1 to 891
    Data columns (total 34 columns):
    Survived                   891 non-null int64
    Age                        891 non-null float64
    Fare                       891 non-null float64
    Size_family                891 non-null int64
    Mum_with_baby              891 non-null int64
    IsFemale                   891 non-null int64
    IsChild                    891 non-null int64
    Embarked_C                 891 non-null int64
    Embarked_Q                 891 non-null int64
    Embarked_S                 891 non-null int64
    Pclass_1                   891 non-null int64
    Pclass_2                   891 non-null int64
    Pclass_3                   891 non-null int64
    deck_A                     891 non-null int64
    deck_B                     891 non-null int64
    deck_C                     891 non-null int64
    deck_D                     891 non-null int64
    deck_Deck NaN              891 non-null int64
    deck_E                     891 non-null int64
    deck_F                     891 non-null int64
    deck_G                     891 non-null int64
    Title_Lady                 891 non-null int64
    Title_Master               891 non-null int64
    Title_Miss                 891 non-null int64
    Title_Mr                   891 non-null int64
    Title_Mrs                  891 non-null int64
    Title_Officer              891 non-null int64
    Title_Sir                  891 non-null int64
    AgeDenomination_Adult      891 non-null int64
    AgeDenomination_Child      891 non-null int64
    Bin_Fare_very_cheap        891 non-null int64
    Bin_Fare_cheap             891 non-null int64
    Bin_Fare_expensive         891 non-null int64
    Bin_Fare_very_expensive    891 non-null int64
    dtypes: float64(2), int64(32)
    memory usage: 243.6 KB



```python
# Then at the test data
df_test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 418 entries, 892 to 1309
    Data columns (total 33 columns):
    Age                        418 non-null float64
    Fare                       418 non-null float64
    Size_family                418 non-null int64
    Mum_with_baby              418 non-null int64
    IsFemale                   418 non-null int64
    IsChild                    418 non-null int64
    Embarked_C                 418 non-null int64
    Embarked_Q                 418 non-null int64
    Embarked_S                 418 non-null int64
    Pclass_1                   418 non-null int64
    Pclass_2                   418 non-null int64
    Pclass_3                   418 non-null int64
    deck_A                     418 non-null int64
    deck_B                     418 non-null int64
    deck_C                     418 non-null int64
    deck_D                     418 non-null int64
    deck_Deck NaN              418 non-null int64
    deck_E                     418 non-null int64
    deck_F                     418 non-null int64
    deck_G                     418 non-null int64
    Title_Lady                 418 non-null int64
    Title_Master               418 non-null int64
    Title_Miss                 418 non-null int64
    Title_Mr                   418 non-null int64
    Title_Mrs                  418 non-null int64
    Title_Officer              418 non-null int64
    Title_Sir                  418 non-null int64
    AgeDenomination_Adult      418 non-null int64
    AgeDenomination_Child      418 non-null int64
    Bin_Fare_very_cheap        418 non-null int64
    Bin_Fare_cheap             418 non-null int64
    Bin_Fare_expensive         418 non-null int64
    Bin_Fare_very_expensive    418 non-null int64
    dtypes: float64(2), int64(31)
    memory usage: 111.0 KB


* **Notes**: 

> The test file (418 entries) is half the training datset (891 rows).

> The test file has 1 column less than the train file. It is actually the 'survived' feature which is to be predicted. Except the 'Survived' output in the training file, I will be using the other 33 features to build the model.

### **2- Getting the data ready to train the model.**

#### One thing to emphasize on is that ML algorithms only accept data in the form of **numeric arrays**. For that simple reason, I need to create one numeric array for the input data and another for the output.
> The input array will be made up with the rows and columns of the training dataset except the 'Survived' column.


```python
# INPUT data: getting all rows and the columns starting from 'Age' onwards....converting those entries into a matrix and each element of the matrix into a float.
X = df_train.loc[:, 'Age' :].as_matrix().astype('float')

# OUTPUT data : taking the 'Survived' column of the train dataset and convert it to a flattened one-dimensional array using the numpy '.ravel()' function.
y = df_train['Survived'].ravel()
```

#### As X and y are numpy arrays, I can get their shapes: 


```python
shape_input, shape_output = X.shape, y.shape
print(f'shape of X: {shape_input} ;; shape of y: {shape_output}')
```

    shape of X: (891, 33) ;; shape of y: (891,)


#### X is a **33-dimensional** array and y a **1-dimensional** array or equaly said a **vector**. That is also why for good practice, I have use lowercase 'y' to denote the vector output array.

#### **Train-test-split:** I now got to split the input array X into 2 parts: this is still the training stage. the split here applies only to the training dataset : df_train. The test dataset df_test is not invloved at all for now.
 > One part for training the classifier and the other for validation purposes (to evaluate the model performance.)
 > This process is called **'Train-Test-Split'**' The test or validation file will be 20% (in general) of the whole training input and will evaluate the model built on the remaining 80%. The **scikit-learn** function **'train_test_split()'** does the split like this: 


```python
# import the 'train_test_split()' function from scikit-learn
from sklearn.model_selection import train_test_split

# splitting...passing to the splitter our arrays X and y...20%≃0.2 for model validation...random_state ensures the same output every time the splitting is invoked.
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
```

#### ** * Checking the split proportions:** 


```python
print(f'input_train_part: {X_train.shape} ;; output_train_part: {y_train.shape}')
```

    input_train_part: (712, 33) ;; output_train_part: (712,)



```python
print(f'input_test_part: {X_val.shape} ;; output_val_part: {y_val.shape}')
```

    input_test_part: (179, 33) ;; output_val_part: (179,)


* **Notes:** 
 * The input array X is subdivised into 2 parts: **X_train** and **X_val**: 
 > X_train : 712 rows and X_val: 179 rows; both with 33 columns
 * The same subdivision is applied to the output vector y: **y_train** and **y_val**
 > y_train:  712 rows and y_val: 179 rows; both are vectors (1 column)

#### ** * Are the train ouput (y_train) and validation output (y_val) vectors balanced enough ?**
 * It is a good practice to check whether they each have approximately the same proportion of positive and negative cases. I am talking here about the 'Survived' column which is the only column in the output vector y. **'Survived'=1** shall be considered as positive while **'Survived'=0** as negative.  
 * Checking their positive cases can be done by taking each of their **np.mean()** like this:


```python
# average survival in the train and validation vectors
positive_cases_training = np.round((np.mean(y_train)*100), 2)
negative_cases_training = 100 - positive_cases_training
positive_cases_validation = np.round((np.mean(y_val)*100), 2)
negative_cases_validation = 100 - positive_cases_validation
print(f'* positive cases training (survived the disaster): {positive_cases_training} %')
print(f'* negative cases training vector (did not survive the disaster): {negative_cases_training} %')
print(f'* positive cases validation vector (survive the disaster): {positive_cases_validation} %')
print(f'* negative cases validation (did not survive the disaster): {negative_cases_training} %')
```

    * positive cases training (survived the disaster): 38.34 %
    * negative cases training vector (did not survive the disaster): 61.66 %
    * positive cases validation vector (survive the disaster): 38.55 %
    * negative cases validation (did not survive the disaster): 61.66 %


#### * **Observations:** 
 * Training and validation both show a similar proportion of positive cases (roughly 39%). Good as it is ideal to have the positive cases equally distributed in the training and validation sets.
 * Only 39% of the data are positive classes, against 61% is made up of negative classes. Thus, in the titanic data, we are facing an imbalance issue between the passengers who survived and the ones who could not survived the titanic disaster. 
 * Though 39 % of positive observations may do for training and validating the titanic survival model, there may be situations with high unbalanced rate between the datasets where some tuning should be necessary to fix the imbalance. So it is always good to check the imbalance situations of the datasets involved in building and evaluating prediction models.

#### ~~ The datasets are now split and ready to feed the model. A good idea is first to build a baseline model, and later tune it to bring it close to a perfect model for predicting titanic passengers' survival status.

### **3- Building and evaluating the baseline model.**###

#### **DummyClassifier()** is another function from **scikit-learn**. This function will help build the baseline model.

#### But there is constraint in using this function because it is available only in scikit-learn from version '0.19.X' onwrads. Let me first check the version of scikit-learn I am using in this notebook.
 


```python
# checking the scikit-learnversion
import sklearn
sklearn.__version__
```




    '0.19.1'



####  As for my notebook, I am running version '0.19.1' of scikit-learn; meaning that the **'DummyClassifier()'** is available. In case of a version prior to '0.19.x', update the scikit-learn. It can be done like this : 
> for native python environment: **....'pip install scikit-learn --upgrade'**

> for conda distribution environment: **....'conda update scikit-learn'**

####  Obviously updating the package means restarting the .ipynb kernel and then re-running all the previous notebook cells for the changes to apply.

####  I can move further with building the baseline model so far as the **'DummyClassifer()'** function is available in the current notebook version of scikit-learn.


```python
# import DummyClassifier
from sklearn.dummy import DummyClassifier
```

####  Creating the model object


```python
# 'most_frequent' because the baseline classifier always outputs the majority class...here the negative cases (61.66%) with 'Survived'=0.
baseline_model = DummyClassifier(strategy='most_frequent', random_state=0)
```

#### Training the model


```python
# train model_dummy by passing input and output data to the fit() function.
baseline_model.fit(X_train, y_train)
```




    DummyClassifier(constant=None, random_state=0, strategy='most_frequent')



#### Evaluate the model perfomance on the test data (Validation) is done as follows:

> - first the model recieves the input data **X_val** and predicts its output.
> - secondly the model compares the predicted output with the actual output **y_val**.


```python
# evaluate 'baseline_model' by passing to the test data to the .score()
baseline_model_score = np.round(baseline_model.score(X_val, y_val), 3)

print(f'Baseline model score: {baseline_model_score} counting for {baseline_model_score*100} %.')
```

    Baseline model score: 0.615 counting for 61.5 %.


 This score of 61.5 % is the baseline model accuracy, point from which can be applied some techniques to improve its performance. It means that, even without using any high level technique or some kind of machine learning algorithm, predicting the value of passengers who did not survived (remember our object model predicts the most frequent case) will be accurate at 61%. 

#### The score is in others terms the accuracy of the model. Other model metrics including the accuracy itself can be computed as follow:


```python
# computing model performance metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
```


```python
# model accuracy score
model_accuracy_score = np.round(accuracy_score(y_val, baseline_model.predict(X_val)),3)
print(f'model accuracy score: {model_accuracy_score*100} %')
```

    model accuracy score: 61.5 %



```python
# evaluating model confusion matrix....count of true positive(tp), true negative(tn), false positive(fp), false negative(fn)
model_confusion_matrix = confusion_matrix(y_val, baseline_model.predict(X_val))
print(f'model confusionmatrix: \n {model_confusion_matrix}')
```

    model confusionmatrix: 
     [[110   0]
     [ 69   0]]



```python
# precision score...ability of a classifier not to label as positive an observation that is negative
model_precision_score = precision_score(y_val, baseline_model.predict(X_val))
print(f'model precision score: {model_precision_score}')
```

    model precision score: 0.0


    /home/cv-dlbox/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)



```python
# recall score : ability of the classifier to find all the positive observations....Survived = 1
model_recall_score = recall_score(y_val, baseline_model.predict(X_val))
print(f'model recall: {model_recall_score}')
```

    model recall: 0.0


#### The above computed performance metrics are so low someone will say. This is still a baseline model, right? we will keep tuning and building upon it and make the performance metrics better.

### **3- First Kaggle Submission: baseline model to [Kaggle](https://www.kaggle.com).**: Now comes the test data df_test into play.
 At this point submitting our model to kaggle will tell how better our prediction model is compared to other submitted models. the df_test is the dataframe for which there is no actual answers. The kaggle platform has the actual answers to compare our submitted predictions with.


```python
# creating 2D matrix of float elements from the df_test
test_X = df_test.as_matrix().astype('float')
```

#### Get the predictions by passing the test_X matrix to the .predict() method.


```python
# getting predictions
predictions = baseline_model.predict(test_X)
```

- I have the predictions. I should now link them to the **'PassengerId'**. The file to be submitted on kaggle should have a **'Survived'** column holding the Survived value predicted for each PassengerId. 
- Also remember that when importing the processed data as df_train and df_test dataframes, 'PassengerId' was set as index column. So in the submission file, intuitively the 'PassengerId' column will hold the df_test.idex values and the 'Survived' column, the predictions.


```python
# dataframe for the submission file with 2 columns: PassengerIds with their respective Survival predictions.
df_submission = pd.DataFrame({'PassengerId': df_test.index, 'Survived': predictions})
```


```python
# top 6 rows of the submission dataframe
df_submission.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>897</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Now, we have the survival predictions attached to the PassengerIds. Writing those to a **'.csv'** file like this:


```python
# create path to save the '.csv' submission file.
path_submission_data = os.path.join(os.path.pardir, 'data', 'external')       # the 'external' folder was created when initializing the cookiecutter environment.
path_submission_file = os.path.join(path_submission_data, 'first_titanic_submission.csv')

# write the data to the submission dataframe to '01_baseline.csv'
df_submission.to_csv(path_submission_file, index=False)     # index = False, we don't want to add more columns in the final output file with indexes.
```

### **4- Good practice: Automate the creation of the submission file.**###
Submission file created. As a good practice, I will create a little function to keep all the steps to create a submission file, to avoid repeating the same steps for next improved models.The function will just be called to generate the required submission file. 


```python
# function to generate a submission file...model and filename parameters are respective to the current model and its corresponding submission file.
def generate_submission_file(model, filename):
    #fisrt converting the test file to a 2D matrix
    test_X = df_test.as_matrix().astype(float)
    #getting the predictions
    predictions = model.predict(test_X)
    #creating the submission dataframe
    df_submission = pd.DataFrame({'PassengerId': df_test.index, 'Survived': predictions})
    #path to the submission file
    path_submission_data = os.path.join(os.path.pardir, 'data', 'external')      
    path_submission_file = os.path.join(path_submission_data, filename)
    # writing the dataframe to the submission file
    df_submission.to_csv(path_submission_file, index=False)
```

Verifying if our function works well by applying it to the previous **'baseline_model'**


```python
# calling the generate_submission_file() on the baseline_model to generate the file: test_submission.csv
generate_submission_file(baseline_model, 'test_submission.csv')
```

The 'test_submission.csv' just generated by invoking the generate_submission_file() function, should be present in the **'../data/processed/''** folder path. A **'tree -L 3'** command shows the structure of the external folder.


```python
%%html
<img src="/images/submission_file_created_.png" />    
```


<img src="//images/submission_file_created_.png" />    


* To submit this file, I logged into Kaggle, joined the **[Titanic competition](https://www.kaggle.com/c/titanic)** and submitted the **'first_titanic_submission.csv'**. my model score is ranked at the **9346th** position as shown below:


```python
%%html
<img src='/images/first_titanic_kaggle_score_.png' />
```


<img src='/images/first_titanic_kaggle_score_.png' />


*  I should point that this relatively bad score should not make us worry, because again this is just a baseline model, from which to start improving. As we get into the world of **Machine Learning**,we will  build a model with a better performance than the baseline model. I also guess that the idea of **'baseline model'** somehow got clearer. 
* The Machine Learning classifier to build now is a **[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)** model. There are many other types, but again **logistic regression** is the most used of the machine learning algorithms for **Classification problems.**
Logistic regression is great at classifying binary categorical ouputs, such as **Survived=1 or Survived=0**.

### **5- Machine learning: Logistic regression Model.**###
* The scikit-learn library will be used for the purpose. It has a bunch of function for machine learning among which the **LogisticRegression()** method that comes into play here.

### A) **First version: Logistic Model**


```python
# import LogisticRegression into the notebook
from sklearn.linear_model import LogisticRegression
```


```python
# creating the logistic regression model object
logistic_model_一番 = LogisticRegression(random_state=0) 
```


```python
# training the logistic model on the training data
logistic_model_一番.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
# Get the model on the test data
logistic_model_一番_score = np.round(logistic_model_一番.score(X_val, y_val), 3)
print(f'First version logistic model SCORE: {logistic_model_一番_score*100} %.')
```

    First version logistic model SCORE: 83.2 %.


YES !! , I remember telling you not to worry about the baseline model score of 61.5%. The performance score has improved to 83.2%
Let's measure other important model metrics such as **accuracy**, **confusion matrix**, **precision**, and **model recall** as for the baseline model to have a better feel of the logistic model performance.


```python
# model accuracy
logistic_model_一番_accuracy = np.round(accuracy_score(y_val, logistic_model_一番.predict(X_val)), 3)
print(f'~ model Accuracy: {logistic_model_一番_accuracy*100} %. Note that, model accuracy and model score just mean the same thing.')

# model precision...not to label as positive a negatice category.
logistic_model_一番_precision = np.round(precision_score(y_val, logistic_model_一番.predict(X_val)), 3)
print(f'~ model Precision: {logistic_model_一番_precision*100} %')

# model recall...all Survived=1
logistic_model_一番_recall = np.round(recall_score(y_val, logistic_model_一番.predict(X_val)), 3)
print(f'~ model recall: {logistic_model_一番_recall*100} %')

# model confusion matrix
logistic_model_一番_confusion_matrix = confusion_matrix(y_val, logistic_model_一番.predict(X_val))
print(f'~ model confusion matrix: \n{logistic_model_一番_confusion_matrix}')
```

    ~ model Accuracy: 83.2 %. Note that, model accuracy and model score just mean the same thing.
    ~ model Precision: 78.3 %
    ~ model recall: 78.3 %
    ~ model confusion matrix: 
    [[95 15]
     [15 54]]


**Notes:** the logistic model performance metrics show significantly better ratios compared to the baseline model. 

the main goal of training a logistic regression model is to find out the optimal weights of its coefficients, also called model parameters. Each weight is optimized internally during the model training and will be computed with each input feature value of every passenger in the test file to make predictions on their survival outcome. The **.coef()** function retrieves the model coefficients:


```python
# model coefficients
model_weights = logistic_model_一番.coef_
print(f'* model weights for each of the input features: \n\n {model_weights}.  \n\n ~~ We can count 33 coefficients for 33 columns or features in the test file.')
```

    * model weights for each of the input features: 
    
     [[-0.02817035  0.00456916 -0.50364001  0.62264919 -0.83649477  0.47542061
       0.4633412   0.44063153  0.12234292  0.93867477  0.45936653 -0.37172566
       0.11949752 -0.18065416 -0.40407468  0.51170583 -0.31445475  1.08821123
       0.3988848  -0.19280013  0.26926215  1.20058334  0.53886011 -1.44542765
       1.05468816 -0.11368693 -0.47796353  0.55089504  0.47542061  0.14621032
       0.22709758  0.26041992  0.39258783]].  
    
     ~~ We can count 33 coefficients for 33 columns or features in the test file.



```python
print(f'model categories: {logistic_model_一番.classes_}.... As 1 for \'Survived\' and 0 for \'Did not Survived\'.')
```

    model categories: [0 1].... As 1 for 'Survived' and 0 for 'Did not Survived'.


#### **1-a/- Second Kaggle Submission: Logistic regression model.** ####
- As our model performance got better, let's attempt a second submission to see if our logistic model scales up in the Kaggle ranking.


```python
# Invoking the 'generate_submission_file()' function on the logistic model to produce the submission file 'second_titanic_submission.csv'
generate_submission_file(logistic_model_一番, 'second titanic_submission.csv')
```


```python
%%html
<img src='/images/second_submission_file_created.png' />
```


<img src='/images/second_submission_file_created.png' />


* Submitting the file to kaggle


```python
%%html
<img src='/images/second_titanic_kaggle_score_.png' />
```


<img src='/images/second_titanic_kaggle_score_.png' />


####  ** ~~ Remarkable improvement; isn't it ?** ####
* We will look even further to improve the this prediction accuracy.

I want us to remember that when creating the logistic regression model object: **'logistic_model_一番 = LogisticRegression(random_state=0)'**,  we did care only about the random_state. there are a lot more parameters when not changed have their default value applied to the Logistic regression object. Below is the overview of the logistic regression parameters.:


```python
%%HTML
<img src='/images/Logreg_hyperparams.png' width='800' />
```


<img src='/imagesLogreg_hyperparams.png' width='800' />


Some of these parameters or hyperparameters, depending on their settings, have some incidence on the accuracy of the logistic regression model. For instance **C (Regularization parameter)** and **penalty** which play a role in Regularization as well; that tackles the model overfitting occurences by lowering its complexity. The goal is to find a balance between **model overfitting** (in this case the model has perfectly learned the patterns in the training data and even started applying the learned patterns on the same training set) and **model underfitting** (case where the model is not able to learn the pattern in the training data). For instance, a high value of **C** will overfit the model while a lower **C** value will lead to model underfitting. So these parameters need some tunning to persist a good balanced model through **Regularization**. Tunning the model parameters is referred to **Hyperparameter optimization**. Kind of asking: what is the best possible combination of the hyperparameters for a well balanced model? 

### B) **LogisticRegression : Hyperparameter Optimization - Understand GridSearch**
> GridSearch is probably be the most common used tweak for hyperparameter optimization. 

> The intuition behind GridSearch is to create multiple combinations of the hyperparameters values, then apply each combination to the model, and finally keep the combination for which the model performance is the best.
   
 #### Then the question of how to evaluate the model performance may arise ? ####
  * #### One way that I used previously to do it was the **Train-Test-Split** method which consisted of splitting the training data into 2 parts: One for training the model (80%) and the other for evaluation purposes (20%). ####
  * #### However for hyperparameter optimization, the model performance evaluation can be done with a slightly different method called as **cross-validation**. ####
  * #### Cross-Validation also splits the training data, but this time into 3 parts or subsets: ####
     - Training subset.
     - Cross-validation subset.
     - Test subset.
       * ** How is that done concretly ?:**
        - 1/- trainning the model with the train subset data.
        - 2/- passing the trained model to the cross-validation subset data to evaluate its performance.
          * **How ? What is the difference with the 'Train-Test-split' process ?** 
            > In fact, the trained model encapsulates multiple possible combinations of its hyperparameters. Each of these combinations makes a different and unique model. 

            > Each of those models is then passed to the cross-validation subset to evaluate its performance. So basically, each combination of the model hyperparameters leads to one score, meaning that, when done with cross-validating, I end up with a number of scores corresponding to the different hyperparameters combinations. The cross-validation subset is used n times (n = number of combinations of hyperparameters) during the training process.
            
            > The best score tells me which of those models is the best. Having the best model at hand, the test subset comes into play.
        - 3/ passing the best model (out of the cross-validation) to the test subset only once to evaluate the final performance for the logistic regression model.
        
**Note:** So wrapping it up, The cross-validation subset is used __n__ times (n = number of combinations of hyperparameters) during the model trainning stage to generate the **best model**, which best model is evaluated once by the test subset.  
~~ But wait!; well before even using the cross-validation subset, I need to have it; You will tell me that I already have it; which is true. But in which proportion ?
 *  #### How to split the training set do to cross-validation ? ####
> The **K-fold cross-valiadtion** very popular in fine-tuning machine learning models, it splits and folds the training data into **k** partitions.
In a **3-fold cross-validation (k=3)**, we split the data into 3 parts **(subset_1, subset_2 and subset_3)** where each part will be leading each of the 3 following setups: 
   * **first setup**: taking **subset_1** to test the model performance and **subset_2** and **subset_3** to train the model itself. Then record the model performance score.
   * **second setup**: testing the model using **subset_2** and train the model with the subsets **subset_1** and **subset_3**. Again record the performance model score.
   * **third setup**: repeating the same, **subset_3** for testing and the other 2 for training the model itself. Then evaluate and record the third model score.
   
 **==>!** The next plausible practice may be to get the average and the standard deviation of the scores recorded (out of the k-fold) to kind of estimating the score difference between the previous k-fold setups.

 This was a very straight-to-the-point and on the surface overview of GridSearch and Cross-validation tweaks on optimizing model hyperparameters. 
 Let's now dive right in applying spme of those regularization tweaks to the logistic regression model we have built up till now : **'logistic_model_一番'**.

### C) **Optimizing the titanic model hyperparameters**
  * scikit-learn library encapsulates a function well suited for hyperparameter optimization operations: **'GridSearchCV'**


```python
# our foundation logistic regression model
logistic_model_一番 =LogisticRegression(random_state=0)
```


```python
# to make use of the GridSearchCV, it needs to be imported
from sklearn.model_selection import GridSearchCV
```


```python
# creating a parameter dictionnary which holds the possible values to assign to each model regularization hyperparameter; namely 'C'and 'penalty'.
regularization_parameters = {    
    'C': [1e-3, 1e-2, 5e-1, 1e-1, 0.1, 50.0, 100.0, 1000.0], 'penalty': ['l1', 'l2']
}
```


```python
# creating a gridSearch object by passing in the titanic model, the regularization parameters' values.
# ... cv for cross-validation splitting startegy: here splitting the titanic training data into 3 parts. setting cv=None uses the default 3-fold cross-validation. 
logistic_model_v2 = GridSearchCV(logistic_model_一番, param_grid=regularization_parameters, cv=3)
```


```python
# train different models with different combinations of C and penalty.
logistic_model_v2.fit(X_train, y_train)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'C': [0.001, 0.01, 0.5, 0.1, 0.1, 50.0, 100.0, 1000.0], 'penalty': ['l1', 'l2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
### best regularization hyperparameters combination
best_model_params = logistic_model_v2.best_params_
print(f'best combination of C and penalty: {logistic_model_v2.best_params_}. \n\t\t ==> It turns out that the best model is trained with C=1.0  and penalty=l1.')
```

    best combination of C and penalty: {'C': 0.5, 'penalty': 'l1'}. 
    		 ==> It turns out that the best model is trained with C=1.0  and penalty=l1.



```python
### best model accuracy score
best_model_score= np.round(logistic_model_v2.best_score_ , 2)
print(f'best model score: {best_model_score}. \n\t\t ==> out of of the hyperparameters combinations, the best combination(c=1 and penalty=l1) leads to 83% accuracy.')
```

    best model score: 0.83. 
    		 ==> out of of the hyperparameters combinations, the best combination(c=1 and penalty=l1) leads to 83% accuracy.


* No much improvment from the first version of our logistic model logistic_model_一番, which was pointing to the same accuracy score. This will be kind of the limit ceiling score for the titanic survival predictive model using the Logistic Regression algorithm. More advanced algorithm might certainly improve its performance.


```python
### apply the test set once for the final score of the model....by passing the test data to the second version of the model.
#...the best model is use here internally to yield the final model performance score.
model_2_score = np.round(logistic_model_v2.score(X_val, y_val), 2)
print(f'model accuracy version 2 - logistic regression: {model_2_score}')
```

    model accuracy version 2 - logistic regression: 0.81


* We have used regularizatiion through hyperparameter tuning to improve the titanic prediction model. There are few more techniques which can help on the same. I will explore few tweaks of **Feature Normalization and Standardization** on increasing the performance of the model.

### D) **Feature Normalization and Standardization **
  * providing features on the same scale can tremendously help machine learning models perform better. 
  * Scaling up the passengers' features may not be of any significant effect on our logistic regression model built up till here. 
  * Usually more sophisticated machine learning algorithms such as Neural nets need some feature normalization before feeding the data to the model.

   ### Intuition behind feature normalization
* Observing some of the passenger features such as Age, Fare and Size_family, show that they are on different scale ranges as below:


```python
f'Age range of the titanic passengers: {df_train.Age.min()} ~ {df_train.Age.max()}' 
```




    'Age range of the titanic passengers: 0.42 ~ 80.0'




```python
f'Fare range of the titanic passengers: {df_train.Fare.min()} ~ {df_train.Fare.max()}' 
```




    'Fare range of the titanic passengers: 0.0 ~ 512.3292'




```python
f'Size of family range of the titanic passengers: {df_train.Size_family.min()} ~ {df_train.Size_family.max()}'
```




    'Size of family range of the titanic passengers: 1 ~ 11'



* The best situation will be to have all the input features on the same scale. Either **'0 ~ 1'** or **'-1 to 1'**, which are the most commonly used scaling ranges. However the common scaling range depends on the features at hand. Let's say for instance, assigning a negative values to the passenger's Age or fare will make no sense.

### Titanic dataset feature normalization


```python
# MinMaxscaler and StandardScaler are the functions responsible respectively for normalization and standardization.
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```


```python
# creating the scaler object
scaler = MinMaxScaler()
# pass the train data to .fit_transform() which has 2 steps: ~ fit the scaler object with the train set ~ transform the train data to generate the scaled output.
X_train_scaled = scaler.fit_transform(X_train)
```

* At this point the model features should have been scaled in the range of ( 0 ~ 1), which is the default range  for **MinMaxScaler()**


```python
# confirm the effect of the scaler job by extracting 1 column of the scaler output and retrieve its minimum and maximum values.
X_train_scaled[:, 0].min(), X_train_scaled[:, 0].max()
```




    (0.0, 1.0)



* the scaler output has all its features with values between 0 and 1 as expected. The test dataset should also be normalized by the same. For the simple reason that training a model with normalized data requires normalized test data for its evaluation.


```python
# scaling the test data
X_test_scaled = scaler.transform(X_val)
```


```python
# confirm the normalization of the test features
X_test_scaled[:, 1].min(), X_test_scaled[:, 1].max()
```




    (0.0, 0.5133418122566507)



   ### Titanic dataset Feature standardization
* The way the input features are distributed can have an impact on the model performance. The **StandardScaler()** function is used to standardize all the features.


```python
# standardization scaler object
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_val)
```

 * Now, the whole titanic dataset is normalized and standardized. Shall I now use hyperparameter optimization and feature standardization to build another version of our logistic regression model to check whether its performance has improved or not.

### E) **Third version: Logistic Regression Model** : After feature standardization.


```python
# model object
logistic_model_v3 = LogisticRegression(random_state=0)

# hyperparameters optimization
regularization_parameters = {
    'C': [1e-3, 1e-2, 5e-1, 1e-1, 0.1, 50.0, 100.0, 1000.0], 'penalty': ['l1', 'l2']
}
model = GridSearchCV(logistic_model_v3, param_grid=regularization_parameters, cv=3)

# train the model on the training data: this time the scaled train input data
model.fit(X_train_scaled, y_train)
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'C': [0.001, 0.01, 0.5, 0.1, 0.1, 50.0, 100.0, 1000.0], 'penalty': ['l1', 'l2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)




```python
# model best parameters combination
model.best_params_
```




    {'C': 0.5, 'penalty': 'l1'}




```python
# model best score
model.best_score_
```




    0.8230337078651685




```python
# evaluating the model with test data.
model_3_score = np.round(model.score(X_test_scaled, y_val), 2)
print(f'model accuracy version 3 - logistic regression: {model_3_score}, for predicting accuracy of {model_3_score*100} %.')
```

    model accuracy version 3 - logistic regression: 0.84, for predicting accuracy of 84.0 %.


* There is a little increase in the logistic regression model performance. For the specific case of logistic regression, hyperpermeter optimization, feature normalization and standardization don't add much value to the model performance. But as good practices, it is advisable to try these tweaks or techniques to possibly improve the model performance.

**==>** I would now say that the final final model for predicting survival status of the titanic passengers is built and ready to be used for further predictions. Let us suppose that the boss and the stackeholders are satisfied with its accuracy of 84 %. They would like to use this model in real time to make the predictions. 

* For this to happen I would have to take the model to a real-world type environment. The steps below show how this can be done.

### F) **Model persistence.**
> consists of taking the trained model and save or persist it to some disk or partition. Once on the disk, it can be reload and used anytime to make predictions. It is kind of powerful process beacuse it avoids training and retraining the model each and every time by running all the cells in the jupyter notebook.

> The persited model can also be used to generate a machine learning API layer above the model.

#### ** Persisting the titanic prediction model
   * model persitence is done using the **'pickle'** library. Another option is the **'dill'** library.


```python
# import pickle library...(available in the anaconda distribution....otherwise do a 'pip install Pickle --upgrade' to install the library).
import pickle

# create the file paths to place the persisted model.
model_file_path = os.path.join(os.pardir, 'models', 'logistic_reg_model.pk1')  #'/models' folder initiated during the cookiecutter data science template creation.

# persist the scaler created previously. it will standardize the new inputs before passing them to the persisted model for new predictions.
scaler_file_path = os.path.join(os.pardir, 'models', 'logistic_reg_scaler.pk1')
```


```python
# OPENING the model and the scaler files in the write mode.....'wb' allows to write in the persisted files in a binary format.
model_file_persisted = open(model_file_path, 'wb')
scaler_file_persisted = open(scaler_file_path, 'wb')
```


```python
# WRITING the model to the persited model file
pickle.dump(model, model_file_persisted)

# WRITING the scaler to the persisted scaler file
pickle.dump(scaler, scaler_file_persisted)
```


```python
# Closing both files
model_file_persisted.close()
scaler_file_persisted.close()
```

* To confirm that the 2 files are effectively persisted locally to the disk.


```python
%%HTML
<img src='/images/persisted_files_.png' />
```


<img src='/images/persisted_files_.png' />


#### ** Loading the persisted files
> this to test whether the saved model and scaler work as expected. by loading back the files in the read mode.


```python
# opening persited model and scaler in the read mode..... 'rb' to read binary aqs the persisted model and scaler were written in a binary format.
model_file_persisted = open(model_file_path, 'rb')
scaler_file_persisted = open(scaler_file_path, 'rb')

# The files are opened....Now load them into memory by reading them with the .load() function from pickle.
model_loaded = pickle.load(model_file_persisted)
scaler_loaded = pickle.load(scaler_file_persisted)

# close the files on the disk after they have been loaded.
model_file_persisted.close()
scaler_file_persisted.close()
```

* At this point I should have the model object back under the name : **model_loaded**. Printing it shows that it is identical to the third version of the logistic regression model built in step E) above.


```python
model_loaded
```




    GridSearchCV(cv=3, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=0, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'C': [0.001, 0.01, 0.5, 0.1, 0.1, 50.0, 100.0, 1000.0], 'penalty': ['l1', 'l2']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)



* Same: The scaler loaded is a StandardScaler object. 


```python
scaler_loaded
```




    StandardScaler(copy=True, with_mean=True, with_std=True)



* Evaluating the loaded model by passing in the titanic passengers in the test file.


```python
# Without feature normalization....that is without using the loaded scaler...test data features not scaled up and normalized
persisted_score_without_normalization = np.round(model_loaded.score(X_val, y_val), 2)
print(f'* Persisted logistic regression model score without feature scaling and normalization: {persisted_score_without_normalization*100} %.')
```

    * Persisted logistic regression model score without feature scaling and normalization: 71.0 %.



```python
# With feature scaling and normalization...thus applying the loaded scaler to transform the test data into a dataframe 'X_test_scaled'
X_test_scaled = scaler_loaded.transform(X_val)

# Feeding the scaled test data into the loaded classifier
persisted_score = np.round(model_loaded.score(X_test_scaled, y_val), 2)
print(f'* Persisted logistic regression model score with normalization and standardization : {persisted_score*100} %.')
```

    * Persisted logistic regression model score with normalization and standardization : 84.0 %.


**!==>!** The loaded classifier accuracy is at 84 % as in step E). The scaler and the model that were persisted and reloaded work just well.

I could normally end this very long titanic disaster trip right here... But what about exposing the final model to be used by any other third party application.? this may really add a nice touch to the whole trip project.    

## **(II)- Exposing our model through an API**
***

The API which will serve as model exposure will be a [**representatational state transfer(REST)**](https://en.wikipedia.org/wiki/Representational_state_transfer) API. It is simply a **'client request - server response'** ping-pong via HTTP protocol. the client or user uses verbs like **POST,** to send some data to the server and **GET,** to get some information from the http server.
> The main goal of this specific API would be to take in new input data (http request from the user), and send back the model predictions (http response) for the data it has been given.

**~~ Intuition behind the API:**
   - The input data on which to make predictions are wrapped in an HTTP request object and sent to the API hosted on the server.
   - The API then graps the input data from the http request object and process the data if being told to do so.
   - The API passes the processed data to the model which makes the predictions.
   - The predictions are then wrapped in an http response object and returned back to the client, happy to see the predictions "magically appearing". 

**~~ How am I going to build that key API ?**
* by bringing into the game one of the famous and powerful libraries built around python, namely **Flask**. That is also why folks like me felt in love with python. It has a plug-in or library for whatever specific task to accomplish.  

   #### ** B- Machine learning API for titanic survival predictions:** using the API boiler template created a while ago.'


```python
ml_api_script_file = os.path.join(os.pardir, 'src', 'models', 'ml_api.py')
```


```python
%%writefile $ml_api_script_file

#### content of the ml_api.py script to create th machine learning API ######

# importing libraries
from flask import Flask, request
import pandas as pd
import numpy as np
import os
import pickle
import json

# setup the flask application
app = Flask(__name__)

#load the model and the scaler from the pickle files...the files I have persisted to the disk.
model_path = os.path.join(os.path.pardir, os.path.pardir, 'models')
model_file_path = os.path.join(model_path, 'logistic_reg_model.pk1')
scaler_file_path = os.path.join(model_path, 'logistic_reg_scaler.pk1')

scaler = pickle.load(open(scaler_file_path, 'rb'))
model = pickle.load(open(model_file_path, 'rb'))

# columns in a specific order expected by the model.
columns = ['Age', 'Fare', 'Size_family', 'Mum_with_baby', 'IsFemale', 'IsChild', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2',
       'Pclass_3', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_Deck NaN', 'deck_E', 'deck_F', 'deck_G', 'Title_Lady', 'Title_Master',
       'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Sir', 'AgeDenomination_Adult', 'AgeDenomination_Child', 'Bin_Fare_very_cheap',
       'Bin_Fare_cheap', 'Bin_Fare_expensive', 'Bin_Fare_very_expensive']

# API function: generate_predictions()....will be executed when the API will be invoked.
@app.route('/api', methods=['POST'])
def generate_predictions():
    # read the json input and convert it to json string
    input_data = json.dumps(request.get_json(force=True))
    
    # create a pandas dataframe using the json string 
    df = pd.read_json(input_data)
    
    # extracting all the passengerIds from the input df
    passenger_ids = df['PassengerId'].ravel()

    # actual survived values....A real world example does not normally make actual values available.
    actuals = df['Survived'].ravel()
    
    # extracting the required columns from the input df, convert them to a matrix X of floating numbers.
    X = df[columns].as_matrix().astype('float') 
    
    # transforming the input matrix X using the scaler loaded from the disk....this results in an input scaled matrix
    X_scaled = scaler.transform(X)
    
    # make predictions on the input scaled dataframe
    predictions = model.predict(X_scaled)
    
    # create the output dataframe as the final API response...it will hold the predictions and the actual values
    df_output = pd.DataFrame({'PassengerId': passenger_ids, 'Predicted': predictions, 'Actuals': actuals})
    
    # return the df_output in a json format
    return df_output.to_json()


# entry point of this script. debug=True help tracking possible errors in the process. However in production, it is advisable to set debug=False
if __name__ == '__main__':
    # host the flask application on port 7007
    app.run(port=7007, debug=True)
```

    Overwriting ../src/models/ml_api.py


* Now heading to the terminal to run the **'ml_api'** script to start the flask app, by typing: **python ml_api.py".**
    See below the flask app effectively running on **port=7007.**


```python
%%HTML
<img src='/amges/flask_ml_api_.png' />
```

* the machine learning API is up and effectively running. let us use it.

**~~ Invoking the machine learning API to make predictions**
   * I am first using the titanic training dataset (files processed and saved in the processed folder) to test if the API is working fine


```python
# reading the processed training data files
processed_data_path = os.path.join(os.path.pardir, 'data', 'processed') 
train_file_path = os.path.join(processed_data_path, 'train.csv')  
df_train = pd.read_csv(train_file_path)
```


```python
# selecting 5 passengers that surived the disaster
few_survived_passengers = df_train[df_train['Survived'] == 1][:5]
few_survived_passengers
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Age</th>
      <th>Fare</th>
      <th>Size_family</th>
      <th>Mum_with_baby</th>
      <th>IsFemale</th>
      <th>IsChild</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>...</th>     
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
          </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>7.9250</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
          </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>35.0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>      
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>27.0</td>
      <td>11.1333</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>     
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>14.0</td>
      <td>30.0708</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>     
    </tr>
  </tbody>
</table>
<p>5 rows × 35 columns</p>
</div>



passing these passengers to the API, it should also return Survived == 1 for these same passengers. Let us test this:


```python
# creating a helper function to make the request to the API
import requests
def make_request_to_api(data):
    # api url
    url_api = 'http://127.0.0.1:7007/api'
    # post request
    r = requests.post(url_api, data)
    # return the response in json format
    return r.json()    
```


```python
# Using the helper function to call the api...pass the 5 passengers to test it
make_request_to_api(few_survived_passengers.to_json())
```




    {'Actuals': {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1},
     'PassengerId': {'0': 2, '1': 3, '2': 4, '3': 9, '4': 10},
     'Predicted': {'0': 1, '1': 1, '2': 1, '3': 1, '4': 1}}



And hence, all the 5 passengers are predicted as 'Survived'==1 by the machine learning API, meaning the API is working as expected.


```python
import json
# Pass the whole training dataframe to the helper function and convert the result to a json string
result = make_request_to_api(df_train.to_json())
# turn the json result into a dataframe
df_result = pd.read_json(json.dumps(result))
# top rows of the results
df_result.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actuals</th>
      <th>PassengerId</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>108</td>
      <td>0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0</td>
      <td>190</td>
      <td>0</td>
    </tr>
    <tr>
      <th>101</th>
      <td>1</td>
      <td>20</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#accuracy level....take the mean of the matching values between the actuals survived and the predicted survived.
np.round(np.mean(df_result.Actuals == df_result.Predicted), 2)
```




    0.84



Remember that we had the same 84 % accuracy for the model persisted previously.

**One challenge** could be the improvment of the machine learning API code. Looking back at it, we are passing the data already processed to the API to generate the predictions. Having the data processing logic itself in the API code, would make the raw data available in the API rather than the processed ones. The machine learning API would then take the raw training data as input and go on with the data processing, pass the processed data to the model to make the ultimate predictions.
