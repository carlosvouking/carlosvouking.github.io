---
published: true
---


The data have been downloaded and stored in the raw folder of the project template. The following steps guide us through data analysis.

## **(I)- Data Acquisition: Import Titanic datasets**
***


```python
#import the libraries
import os
import pandas as pd
import numpy as np
```


I set the path to the raw data (as downloaded from Kaggle) like this:
#### raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
#### train_file_path = os.path.join(raw_data_path, 'train.csv')
#### test_file_path = os.path.join(raw_data_path, 'test.csv')




Then I convert the files into dataframes like this:
#### train_df = pd.read_csv(train_file_path, index_col='PassengerId')
#### test_df = pd.read_csv(test_file_path, index_col='PassengerId')




Let's just make sure the files have been turned into dataframes.
type(train_df), type(test_df)


    (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame)



## **(II)- Organizing Titanic project cycle**
***

### **1*- Exploring, analysing and Processing Data (EDA).**
  #- EDA is usually the next step in a Data science project after the raw data are imported.
  - it makes use of several techniques to some interesting insights and patterns about the titanic dataset.

> ### **(A)- Basic structure of the dataframes**


```python
# .info() method gives a brief definition about the dataframes
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 1 to 891
    Data columns (total 11 columns):
    Survived    891 non-null int64
    Pclass      891 non-null int64
    Name        891 non-null object
    Sex         891 non-null object
    Age         714 non-null float64
    SibSp       891 non-null int64
    Parch       891 non-null int64
    Ticket      891 non-null object
    Fare        891 non-null float64
    Cabin       204 non-null object
    Embarked    889 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 83.5+ KB



```python
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 418 entries, 892 to 1309
    Data columns (total 10 columns):
    Pclass      418 non-null int64
    Name        418 non-null object
    Sex         418 non-null object
    Age         332 non-null float64
    SibSp       418 non-null int64
    Parch       418 non-null int64
    Ticket      418 non-null object
    Fare        417 non-null float64
    Cabin       91 non-null object
    Embarked    418 non-null object
    dtypes: float64(2), int64(3), object(5)
    memory usage: 35.9+ KB


**Notes on above**:
   - 891 enries for the training file for 491 entries for the test file.
   - some missing values in columns (Age, Cabin) for the training file and in columns(Age,Fare, Cabin) for the test file. This will be tackled further.
   - There is no 'Survived' column in the test file. As our challenge is to predict the survival for passengers in the test file, let's create that column.    

I am adding a survival column in the test dataframe so that it has the samen structure as the training dataframe


```python
# creating 'Survived' feature with a default value: 555 for the est file
test_df['Survived'] = 555
```

I will then concatenate both training and test file as they now have similar columns structure.
  - axis = 0 does a row-wize concatenation, meaning that values from dataframes will be stacked one over another.
  - axis = 1, does a colum-wise concatenation : concatenated dataframe values will be placed side by side.


```python
# passing both dataframes as a tuple in the pandas' concat function.
df = pd.concat((train_df, test_df), axis=0)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 11 columns):
    Age         1046 non-null float64
    Cabin       295 non-null object
    Embarked    1307 non-null object
    Fare        1308 non-null float64
    Name        1309 non-null object
    Parch       1309 non-null int64
    Pclass      1309 non-null int64
    Sex         1309 non-null object
    SibSp       1309 non-null int64
    Survived    1309 non-null int64
    Ticket      1309 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 122.7+ KB


**Note** :: The full dataframe (training concatenated to test) has 1309 entries or rows. 


```python
# .head() retrieves the top 5 rows of the dataframe.
df.head()
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The top 10 entries of the dataframe are extracted by passing 10 to .head()
df.head(10)
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17599</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>STON/O2. 3101282</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>8.4583</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54.0</td>
      <td>E46</td>
      <td>S</td>
      <td>51.8625</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>21.0750</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>3</td>
      <td>0</td>
      <td>349909</td>
    </tr>
    <tr>
      <th>9</th>
      <td>27.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>11.1333</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>2</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>347742</td>
    </tr>
    <tr>
      <th>10</th>
      <td>14.0</td>
      <td>NaN</td>
      <td>C</td>
      <td>30.0708</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>0</td>
      <td>2</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>237736</td>
    </tr>
  </tbody>
</table>
</div>




```python
# .tail() to get the last 5 rows of the dataframe.
df.tail()
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1305</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Spector, Mr. Woolf</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>A.5. 3236</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>39.0</td>
      <td>C105</td>
      <td>C</td>
      <td>108.9000</td>
      <td>Oliva y Ocana, Dona. Fermina</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>PC 17758</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>38.5</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>SOTON/O.Q. 3101262</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Ware, Mr. Frederick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>359309</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>22.3583</td>
      <td>Peter, Master. Michael J</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>2668</td>
    </tr>
  </tbody>
</table>
</div>



#### * **First 10 passengers in the dataset**


```python
# Here are the names of the first 10 passengers.
df.Name.head(10)
```




    PassengerId
    1                               Braund, Mr. Owen Harris
    2     Cumings, Mrs. John Bradley (Florence Briggs Th...
    3                                Heikkinen, Miss. Laina
    4          Futrelle, Mrs. Jacques Heath (Lily May Peel)
    5                              Allen, Mr. William Henry
    6                                      Moran, Mr. James
    7                               McCarthy, Mr. Timothy J
    8                        Palsson, Master. Gosta Leonard
    9     Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)
    10                  Nasser, Mrs. Nicholas (Adele Achem)
    Name: Name, dtype: object




```python
# Extracting the 'Name','Class' and the 'Fare' of the first 5 passengers like this:
df[['Name', 'Pclass', 'Fare']].head(5)
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
      <th>Name</th>
      <th>Pclass</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Braund, Mr. Owen Harris</td>
      <td>3</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>1</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>3</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>1</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Allen, Mr. William Henry</td>
      <td>3</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>



#### ** * Extracting some rows patterns from the dataframe**
- I may want to select and extract particular rows from the dataframe. This by providing indexes of those rows and columns :: pandas indexing.
- Pandas supports different types of indexing. 

   #### -> **Label Based indexing** : 
   - using **'loc'** for label based indexing.  
   - **usage**: df.loc[ rows indexes, columns indexes ]
     * The passengerId is the index column as set when defining the training and test dataframes.


```python
# select rows with passengerID labels ranging from 4-9 and all the columns; so no specific label index for columns.
df.loc[4:9,]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>113803</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>8.4583</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54.0</td>
      <td>E46</td>
      <td>S</td>
      <td>51.8625</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>21.0750</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>3</td>
      <td>0</td>
      <td>349909</td>
    </tr>
    <tr>
      <th>9</th>
      <td>27.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>11.1333</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>2</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>347742</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extracting rows from passengerId 4 to passengerId 8 for columns from passenger's Age to passenger's class.
df.loc[4:8, 'Age':'Pclass']
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>8.4583</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54.0</td>
      <td>E46</td>
      <td>S</td>
      <td>51.8625</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>21.0750</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Extract 'Survived', 'Fare', 'Embarked' information for passengerID from 4-8 and few columns.
df.loc[4:8, ['Survived', 'Fare', 'Embarked']]
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
      <th>Survived</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>51.8625</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>21.0750</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



* **Positioning Based indexing** : 
   - using **'iloc'** for positional based indexing.  
   - **usage**: df.iloc[ rows indexes, columns indexes ]  
   - Remember: Python uses 0 based indexing.


```python
# select dataframe rows from 4:8 and columns from 5:8 
df.iloc[4:8, 2:8]
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
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Q</td>
      <td>8.4583</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
    </tr>
    <tr>
      <th>7</th>
      <td>S</td>
      <td>51.8625</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
    </tr>
    <tr>
      <th>8</th>
      <td>S</td>
      <td>21.0750</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



* **Filter rows based on certain conditions.**
    - use loc[condition on dataframe, column indexing]    


```python
# Suppose I want to figure out how many first class (Pclass = 1) passengers are in the whole dataset. With a simple bolean expression. I will do it like this:
first_class_passengers = df.loc[df.Pclass == 1, :]
print(f'Number of passengers in the first class: {len(first_class_passengers)}.')
```

    Number of passengers in the first class: 323.



```python
# What about the number of male passengers in the titanic.
male_passengers = df.loc[df.Sex == 'male', :]
print(f'Number of male passengers in the dataset: {len(male_passengers)}.')
```

    Number of male passengers in the dataset: 843.



```python
# And how many male passengers were travelling in the first class ?
male_first_class = df.loc[(df.Sex == 'male') & (df.Pclass == 1), :]
print(f'There were: {len(male_first_class)} travelling in the first class.')
```

    There were: 179 travelling in the first class.


> ### **(B)- Summary statistics for the titanic dataframe.**

#### * **Numerical Statistics**
   - Centrality measures **(mean, median)**
   - Dispersion / spread measures **(range, percentiles, variance, standard deviation)**
   
#### * **Categorical statictics**
   - Total count, Unique count, proportions


```python
#  .describe() will get me statistics for all numeric columns in the dataframe.
df.describe()
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
      <th>Age</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1046.000000</td>
      <td>1308.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.881138</td>
      <td>33.295479</td>
      <td>0.385027</td>
      <td>2.294882</td>
      <td>0.498854</td>
      <td>177.488159</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.413493</td>
      <td>51.758668</td>
      <td>0.865560</td>
      <td>0.837836</td>
      <td>1.041658</td>
      <td>258.670166</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.000000</td>
      <td>7.895800</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.000000</td>
      <td>14.454200</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>39.000000</td>
      <td>31.275000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>555.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>512.329200</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>8.000000</td>
      <td>555.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### **Numerical features - Centrality measures**


```python
# Mean fare
print(f'Mean fare: {df.Fare.mean()}')
# Median fare
print(f'Median fare: {df.Fare.median()}')
print('It is interesting to note that the \'mean fare\' >> \'median fare\'.this may give some infformation on the skewness of the features distribution.')
```

    Mean fare: 33.2954792813456
    Median fare: 14.4542
    It is interesting to note that the 'mean fare' >> 'median fare'.this may give some infformation on the skewness of the features distribution.


#### **Numerical features - Dispersion measures**


```python
# Minimum fare value in the dataset
print(f'* Minimum fare: {df.Fare.min()}')
# Maximum fare value in the dataset
print(f'* Maximum fare: {df.Fare.max()}')
# Fare range in the dataset
print(f'* Fare range: {(df.Fare.max() - df.Fare.min())}')
# 25th percentile of the Fare column
print(f'* 25th percentile: {df.Fare.quantile(.25)}')
# 50th percentile of the fare
print(f'* 50th percentile: {df.Fare.quantile(.50)}')
#75 percentile
print(f'* 75th percentile: {df.Fare.quantile(.75)}')
# Variance Fare
print(f'* Variance fare: {df.Fare.var()}')
# Standard deviation
print(f'* Standard deviation: {df.Fare.std()}')
```

    * Minimum fare: 0.0
    * Maximum fare: 512.3292
    * Fare range: 512.3292
    * 25th percentile: 7.8958
    * 50th percentile: 14.4542
    * 75th percentile: 31.275
    * Variance fare: 2678.959737892894
    * Standard deviation: 51.75866823917414


 * #### Create a **Box-whisker-plot** : distribution of data features using quartiles.


```python
# magic function to allow inline plotting: show the actual plot in the notebook.
%matplotlib inline
# Box-plot n the passenger fare
df.Fare.plot(kind='box', figsize=(10,6), title='Box plot: Fare');
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_44_0.png)      


**Notes on the box plot:** 
    - The green line within the rectangular box represents the median fare (which is 14.4542 as computed above).
    - the left and right borders of the box plot represent respectively the '25th Fare percentile' and the '75 Fare percentile' (respectively 7.8958 and 31.275)

####  **Categorical features - Count and proportions**    


```python
# passing 'include='all' parameter to .describe() retrieve statistics for both numerical and non-numerical columns.
df.describe(include='all')
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1046.000000</td>
      <td>295</td>
      <td>1307</td>
      <td>1308.000000</td>
      <td>1309</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1309</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>186</td>
      <td>3</td>
      <td>NaN</td>
      <td>1307</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>929</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>C23 C25 C27</td>
      <td>S</td>
      <td>NaN</td>
      <td>Connolly, Miss. Kate</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA. 2343</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>6</td>
      <td>914</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>843</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>29.881138</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33.295479</td>
      <td>NaN</td>
      <td>0.385027</td>
      <td>2.294882</td>
      <td>NaN</td>
      <td>0.498854</td>
      <td>177.488159</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>14.413493</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>51.758668</td>
      <td>NaN</td>
      <td>0.865560</td>
      <td>0.837836</td>
      <td>NaN</td>
      <td>1.041658</td>
      <td>258.670166</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.170000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>21.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.895800</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>28.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14.454200</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>39.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>31.275000</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>555.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>512.329200</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>8.000000</td>
      <td>555.000000</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#  .value_counts() gets me the number of females and males in the dataset.
df.Sex.value_counts()
```




    male      843
    female    466
    Name: Sex, dtype: int64




```python
# passing  the attribute normalize='True' to get this time the proportion of males and females instead of their numbers.
df.Sex.value_counts(normalize=True)
```




    male      0.644003
    female    0.355997
    Name: Sex, dtype: float64




```python
# What about checking the number of passengers who survived?  Remeber that we added a 'Survived' feature for the test dataset with a default value of '--' 
df[df.Survived != 555].Survived.value_counts()     # 342 passengers survived whereas 549 did not. 
```




    0    549
    1    342
    Name: Survived, dtype: int64




```python
# It will be obvious to check the number of passengers in each Class category ?
df.Pclass.value_counts()
print(f'the third class has: 709 passengers.')
print(f'the first class has: 323 passengers.')
print(f'the second class has: 277 passengers.')
```

    the third class has: 709 passengers.
    the first class has: 323 passengers.
    the second class has: 277 passengers.



```python
# Deducing the proportion of passengers per class category is therefore done with passing the attribute 'normalize=True' to the value_counts().
df.Pclass.value_counts(normalize=True)
print(f'The passengers in the third class count for 54%.')
print(f'The passengers in the third class count for 25%.')
print(f'The passengers in the third class count for 21%.')
```

    The passengers in the third class count for 54%.
    The passengers in the third class count for 25%.
    The passengers in the third class count for 21%.



```python
# Vsually access the number of passengers per class category
df.Pclass.value_counts().plot(kind='bar',figsize=(8,6), rot=0, title='Passenger per Class category', color='darkslateblue'),;  #';' suppresses the matplotlib line description 
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_53_0.png)



```python

df.Pclass.value_counts(normalize=True).plot(kind='bar', figsize=(8,6), rot=0, color='y', title='Proportin of passengers per Class category.' );
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_54_0.png)


### **(C)- Distributions on the titanic Dataframe** : 
The different types of distributions shall help better understand the history behind the titanic dataset.

  > **1/- HISTOGRAM :: Univariate Distribution (visualize the distribution of only ONE feature at a time) **


```python
# Visualising the age distribution of the passengers who took part to the trip.
df['Age'].plot(kind='hist', title='Age distribution for the passengers on the Titanic.', figsize=(10,7), color='y');
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_56_0.png)


**Few patterns on the Passengers' Age distribution:** 
    - Most of the passengers were between 17 and 32 years old.
    - Close to 70 passengers were kids between 0 and 10 years old.
    - 15 Old passengers between 64 and 80 years old.


```python
# Add more bins to get more indepth infos aon the passengers' Age distribution.
df.Age.plot(kind='hist', title='Age Distribution for the passengers on the Titanic.', figsize=(10,7), color='y', bins=30);

print(f'mean age: {df.Age.mean()}.')
print(f'median age: {df.Age.median()}.')
```

    mean age: 29.881137667304014.
    median age: 28.0.



![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_58_1.png)



```python
# Histogram distribution for passengers' fare
df.Fare.plot(kind='hist', title='Fare distribution for the passengers on the Titanic.', color='c', figsize=(10,7), bins=30);
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_59_0.png)


* I try here to compare skewness of pasenger's Age to passenger's fare


```python
print(f'Age skewness: {df.Age.skew()}.')
print(f'Fare skewness: {df.Fare.skew()}.')
```

    Age skewness: 0.40767455974362266.
    Fare skewness: 4.367709134122922.


**Notes:**
    - Adding more bins to the plot help better appreciate the skewness of the distribution.
    - 'mean age' > 'median age'; so the age distribution is positively skewed.
    - the fare plot is very skewed compared to the age plot.
    
**Few patterns on the Passengers' Age distribution:** 
    - Most of the passengers had their fare between less than a 100 units of fare while only approximately 5 had their fares above the 500 units of fare.


  - **2/- KDE (Kernel Density estimation) :: Univariate Distribution (curved-shaped distribution for ONE feature at a time) **


```python
df.Age.plot(kind='kde', title='Density of Age distribution', color='r', figsize=(12,6));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_64_0.png)


**Density plot on the Passengers' Age distribution:** 
   As seen previously with the histogram,
    - The high density of passengers is between 17 and 32 years old.    
    - the low density corresponds to kids and old passengers.

   - **3/- Scatter plot Bivariate Distribution (visualize the distribution of TWO features at a time) **

**Question:: Is there any interesting pattern between passenger Age and passenger fare ?**


```python
# scatter plot for Age against Fare.
df.plot.scatter(x='Age', y='Fare', title='Scatter plot: Age vs Fare', color='c', figsize=(12,6));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_68_0.png)



```python
# Because we have lots of dots overlaping, we need to ajust the opacity with the help of the 'alpha' attribute.
df.plot.scatter(x='Age', y='Fare', title='Scatter plot: Age vs Fare', color='c', figsize=(12,6), alpha=0.1);
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_69_0.png)


* **Note ::** 
      Amidst seeing that most of the passengers having low fares were mostly young ones between 17 and 37 years old, 
      there is no significant correlation to point out between Age and Fare, 
      because as the Age increases, Fare is not changing very much.
      Checking this correlation was worth a try anyway.

**Possible correlation:: Let's see if the passenger's class has an impact on the passenger's fare ?**


```python
# scatter plotting passenger's class against passenger's fare
df.plot.scatter(x='Pclass', y='Fare', title='Scatter plot : Passenger class vs Passenger Fare !', color='b', figsize=(12,8), alpha=0.15);
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_72_0.png)


**Notes:** 
   - 3 lines representing the categorical possibilities for Passenger class (1 for first class, 2 for second class, and 3 fro third class).
   - first class fares are high as expected, but some of those passengers in the first class with low fares may have booked their trip long time back or they had some discounts.
   - passengers in second and third class have relatively low fares, obviously.

> ### **(D)- Grouping or Aggregations** : 


```python
# Let's find out the median value for Age attribute for male and female passengers in the titanic ship.
df.groupby('Sex').Age.median()
```




    Sex
    female    27.0
    male      28.0
    Name: Age, dtype: float64



**Notes:** 
   - Only 2 groups in the Sex Feature: Male and Female.
   - Notice that median age for both males and females are similar.


```python
# What about the median passenger fare for each Class group ?
df.groupby('Pclass').Fare.median()
```




    Pclass
    1    60.0000
    2    15.0458
    3     8.0500
    Name: Fare, dtype: float64



**Notes:** 
   - 3 groups in the Class Feature: 1, 2 and 3 for first, second and third classes.
   - it is also obvious from the result above that first class passengers spent more in Fare than those in the second anf third class.


```python
# What about the median age for each of the passenger Class ?
df.groupby(['Pclass']).Age.median()
```




    Pclass
    1    39.0
    2    29.0
    3    24.0
    Name: Age, dtype: float64




```python
# In case of extracting the median Age and median Fare for each category in the passenger class group, I do it like this:
df.groupby(['Pclass'])['Age', 'Fare'].median()
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
      <th>Age</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>39.0</td>
      <td>60.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.0</td>
      <td>15.0458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24.0</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Little complicated: In one go, I want ot extract the Fare 'mean' but the Age 'median' of the passengers in each class categroy.
#...I do that by passing the mean fare and median age dictionnary to the agg() function :: aag for aggregate.
df.groupby(['Pclass']).agg({'Fare':'mean', 'Age':'median'})
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
      <th>Fare</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>87.508992</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.179196</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>13.302889</td>
      <td>24.0</td>
    </tr>
  </tbody>
</table>
</div>



#### **' Some extended aggregation with pandas and numpy**
  - The thing here is to create some aggregations separately and pass them to the agg() function on the specified feature group
  


```python
# Create an aggregation with some Fare and Age summary statictics
aggregations_1 = {
    'Age': {
        'Median age': 'median',
        'Minimum age': min,
        'Maximum age': max,
        'Age range': lambda x: max(x)-min(x)   # the lamdba function simply computes the difference between the maximum and the minimum age.
    },
    'Fare': {
        'Mean fare': 'mean',
        'Median fare': 'median',
        'Maximum Fare': max,
        'Minimun Fare': np.min
    }
}
```


```python
# my aggregation is then passed to the .agg() function on the class groups
df.groupby(['Pclass']).agg(aggregations_1)
```

    /home/cv-dlbox/anaconda3/lib/python3.6/site-packages/pandas/core/groupby.py:4291: FutureWarning: using a dict with renaming is deprecated and will be removed in a future version
      return super(DataFrameGroupBy, self).aggregate(arg, *args, **kwargs)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Age</th>
      <th colspan="4" halign="left">Fare</th>
    </tr>
    <tr>
      <th></th>
      <th>Median age</th>
      <th>Minimum age</th>
      <th>Maximum age</th>
      <th>Age range</th>
      <th>Mean fare</th>
      <th>Median fare</th>
      <th>Maximum Fare</th>
      <th>Minimun Fare</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>39.0</td>
      <td>0.92</td>
      <td>80.0</td>
      <td>79.08</td>
      <td>87.508992</td>
      <td>60.0000</td>
      <td>512.3292</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29.0</td>
      <td>0.67</td>
      <td>70.0</td>
      <td>69.33</td>
      <td>21.179196</td>
      <td>15.0458</td>
      <td>73.5000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>24.0</td>
      <td>0.17</td>
      <td>74.0</td>
      <td>73.83</td>
      <td>13.302889</td>
      <td>8.0500</td>
      <td>69.5500</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# The same way I can extract passenger age and fare summary statistics for the spouse siblings in the Titanic passengers.
df.groupby(['SibSp']).agg(aggregations_1)
```

    /home/cv-dlbox/anaconda3/lib/python3.6/site-packages/pandas/core/groupby.py:4291: FutureWarning: using a dict with renaming is deprecated and will be removed in a future version
      return super(DataFrameGroupBy, self).aggregate(arg, *args, **kwargs)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">Age</th>
      <th colspan="4" halign="left">Fare</th>
    </tr>
    <tr>
      <th></th>
      <th>Median age</th>
      <th>Minimum age</th>
      <th>Maximum age</th>
      <th>Age range</th>
      <th>Mean fare</th>
      <th>Median fare</th>
      <th>Maximum Fare</th>
      <th>Minimun Fare</th>
    </tr>
    <tr>
      <th>SibSp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>28.0</td>
      <td>0.33</td>
      <td>80.0</td>
      <td>79.67</td>
      <td>25.785406</td>
      <td>9.5000</td>
      <td>512.3292</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30.0</td>
      <td>0.17</td>
      <td>76.0</td>
      <td>75.83</td>
      <td>48.711300</td>
      <td>26.0000</td>
      <td>263.0000</td>
      <td>6.4375</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.5</td>
      <td>0.75</td>
      <td>59.0</td>
      <td>58.25</td>
      <td>48.940576</td>
      <td>24.1500</td>
      <td>262.3750</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>14.5</td>
      <td>2.00</td>
      <td>33.0</td>
      <td>31.00</td>
      <td>71.332090</td>
      <td>25.4667</td>
      <td>263.0000</td>
      <td>15.8500</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.0</td>
      <td>1.00</td>
      <td>38.0</td>
      <td>37.00</td>
      <td>30.594318</td>
      <td>31.2750</td>
      <td>39.6875</td>
      <td>7.7750</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10.5</td>
      <td>1.00</td>
      <td>16.0</td>
      <td>15.00</td>
      <td>46.900000</td>
      <td>46.9000</td>
      <td>46.9000</td>
      <td>46.9000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>14.5</td>
      <td>14.50</td>
      <td>14.5</td>
      <td>NaN</td>
      <td>69.550000</td>
      <td>69.5500</td>
      <td>69.5500</td>
      <td>69.5500</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Getting median FARE for each of the combinatgion of passenger class and embarked is done like this:
#...I first group the passengers by Class, 
#...then in each class, I group them by Embarcation point. 
#...before applying the median to the fare on each embarcation point for the respective passenger class
df.groupby(['Pclass', 'Embarked']).Fare.median()
```




    Pclass  Embarked
    1       C           76.7292
            Q           90.0000
            S           52.0000
    2       C           15.3146
            Q           12.3500
            S           15.3750
    3       C            7.8958
            Q            7.7500
            S            8.0500
    Name: Fare, dtype: float64




```python
# What about getting median FARE and median AGE for each of the combinations of passenger class and embarked? I will do it like this:
df.groupby(['Pclass', 'Embarked'])['Fare', 'Age'].median()
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
      <th></th>
      <th>Fare</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th>Embarked</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>C</th>
      <td>76.7292</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>90.0000</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>S</th>
      <td>52.0000</td>
      <td>39.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">2</th>
      <th>C</th>
      <td>15.3146</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>12.3500</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>S</th>
      <td>15.3750</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">3</th>
      <th>C</th>
      <td>7.8958</td>
      <td>20.5</td>
    </tr>
    <tr>
      <th>Q</th>
      <td>7.7500</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>S</th>
      <td>8.0500</td>
      <td>24.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Another interesting thing will be to find the mean fare and the median age for male and female at each embarcation point within each passenger class.
df.groupby(['Pclass', 'Embarked', 'Sex']).agg({'Fare':'mean', 'Age':'median'})
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
      <th></th>
      <th></th>
      <th>Fare</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Pclass</th>
      <th>Embarked</th>
      <th>Sex</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">1</th>
      <th rowspan="2" valign="top">C</th>
      <th>female</th>
      <td>118.895949</td>
      <td>38.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>94.622560</td>
      <td>39.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Q</th>
      <th>female</th>
      <td>90.000000</td>
      <td>35.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>90.000000</td>
      <td>44.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">S</th>
      <th>female</th>
      <td>101.069145</td>
      <td>34.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>53.670756</td>
      <td>42.00</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">2</th>
      <th rowspan="2" valign="top">C</th>
      <th>female</th>
      <td>27.003791</td>
      <td>23.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>20.904406</td>
      <td>29.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Q</th>
      <th>female</th>
      <td>12.350000</td>
      <td>30.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>11.489160</td>
      <td>59.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">S</th>
      <th>female</th>
      <td>23.023118</td>
      <td>28.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>20.073322</td>
      <td>29.00</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">3</th>
      <th rowspan="2" valign="top">C</th>
      <th>female</th>
      <td>13.834545</td>
      <td>15.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>9.775901</td>
      <td>24.25</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">Q</th>
      <th>female</th>
      <td>9.791968</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>10.979167</td>
      <td>25.00</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">S</th>
      <th>female</th>
      <td>18.083851</td>
      <td>22.00</td>
    </tr>
    <tr>
      <th>male</th>
      <td>13.145977</td>
      <td>25.00</td>
    </tr>
  </tbody>
</table>
</div>



> ### **(E)- Crosstabs** : 
     - very handy and powerful when dealing with categorical features such as 'PCLASS','SEX'
     - used to initiate cross-tabulations between data features and entries.


```python
# Let us extract passenger class and gender attributes through a crosstab
pd.crosstab(df.Sex, df.Pclass)
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
      <th>Pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>144</td>
      <td>106</td>
      <td>216</td>
    </tr>
    <tr>
      <th>male</th>
      <td>179</td>
      <td>171</td>
      <td>493</td>
    </tr>
  </tbody>
</table>
</div>



**little insight:** 
- In the third class the majority of the passengers are male


```python
# Visualizing the crosstab results give this:
pd.crosstab(df.Sex, df.Pclass).plot(kind='bar', rot=0, figsize=(10, 6));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_92_0.png)


> ### **(F)- Pivot table** : 
  - it is a king of extension of the crosstab()
  - useful for computing numerical feature (like 'Age') for different combinations of 2 or more categorical variables (for instance 'Pclass', 'Sex').


```python
# Average of male and female passengers' age in each Class category.
df.pivot_table(index='Sex', columns='Pclass', values='Age', aggfunc='mean')
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
      <th>Pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>37.037594</td>
      <td>27.499223</td>
      <td>22.185329</td>
    </tr>
    <tr>
      <th>male</th>
      <td>41.029272</td>
      <td>30.815380</td>
      <td>25.962264</td>
    </tr>
  </tbody>
</table>
</div>



**Note:** 
   - On average female passenger's age is lower than male passenger's age.
   - Pivot table in pandas is just one of those methods to achieve the same result.


```python
# For fun, saving the above pivot result as excel file 'pivot.xlsx'.
df.pivot_table(index='Sex', columns='Pclass', values='Age', aggfunc='mean').to_excel('pivot.xlsx', sheet_name='genre_pivot.xlsx', encoding='UTF-8')
```


```python
# A groupby() on SEX and CLASS for median Age leads to the same result.
df.groupby(['Sex', 'Pclass']).Age.mean()
```




    Sex     Pclass
    female  1         37.037594
            2         27.499223
            3         22.185329
    male    1         41.029272
            2         30.815380
            3         25.962264
    Name: Age, dtype: float64




```python
# get the result in a table format with .unstack() function
df.groupby(['Sex', 'Pclass']).Age.mean().unstack()
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
      <th>Pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>37.037594</td>
      <td>27.499223</td>
      <td>22.185329</td>
    </tr>
    <tr>
      <th>male</th>
      <td>41.029272</td>
      <td>30.815380</td>
      <td>25.962264</td>
    </tr>
  </tbody>
</table>
</div>



**' Next steps:** from here I will be applying some DATA MUNGING, FEATURE ENGINEERING and VISUALIZATIONS on our titanic dataset to close the Organizing phase of our project cycle, before getting into building a model to predict passengers who may have survived in the test dataset.

## **2*- Data Munging....Accessing missing values**
  * Here, it is to look in the data for potential issues and correct them.
  * Most data issues are related to data quality such as: 
     - MISSING values in the datasetsin Case of missing values, there is no information for some features in respect to some entries in the dataset. 
     - OUTLIERS (extreme values) 
     - Erroneous or mislabelled values: can be really difficult to reveal.
  So this needs to be adressed and fixed.

> ### **(G)- Missing values** : 
  - The origin could be the lack of those values or even manipulation errors during manual data entry processes.
  - Why worrying about? : missing values lead to inaccurate data analysis , thus affecting predictions' accuracy.
  - Few solutions: **Delete missing value** (not always recommended, could discard useful information) and **imputation**(replacing missing missing values by some acceptable ones).


```python
# .info() function on the dataframe to view missing values.
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 11 columns):
    Age         1046 non-null float64
    Cabin       295 non-null object
    Embarked    1307 non-null object
    Fare        1308 non-null float64
    Name        1309 non-null object
    Parch       1309 non-null int64
    Pclass      1309 non-null int64
    Sex         1309 non-null object
    SibSp       1309 non-null int64
    Survived    1309 non-null int64
    Ticket      1309 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 162.7+ KB


**  Feel of the missing values in the titanic dataset:**
    - Age: 263 missings.
    - Cabin: 1014 missings.
    - Embarked: 2 missing values.
    - Fare: 1 missing value.

### ** * Missings on the "Embarked" column**


```python
# missing are typed Null. extract entries with Embarked as Null
df[df.Embarked.isnull()]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62</th>
      <td>38.0</td>
      <td>B28</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Icard, Miss. Amelie</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>113572</td>
    </tr>
    <tr>
      <th>830</th>
      <td>62.0</td>
      <td>B28</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>113572</td>
    </tr>
  </tbody>
</table>
</div>




```python
# It will be a nice hint to check the number of passengers at diverse embarkment points
df.Embarked.value_counts()
```




    S    914
    C    270
    Q    123
    Name: Embarked, dtype: int64




```python
df.Embarked.value_counts(normalize=True)
```




    S    0.699311
    C    0.206580
    Q    0.094109
    Name: Embarked, dtype: float64



**Trick for a solution:** 
  - As Southampton is the most embakment point (70% of the passengers boarded the ship). So we could use **'S'** for Southampton to fill the missing values for the'Embarked' feature.
  - However, I notice that these 2 passengers with unknown embarkment point have both survived(1) to the catastrophy. It would then be interesting to check which embarkment point has the highest survival ratio. 


```python
# To check the survival ratio per embarcation, i do this:
pd.crosstab(df[df.Survived != 555].Survived, df[df.Survived != 555].Embarked)
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
      <th>Embarked</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75</td>
      <td>47</td>
      <td>427</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93</td>
      <td>30</td>
      <td>217</td>
    </tr>
  </tbody>
</table>
</div>



  - As noted before, Southampton has the higher rate of passengers who survived So we can definetly use **'S'** for Southampton to fill the missing values for the'Embarked' feature.
  - we will be tempted to replace the missing values by 'S' for Southampton like this:


```python
#df.Embarked.fillna('s', inplace=True)   # inplace=True makes changes in the actual dataframe, inplace=False will create a new datagrame.
```

  - However by looking again at the 2 passengers with unknown embarcation point, we note that they belong to first class and their fare was 80 units.
      So if we find some other boarding point with a fare closed to 80 units, they will be a plausible indication for the embarcation point of these 2 passengers.


```python
# i will do this by extracting the median fare of each class for each embarkement point 
df.groupby(['Pclass', 'Embarked']).Fare.median()
```




    Pclass  Embarked
    1       C           76.7292
            Q           90.0000
            S           52.0000
    2       C           15.3146
            Q           12.3500
            S           15.3750
    3       C            7.8958
            Q            7.7500
            S            8.0500
    Name: Fare, dtype: float64



**Note:**
   - you will notice that i mostly use 'median' for numerical features instead of 'mean'. this is to avoid the effect of extreme values on estimating averages.
   - So we have median Fare values for each passenger class and embarkment combination.
   - Then, among passengers, those in the first class with a Fare closer to 80 units (76.72 ~= 80 than the 90 units for 'Q') mostly embarked at point **'C' for Cherbourg**. 
   - Thus, I can conclude it is highly possible that the 2 passengers with missing 'Embarked' feature value, boarded the ship at Cherbourg. So I will input **'C'** like this:


```python
# fill missing values in Emabarked feature with 'C'.
df.Embarked.fillna('C', inplace=True)
```


```python
# Check if the Embarked missing values have been filled
df[df.Embarked.isnull()]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
#check the .info() again to read the other missing values and fix them
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 11 columns):
    Age         1046 non-null float64
    Cabin       295 non-null object
    Embarked    1309 non-null object
    Fare        1308 non-null float64
    Name        1309 non-null object
    Parch       1309 non-null int64
    Pclass      1309 non-null int64
    Sex         1309 non-null object
    SibSp       1309 non-null int64
    Survived    1309 non-null int64
    Ticket      1309 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 162.7+ KB


### ** * Missings on the "Fare" column**


```python
# null values in the Fare column
df[df.Fare.isnull()]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1044</th>
      <td>60.5</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
      <td>Storey, Mr. Thomas</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>3701</td>
    </tr>
  </tbody>
</table>
</div>



**Hint:** 
   - Combining Embarkment point of Southampton with the passenger class could help guess the fare for this passenger.


```python
# Filter out passenger in class '3' with embarkement point as 'S' and get their median fare
median_fare = df.loc[(df.Pclass == 3) & (df.Embarked == 'S'), 'Fare'].median()
print(f'median fare for passengers in the third class who boarded the ship at Southampton: {median_fare}.')
```

    median fare for passengers in the third class who boarded the ship at Southampton: 8.05.



```python
# replace the missing value with the median_fare
df.Fare.fillna(median_fare, inplace=True)
```


```python
# Missing value filled up?
df[df.Fare.isnull()]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Other missing values ?
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 11 columns):
    Age         1046 non-null float64
    Cabin       295 non-null object
    Embarked    1309 non-null object
    Fare        1309 non-null float64
    Name        1309 non-null object
    Parch       1309 non-null int64
    Pclass      1309 non-null int64
    Sex         1309 non-null object
    SibSp       1309 non-null int64
    Survived    1309 non-null int64
    Ticket      1309 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 162.7+ KB


### ** * Missings on the "Age" column**


```python
df[df.Age.isnull()]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>8.4583</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>13.0000</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>244373</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2250</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>2649</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2250</td>
      <td>Emir, Mr. Farred Chehab</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>2631</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.8792</td>
      <td>O'Dwyer, Miss. Ellen "Nellie"</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>330959</td>
    </tr>
    <tr>
      <th>30</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.8958</td>
      <td>Todoroff, Mr. Lalio</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>349216</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NaN</td>
      <td>B78</td>
      <td>C</td>
      <td>146.5208</td>
      <td>Spencer, Mrs. William Augustus (Marie Eugenie)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17569</td>
    </tr>
    <tr>
      <th>33</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>Glynn, Miss. Mary Agatha</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>335677</td>
    </tr>
    <tr>
      <th>37</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2292</td>
      <td>Mamee, Mr. Hanna</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>2677</td>
    </tr>
    <tr>
      <th>43</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.8958</td>
      <td>Kraeff, Mr. Theodor</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>349253</td>
    </tr>
    <tr>
      <th>46</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Rogers, Mr. William John</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>S.C./A.4. 23567</td>
    </tr>
    <tr>
      <th>47</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>15.5000</td>
      <td>Lennon, Mr. Denis</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>370371</td>
    </tr>
    <tr>
      <th>48</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>O'Driscoll, Miss. Bridget</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>14311</td>
    </tr>
    <tr>
      <th>49</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>21.6792</td>
      <td>Samaan, Mr. Youssef</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2</td>
      <td>0</td>
      <td>2662</td>
    </tr>
    <tr>
      <th>56</th>
      <td>NaN</td>
      <td>C52</td>
      <td>S</td>
      <td>35.5000</td>
      <td>Woolner, Mr. Hugh</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>19947</td>
    </tr>
    <tr>
      <th>65</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>27.7208</td>
      <td>Stewart, Mr. Albert A</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17605</td>
    </tr>
    <tr>
      <th>66</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>15.2458</td>
      <td>Moubarek, Master. Gerios</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>1</td>
      <td>2661</td>
    </tr>
    <tr>
      <th>77</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.8958</td>
      <td>Staneff, Mr. Ivan</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>349208</td>
    </tr>
    <tr>
      <th>78</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Moutal, Mr. Rahamin Haim</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>374746</td>
    </tr>
    <tr>
      <th>83</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7875</td>
      <td>McDermott, Miss. Brigdet Delia</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>330932</td>
    </tr>
    <tr>
      <th>88</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Slocovski, Mr. Selman Francis</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392086</td>
    </tr>
    <tr>
      <th>96</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Shorney, Mr. Charles Joseph</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>374910</td>
    </tr>
    <tr>
      <th>102</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.8958</td>
      <td>Petroff, Mr. Pastcho ("Pentcho")</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>349215</td>
    </tr>
    <tr>
      <th>108</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.7750</td>
      <td>Moss, Mr. Albert Johan</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>312991</td>
    </tr>
    <tr>
      <th>110</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>24.1500</td>
      <td>Moran, Miss. Bertha</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>371110</td>
    </tr>
    <tr>
      <th>122</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Moore, Mr. Leonard Charles</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>A4. 54510</td>
    </tr>
    <tr>
      <th>127</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>McMahon, Mr. Martin</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>370372</td>
    </tr>
    <tr>
      <th>129</th>
      <td>NaN</td>
      <td>F E69</td>
      <td>C</td>
      <td>22.3583</td>
      <td>Peter, Miss. Anna</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
    </tr>
    <tr>
      <th>141</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>15.2458</td>
      <td>Boulos, Mrs. Joseph (Sultana)</td>
      <td>2</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>0</td>
      <td>2678</td>
    </tr>
    <tr>
      <th>155</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.3125</td>
      <td>Olsen, Mr. Ole Martin</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>Fa 265302</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1160</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Howard, Miss. May Elizabeth</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>A. 2. 39186</td>
    </tr>
    <tr>
      <th>1163</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>Fox, Mr. Patrick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>368573</td>
    </tr>
    <tr>
      <th>1165</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>15.5000</td>
      <td>Lennon, Miss. Mary</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>1</td>
      <td>555</td>
      <td>370371</td>
    </tr>
    <tr>
      <th>1166</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2250</td>
      <td>Saade, Mr. Jean Nassr</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>2676</td>
    </tr>
    <tr>
      <th>1174</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>Fleming, Miss. Honora</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>364859</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Franklin, Mr. Charles (Charles Fardon)</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>SOTON/O.Q. 3101314</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>NaN</td>
      <td>F E46</td>
      <td>C</td>
      <td>7.2292</td>
      <td>Mardirosian, Mr. Sarkis</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>2655</td>
    </tr>
    <tr>
      <th>1181</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Ford, Mr. Arthur</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>A/5 1478</td>
    </tr>
    <tr>
      <th>1182</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>39.6000</td>
      <td>Rheims, Mr. George Alexander Lucien</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>PC 17607</td>
    </tr>
    <tr>
      <th>1184</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2292</td>
      <td>Nasr, Mr. Mustafa</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>2652</td>
    </tr>
    <tr>
      <th>1189</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>21.6792</td>
      <td>Samaan, Mr. Hanna</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2</td>
      <td>555</td>
      <td>2662</td>
    </tr>
    <tr>
      <th>1193</th>
      <td>NaN</td>
      <td>D</td>
      <td>C</td>
      <td>15.0458</td>
      <td>Malachard, Mr. Noel</td>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>237735</td>
    </tr>
    <tr>
      <th>1196</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>McCarthy, Miss. Catherine Katie""</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>383123</td>
    </tr>
    <tr>
      <th>1204</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.5750</td>
      <td>Sadowitz, Mr. Harry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>LP 1588</td>
    </tr>
    <tr>
      <th>1224</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2250</td>
      <td>Thomas, Mr. Tannous</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>2684</td>
    </tr>
    <tr>
      <th>1231</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2292</td>
      <td>Betros, Master. Seman</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>2622</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.5500</td>
      <td>Sage, Mr. John George</td>
      <td>9</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>CA. 2343</td>
    </tr>
    <tr>
      <th>1236</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>14.5000</td>
      <td>van Billiard, Master. James William</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>A/5. 851</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.8792</td>
      <td>Lockyer, Mr. Edward</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>1222</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>O'Keefe, Mr. Patrick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>368402</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.5500</td>
      <td>Sage, Mrs. John (Annie Bullen)</td>
      <td>9</td>
      <td>3</td>
      <td>female</td>
      <td>1</td>
      <td>555</td>
      <td>CA. 2343</td>
    </tr>
    <tr>
      <th>1258</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>14.4583</td>
      <td>Caram, Mr. Joseph</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>2689</td>
    </tr>
    <tr>
      <th>1272</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>O'Connor, Mr. Patrick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>366713</td>
    </tr>
    <tr>
      <th>1274</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>14.5000</td>
      <td>Risien, Mrs. Samuel (Emma)</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>364498</td>
    </tr>
    <tr>
      <th>1276</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>12.8750</td>
      <td>Wheeler, Mr. Edwin Frederick""</td>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>SC/PARIS 2159</td>
    </tr>
    <tr>
      <th>1300</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7208</td>
      <td>Riordan, Miss. Johanna Hannah""</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>334915</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>Naughton, Miss. Hannah</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>365237</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Spector, Mr. Woolf</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>A.5. 3236</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Ware, Mr. Frederick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>359309</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>22.3583</td>
      <td>Peter, Master. Michael J</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>2668</td>
    </tr>
  </tbody>
</table>
<p>263 rows × 11 columns</p>
</div>



==> there are 263 entries in the dataframe with missing age values. This is a too large to show in the notebook and can cause a clutter. pandas provide a nice trick to limit the number of displayed in the notebook. It works like this:


```python
# Setting the maximumrows displayed in notebook to 15.
pd.options.display.max_rows = 15
```


```python
df[df.Age.isnull()]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>8.4583</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>13.0000</td>
      <td>Williams, Mr. Charles Eugene</td>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>244373</td>
    </tr>
    <tr>
      <th>20</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2250</td>
      <td>Masselmani, Mrs. Fatima</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>2649</td>
    </tr>
    <tr>
      <th>27</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>7.2250</td>
      <td>Emir, Mr. Farred Chehab</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>2631</td>
    </tr>
    <tr>
      <th>29</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.8792</td>
      <td>O'Dwyer, Miss. Ellen "Nellie"</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>330959</td>
    </tr>
    <tr>
      <th>30</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.8958</td>
      <td>Todoroff, Mr. Lalio</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>349216</td>
    </tr>
    <tr>
      <th>32</th>
      <td>NaN</td>
      <td>B78</td>
      <td>C</td>
      <td>146.5208</td>
      <td>Spencer, Mrs. William Augustus (Marie Eugenie)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17569</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1274</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>14.5000</td>
      <td>Risien, Mrs. Samuel (Emma)</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>364498</td>
    </tr>
    <tr>
      <th>1276</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>12.8750</td>
      <td>Wheeler, Mr. Edwin Frederick""</td>
      <td>0</td>
      <td>2</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>SC/PARIS 2159</td>
    </tr>
    <tr>
      <th>1300</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7208</td>
      <td>Riordan, Miss. Johanna Hannah""</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>334915</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>Naughton, Miss. Hannah</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>365237</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Spector, Mr. Woolf</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>A.5. 3236</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Ware, Mr. Frederick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>359309</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>C</td>
      <td>22.3583</td>
      <td>Peter, Master. Michael J</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>2668</td>
    </tr>
  </tbody>
</table>
<p>263 rows × 11 columns</p>
</div>



==> THEN, Looking back to the missing values issue in the 'Age' attribute, what can I do about fixing it ?

**" 1rst hint for fixing Age missing values:: (mean of Age)**
>- Intuitively replacing all 'Age' missing values by the mean of the Age feature. It is straightforward and some of us will be happy with that.
    - A handy help will be first to look back on how the passengers' age are distributed...I will easily do it like this:   


```python
df.Age.plot(kind='hist', title='Titanic Passengers\' age distribution', figsize=(10, 7), bins=20);
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_131_0.png)


**Little Note::** The passengers' age is distributed from 0 to 80 years old. Most of the passengers as we have already seen between 17 and 35 years old.

==> In the meantime, the mean Age of the passengers on the ship is around: 


```python
mean_age = df.Age.mean()
print(f'mean age: {mean_age}')
```

    mean age: 29.881137667304014


==> So, the mean age falls around 30 years old. I can then intuitively replace all the missing values in the Age attribute with 30 like this:


```python
#df.Age.fillna('mean_age, inplace=True)
```

==> But a lookup at the age distribution reveals few extreme values around 70 and 80 years old which can easily affect the mean. So I guess it is better to use a different approach for filling up th missing values.

**" 2nd hint for fixing Age missing values:: (median age per Gender)** 
> Can the 'Sex' feature help to replace the missing values ?
   - Males and females passengers may have different age distribution. In that case why not using the median age of males and females to replace the missing values.? i will proceed like so to retrieve the median age for male and female passengers:       


```python
# median age for male and female passengers on the titanic
df.groupby(['Sex']).Age.median()
```




    Sex
    female    27.0
    male      28.0
    Name: Age, dtype: float64



**!! Surprise:** Good or bad, the median age for male and female passengers are almost similar.

==> Simultaneously, does comparing distributions of male and female passengers' age reveals some intersting patterns. ?
> remember to extract age cells with only non-null values by doing df.Age.notnull() while boxplotting (don't know if this exists, I will check later) the age distribution of male and female passengers.
> 


```python
#Box plot for passengers age distribution
df[df.Age.notnull()].boxplot('Age', 'Sex', figsize=(9,10));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_141_0.png)


**Notes::** I noticed a similar age distribution for male and female passengers. Thus, gender is not a good hint to determine which age to use for filling up the missing values.
> I could have done it like this if otherwise:


```python
# creating the median age like this:   
# sex_median_age = df.groupby('Sex').Age.transform('median')  # return entries with median age in respect with male or female passenger. 
# and replacing the missings like this: df.Age.fillna(sex_median_age, inplace=True)
```

==> I keep the code commented for this because this hint is not the best for fixing our 'Age' missing values. Try exploring a better approach.

**" 3rd hint for fixing Age missing values:: (Median Age per passenger class, Pclass)**        
> Can the mean age on passenger class categories help ?
   - The goal is to check the age distributions between first, second and third class passengers. if their distributions are different, the age median on passengers' class could be a good option to fix the missing age values.


```python
# extract the median agefor each passenger class like this:
df.groupby('Pclass').Age.median()
```




    Pclass
    1    39.0
    2    29.0
    3    24.0
    Name: Age, dtype: float64




```python
# What about the distribution? I retrieve the boxplot like I did with the Genger.
df[df.Age.notnull()].boxplot('Age', 'Pclass', figsize=(9,10));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_146_0.png)


**Notes::** I noticed kind of different age distributions for the passengers class. So this may sound like a good option for replacing the missing values in the age column.
> It can be done like this: same as for the median age for male and female passengers,


```python
# pclass_median_age = df.groupby('Pclass').Age.transform('median')
# df.filna(pclass_median_age, inplace=True)
```

==> The above fixing is not bad. I somehow keep the code commented because, _'trust me'_ there can still be a better strategy for fixing the missing age values.
This leads me to a fourth hint using some possible hidden feature in the data.

**" 4th hint for fixing Age missing values:: (Median Age per Name's title)**         
> **_What ??_** :: Well yes, The titles (such as 'Mr.', 'Mrs.', 'Miss.', 'Major', 'lady'...) contained in the passenger's name can be a hint about determinig how old is that person ?
   - for instance
   - It is not simple though. I need to explore the name column to access the titles hidden in the passengers' names..


```python
# reading passengers' names
df.Name
```




    PassengerId
    1                                 Braund, Mr. Owen Harris
    2       Cumings, Mrs. John Bradley (Florence Briggs Th...
    3                                  Heikkinen, Miss. Laina
    4            Futrelle, Mrs. Jacques Heath (Lily May Peel)
    5                                Allen, Mr. William Henry
    6                                        Moran, Mr. James
    7                                 McCarthy, Mr. Timothy J
                                  ...                        
    1303      Minahan, Mrs. William Edward (Lillian E Thorpe)
    1304                       Henriksson, Miss. Jenny Lovisa
    1305                                   Spector, Mr. Woolf
    1306                         Oliva y Ocana, Dona. Fermina
    1307                         Saether, Mr. Simon Sivertsen
    1308                                  Ware, Mr. Frederick
    1309                             Peter, Master. Michael J
    Name: Name, Length: 1309, dtype: object



> And again remeber that we have restricted the number of rows to be displayed in the notebook to 15.

==> No need to have sharp eyes to notice the pattern in the passengers' names.
   - the names are organized like this:  **"family name" - "," - "title" - "first name"**.   It was not obvious to see this. right ??
   - the present task is to extract the titles from the passengers' names. I will kindly assign this task to a little function whihc will extract those tites for me. The function which I call ***'Extract_Title_from_names()*** will operate as follows:
   
> **first**: the whole passenger name is split by the comma ',' sign. 
     - On the left side of the ',' there is the 'family name' (which is of no interest for me). 
     - However on the right side of the comma ',' there is my lovely 'title' combined to the first name of the passenger. These last two also need to be separated.
      
> **second**: the 'title' gets separated from the 'first name' by the dot '.' sign.
     - the left side of the '.' sign is the 'title' and the right side is the 'first name' (not interesting for me).
     - 'title' can then be isolated as per my requirement.
     
> **third**: Clean the titles.
     - it remains to remove all the white space in the isolated title with the .split() function and convert it to lowercase with .lower(), in case it was in uppercase


```python
# accessing and extracting passenger names'title
def Extract_Title_from_names(name):
    title_and_first_name = name.split(',')[1]      #first.  index [1] means to keep the second part after splitting by the comma ','.
    title = title_and_first_name.split('.')[0]     #second.  index [0] : keeping the first part after splitting by the dot '.'
    title = title.strip().lower()                  #third.     clean the white space with .strip()           
    return title
```

==> No that I have a function which can extract the titles from the passengers' names, I can't wait to get get the titles from the passengers' names.
> How to do that :?
   - using the pandas function .map() on the name column. with the help of the useful lambda() mechanism like so:


```python
df['Name'].map(lambda x : Extract_Title_from_names(x))  #equivalent to df['Name'].map(Extract_Title_from_names); x in lambda() represents the 'name' feature.
```




    PassengerId
    1           mr
    2          mrs
    3         miss
    4          mrs
    5           mr
    6           mr
    7           mr
             ...  
    1303       mrs
    1304      miss
    1305        mr
    1306      dona
    1307        mr
    1308        mr
    1309    master
    Name: Name, Length: 1309, dtype: object



**ET VOILA !** I have all the titles extracted from the passengers' names and turned into lowercase.

==> There is a redundancy in the titles' list. It is handy to get the unique titles
> How:? - again with another pandas function **.unique()** applied on the output of the .map() function used previously. It happens like this:


```python
unique_titles = df['Name'].map(lambda x : Extract_Title_from_names(x)).unique()
print(f'Unique titles: \n{unique_titles}')
```

    Unique titles: 
    ['mr' 'mrs' 'miss' 'master' 'don' 'rev' 'dr' 'mme' 'ms' 'major' 'lady'
     'sir' 'mlle' 'col' 'capt' 'the countess' 'jonkheer' 'dona']


==> The list of possible titles is quite long. A good idea is to clump similar titles together to reduce the list.
> How:? 
    - by bringing a little modification to the Extract_Title_from_names() function,
    - then creating few title groups,
    - finally by mapping each the unique titles to the title groups created. (such as putting titles like 'major', 'dr', 'col', 'captain' in the 'Officer' 
    title group that i have created among others. A python dictionnary that i call 'titles_to_group' will be holding the mappings between the tiles and the title groups.
    - the whole thing is done like this:


```python
# Modifying the Extract_Title_from_names() function
def Extract_Title_from_names(name):        
    # groups of titles : 'Mr', 'Mrs', 'Miss', 'Master', 'Sir', 'Sir', 'Officer', 'Lady'
    # assingning titles to the groups of titles
    titles_to_group = {
        'mr' : 'Mr',
        'mrs' : 'Mrs',
        'miss' : 'Miss',
        'master' : 'Master',
        'don' : 'Sir',
        'rev' : 'Sir',
        'dr' : 'Officer',
        'mme' : 'Mrs',
        'ms' : 'Mrs',
        'major' : 'Officer',
        'lady' : 'Lady',
        'sir' : 'Sir',
        'mlle' : 'Miss',
        'col' : 'Officer',
        'capt' : 'Officer',
        'the countess' : 'Lady',
        'jonkheer' : 'Sir',
        'dona' : 'Lady'        
    }   
    
    # extracting titles from names
    first_name = name.split(',')[1]
    title = first_name.split('.')[0]
    title = title.strip().lower()
    # returning the customized group of titles as titles.
    return titles_to_group[title]
```


```python
# extracting the customized titles as did previously with the original titles.
df['Name'].map(lambda x : Extract_Title_from_names(x))
```




    PassengerId
    1           Mr
    2          Mrs
    3         Miss
    4          Mrs
    5           Mr
    6           Mr
    7           Mr
             ...  
    1303       Mrs
    1304      Miss
    1305        Mr
    1306      Lady
    1307        Mr
    1308        Mr
    1309    Master
    Name: Name, Length: 1309, dtype: object



==> I now have my new titles list shrinked to a short group list. It is a good idea to have those visible in the dataframe.
> How:? 
    - by creating and adding a new column to the dataframe.
    - the new column called 'Title' will hold our customized titles :: I usually prefer this handy method to create new additional feature in pandas dataframe.
    Can be done like this:


```python
# Creating a new feature 'Title' to the dataframe to hold our customized titles
df['Title'] = df['Name'].map(lambda x : Extract_Title_from_names(x))
```

==> Simultaneously, I make use of the %%HTML [magical]("https://www.ipython.readthedocs.io/en/stable/interactive/magics.html") function to display below an image highlighting the title feature newly added to the dataframe. 


```python
%%HTML
<img src="/images/hightlighted_titles_extracted.png"></img>
```


<img src="/images/hightlighted_titles_extracted.png"></img>


==> And yes, I have titles' names exctracted from passengers' names, customized and added as a new feature to the titanic dataframe...very very handy !
> As previously, how is the age variation distributed among the passengers names'titles.?


```python
# Age distribution for for titles in passengers' names.
df[df.Name.notnull()].boxplot('Age', 'Title', figsize=(9,8));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_167_0.png)


**Notes on above:** 

> It is quite interesting to see the diversity in age distribution per passenger title. Few useful insights could be drawn here:
   - As expected the median age of **''Master'(title group for kids) around 2 years)** is the lowest.
   - Also passengers with the title **'Miss'** are slightly younger than those with the **'Mrs'** and **Lady** titles which somehow follws a certain logic.
   
==> Passengers' titles would be a good indicator to help assigning age values to those passengers with missing values.
   - After all this delay, the best option is found. Replacing the age missing values is done like this:


```python
# return entries with median age in respect with passenegr's title.
median_age_for_title = df.groupby('Title').Age.transform('median') 
# then finally replacing the missng age values like so:
df.Age.fillna(median_age_for_title, inplace=True)
```


```python
# Checking the passengers' age 
df[df.Age.isnull()]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



==> As expected , no entry corresponding to nullable type (missing values) is returned. This shows that the missing values have been replaced. .info() tells more.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 12 columns):
    Age         1309 non-null float64
    Cabin       295 non-null object
    Embarked    1309 non-null object
    Fare        1309 non-null float64
    Name        1309 non-null object
    Parch       1309 non-null int64
    Pclass      1309 non-null int64
    Sex         1309 non-null object
    SibSp       1309 non-null int64
    Survived    1309 non-null int64
    Ticket      1309 non-null object
    Title       1309 non-null object
    dtypes: float64(2), int64(4), object(6)
    memory usage: 172.9+ KB


==> What remains to be fixed is the missing values for the Cabin column. However, I will not directly use it to build the prediction model for the passenger's survival status. Remember the 'Survived' column we added to the test file which a deafult value of '--'. 
> For now I am going to tackle another issue in the dataset : **outliers** or **extreme values**. Jump right in with me !

> ### **(H)- Outliers or Extreme values in the Titanic dataset** : 
  - they are significantly different from the normal state or appearance of data.
  - extreme values in a dataset can happen during data entry processes
  - data analysis results can get biased and model accuracy and output impacted by some outliers.

#### * **Is there any outlier in the titanic dataset ?**

#### **>>> Outliers in the 'Age' column.**


```python
# Age distribution plot
df.Age.plot(kind='hist', title='histogram for Age', bins=20, color='g', figsize=(10,6));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_177_0.png)


* While most passengers are between 17 and 35 years old, there are few passengers older than 70 years. Let me try to explore those entries for passengers older than 70.


```python
# extract  all columns for rows with age greater than 70 .
df.loc[df.Age > 70, : ]  
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>71.0</td>
      <td>A5</td>
      <td>C</td>
      <td>34.6542</td>
      <td>Goldschmidt, Mr. George B</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17754</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>117</th>
      <td>70.5</td>
      <td>NaN</td>
      <td>Q</td>
      <td>7.7500</td>
      <td>Connors, Mr. Patrick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>370369</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>494</th>
      <td>71.0</td>
      <td>NaN</td>
      <td>C</td>
      <td>49.5042</td>
      <td>Artagaveytia, Mr. Ramon</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17609</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>631</th>
      <td>80.0</td>
      <td>A23</td>
      <td>S</td>
      <td>30.0000</td>
      <td>Barkworth, Mr. Algernon Henry Wilson</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>27042</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>852</th>
      <td>74.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.7750</td>
      <td>Svensson, Mr. Johan</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>347060</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>988</th>
      <td>76.0</td>
      <td>C46</td>
      <td>S</td>
      <td>78.8500</td>
      <td>Cavendish, Mrs. Tyrell William (Julia Florence...</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>555</td>
      <td>19877</td>
      <td>Mrs</td>
    </tr>
  </tbody>
</table>
</div>



* 6 passengers are older than 70, among which 1 male of 80 years old, probably the oldest passenger in the titanic trip. And he actually survived to the catastrophy.

#### **>>> Outliers in the 'Fare' column.**


```python
df.Fare.plot(kind='hist', title='histogram for Fare', bins=20, color='g',figsize=(10,7));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_182_0.png)


* Most passengers paid between 0 and 100 for their trip. Few passengers paid exceptionally high fares than the others (> 400). A box plot on the fare can help see more clearly.


```python
df.Fare.plot(kind='box', title='Box plot for passenegr Fare', figsize=(10,7));  
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_184_0.png)


* All the fare values outside the whiskers plot boundaries confirm the presence of outliers, with a specific outlier meaningfully far from others beyond 500.
> What could have happened here...?


```python
# Looking into maximum fare outliers...
df.loc[df.Fare > 400]
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>259</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>C</td>
      <td>512.3292</td>
      <td>Ward, Miss. Anna</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>680</th>
      <td>36.0</td>
      <td>B51 B53 B55</td>
      <td>C</td>
      <td>512.3292</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>1</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>738</th>
      <td>35.0</td>
      <td>B101</td>
      <td>C</td>
      <td>512.3292</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1235</th>
      <td>58.0</td>
      <td>B51 B53 B55</td>
      <td>C</td>
      <td>512.3292</td>
      <td>Cardeza, Mrs. James Warburton Martinez (Charlo...</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>PC 17755</td>
      <td>Mrs</td>
    </tr>
  </tbody>
</table>
</div>



* These 4 passengers have paid th highest fare of 512. They obviously all travel in the first class and have the same ticket number(could be from the same family or at least they were in a group). There is also a possibility that they paid a high fare because they booked their ticket very late. It is well noticeable that 3 of them have survived the Titanic disaster. The fourth one is to be predicted.
> These outliers can be treated by applying some transformations to the fare column.


```python
# Transformation can reduce the skewness of the fare distribution
# ....as log(0) throws an error, 1 is deliberately added to the fare value in case the fare is 0.
Log_of_Fare = np.log(df.Fare + 1.0) 
```


```python
# Now the histogram, but this time for the LogFare
Log_of_Fare.plot(kind='hist', title='Histogram for the Log(Fare)', figsize=(10,6), color='c');
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_189_0.png)


* the fare distribution is less skewed. 
> Binnig can also be used to treat the Fare outliers by using the **qcut()** function.
  - qcut() performs quantile-based binning ('q' for quantile). this is to split the passengers fares in 4 bins, all containing the same number of fare observations.


```python
# binning on the Fare feature...4 is the number of bins
pd.qcut(df.Fare, 4)
```




    PassengerId
    1         (-0.001, 7.896]
    2       (31.275, 512.329]
    3         (7.896, 14.454]
    4       (31.275, 512.329]
    5         (7.896, 14.454]
    6         (7.896, 14.454]
    7       (31.275, 512.329]
                  ...        
    1303    (31.275, 512.329]
    1304      (-0.001, 7.896]
    1305      (7.896, 14.454]
    1306    (31.275, 512.329]
    1307      (-0.001, 7.896]
    1308      (7.896, 14.454]
    1309     (14.454, 31.275]
    Name: Fare, Length: 1309, dtype: category
    Categories (4, interval[float64]): [(-0.001, 7.896] < (7.896, 14.454] < (14.454, 31.275] < (31.275, 512.329]]



* **Note:** As seen in the result passengers fares are divide into 4 groups or bins to which some names were assigned, ranging from very low to high fare.
> What is happening here is that by giving those names to the 4 bins, there is a conversion of Fare from numerical values to a categorical.
> Each bin level is one category. this is called **Discretization::**(fast of creating discrete categories on continuous numerical features).


```python
# assigning level labels to the 4 Fare groups....or Discretization.
pd.qcut(df.Fare, 4, labels=['very_cheap','cheap','expensive','very_expensive'])   
```




    PassengerId
    1           very_cheap
    2       very_expensive
    3                cheap
    4       very_expensive
    5                cheap
    6                cheap
    7       very_expensive
                 ...      
    1303    very_expensive
    1304        very_cheap
    1305             cheap
    1306    very_expensive
    1307        very_cheap
    1308             cheap
    1309         expensive
    Name: Fare, Length: 1309, dtype: category
    Categories (4, object): [very_cheap < cheap < expensive < very_expensive]



- It then becomes easy to visualize the fare distribution by plotting fares of each bin.


```python
# this time a bar plot for the fare bins
pd.qcut(df.Fare, 4, labels=['very_cheap','cheap','expensive','very_expensive']).value_counts().plot(kind='bar', color='c', rot=0);
```


![png](very_long_titanic_trip_files/very_long_titanic_trip_195_0.png)


* The fare distributions in the 4 bins are quite normally distributed. These corrected fare values will be used from now on. I need to create a new column in the dataframe to store them.


```python
# Creating and adding a new column named : 'Bin_fare' in the dataframe df
df['Bin_Fare'] = pd.qcut(df.Fare, 4, labels=['very_cheap', 'cheap', 'expensive', 'very_expensive'])
```


```python
# check the new Bin_Fare added column
df
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
      <th>Bin_Fare</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>Mr</td>
      <td>very_cheap</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17599</td>
      <td>Mrs</td>
      <td>very_expensive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>STON/O2. 3101282</td>
      <td>Miss</td>
      <td>cheap</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>113803</td>
      <td>Mrs</td>
      <td>very_expensive</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>Mr</td>
      <td>cheap</td>
    </tr>
    <tr>
      <th>6</th>
      <td>29.0</td>
      <td>NaN</td>
      <td>Q</td>
      <td>8.4583</td>
      <td>Moran, Mr. James</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>Mr</td>
      <td>cheap</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54.0</td>
      <td>E46</td>
      <td>S</td>
      <td>51.8625</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>Mr</td>
      <td>very_expensive</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1303</th>
      <td>37.0</td>
      <td>C78</td>
      <td>Q</td>
      <td>90.0000</td>
      <td>Minahan, Mrs. William Edward (Lillian E Thorpe)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>555</td>
      <td>19928</td>
      <td>Mrs</td>
      <td>very_expensive</td>
    </tr>
    <tr>
      <th>1304</th>
      <td>28.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.7750</td>
      <td>Henriksson, Miss. Jenny Lovisa</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>347086</td>
      <td>Miss</td>
      <td>very_cheap</td>
    </tr>
    <tr>
      <th>1305</th>
      <td>29.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Spector, Mr. Woolf</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>A.5. 3236</td>
      <td>Mr</td>
      <td>cheap</td>
    </tr>
    <tr>
      <th>1306</th>
      <td>39.0</td>
      <td>C105</td>
      <td>C</td>
      <td>108.9000</td>
      <td>Oliva y Ocana, Dona. Fermina</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>0</td>
      <td>555</td>
      <td>PC 17758</td>
      <td>Lady</td>
      <td>very_expensive</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>38.5</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>SOTON/O.Q. 3101262</td>
      <td>Mr</td>
      <td>very_cheap</td>
    </tr>
    <tr>
      <th>1308</th>
      <td>29.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Ware, Mr. Frederick</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>555</td>
      <td>359309</td>
      <td>Mr</td>
      <td>cheap</td>
    </tr>
    <tr>
      <th>1309</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>C</td>
      <td>22.3583</td>
      <td>Peter, Master. Michael J</td>
      <td>1</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>2668</td>
      <td>Master</td>
      <td>expensive</td>
    </tr>
  </tbody>
</table>
<p>1309 rows × 13 columns</p>
</div>



* **Note:** In the future processes, instead of dealing with numeric fare values, The **Bin_Fare** feature will be used as it has no outliers that can biase the survival prediction.

==> This closes the detection and treatment of **missing values** and **outliers(or extreme values)** in the titanic dataset. I equally find it useful to go through some feature engineering to add more value to the dataframe.

## **3*- Feature Engineering (FE)**....Creating new features for the dataframe.
  * What is meant here is usually transform raw pieces of data in diverse ways to better plausible and representative features.
  * This helps to create better predictive model. FE can be initiated by different approaches:
     - Feature transformation: to have more usable data.
     - Feature creation: new feature based on existing ones that may not be well suited for data processing.
     - Feature selection

> ### **(1)- New feature: 'AgeDenomination'** (passeneger: Child or Adult):


```python
# New feature AgeDenomination from Age feature
df['AgeDenomination'] = np.where(df.Age >= 18, 'Adult', 'Child')
```


```python
# How many children and Adults in the Titanic dataset
df.AgeDenomination.value_counts()
```




    Adult    1147
    Child     162
    Name: AgeDenomination, dtype: int64



==> It will kind of interesting to check the survival rate for the AGedenomination: Adult vs Child


```python
# crosstab for survival rate of adults and children..only for passengers having a Survival information...briefly the entries in the dataset.
pd.crosstab(df[df.Survived != 500].Survived, df[df.Survived != 500].AgeDenomination)
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
      <th>AgeDenomination</th>
      <th>Adult</th>
      <th>Child</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>495</td>
      <td>54</td>
    </tr>
    <tr>
      <th>1</th>
      <td>279</td>
      <td>63</td>
    </tr>
    <tr>
      <th>555</th>
      <td>373</td>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>



* The survival rate is higher for children than for adults. More children (63 out of 162) survived to the catastrophy against 279 out of 1147 for adults.

> ### **(2)- New feature: 'FamilySize'** (based on parents and siblings columns):
  - family size could helpto get more details for the survival prediction. In case of a small family, if one of the members managed to get into the lifeboat, then others members of the same family might have got into the lifeboat as well. However in case of a big size family, there could be losing time and get into panic just to figure out who will first go on the lifeboat. 
  - I need to create a new feature: **'Size_family'** to explore the above hypotheses. this is done by summing up 'parent size' and 'sibling size features.


```python
# Sum of parents and siblings
df['Size_family'] = df.Parch + df.SibSp + 1      # 1 is for the self

# Extract the first 10 rows to check the 'Size_family' feature added.
```


```python
%%HTML
<img src="/images/highlight_size_family.png"></img>    <!--As done previously, I display here in an html markup an image highlighting the 'Size_family' feature.-->
```


<img src="/images/highlight_size_family.png"></img>    <!--As done previously, I display here in an html markup an image highlighting the 'Size_family' feature.-->


" **Dataframe backup :: I have the habit to save the current dataframe as .xlsx file as I proceed with the project. This is just personal, to keep track of modifications applied to the dataframe along the way .**


```python
# Saving the dataframe as 'highlight_Size_family.xlsx' file.
df.to_excel('highlight_Size_family.xlsx', encoding='UTF-8');
```


```python
# Let's check how the size of family is distributed.
df['Size_family'].plot(kind='hist', color='c', title='histogram for Size_family', figsize=(10,6));
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_212_0.png)


**Notes:** 
 - The first observation is that most of the passengers appear to be single or belong to family with 4 members maximum. 
 - In the far right, there are few falnilies with more than 10 members; They can be considered as outliers as they are form the normal behaviour of maximum 4 members per family.
 - let's try to see what these outliers have to tell us. Doing it by selecting rows with maximum family size.


```python
# extract rows with the highest family size
highest_sized_families = df[df.Size_family == df.Size_family.max()]      #equivalent to: df.loc[df.Size_family == df.Size_family.max()]
highest_sized_families
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
      <th>Bin_Fare</th>
      <th>AgeDenomination</th>
      <th>Size_family</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>160</th>
      <td>4.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Master. Thomas Henry</td>
      <td>2</td>
      <td>3</td>
      <td>male</td>
      <td>8</td>
      <td>0</td>
      <td>CA. 2343</td>
      <td>Master</td>
      <td>very_expensive</td>
      <td>Child</td>
      <td>11</td>
    </tr>
    <tr>
      <th>181</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Miss. Constance Gladys</td>
      <td>2</td>
      <td>3</td>
      <td>female</td>
      <td>8</td>
      <td>0</td>
      <td>CA. 2343</td>
      <td>Miss</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>202</th>
      <td>29.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Mr. Frederick</td>
      <td>2</td>
      <td>3</td>
      <td>male</td>
      <td>8</td>
      <td>0</td>
      <td>CA. 2343</td>
      <td>Mr</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>325</th>
      <td>29.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Mr. George John Jr</td>
      <td>2</td>
      <td>3</td>
      <td>male</td>
      <td>8</td>
      <td>0</td>
      <td>CA. 2343</td>
      <td>Mr</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>793</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Miss. Stella Anna</td>
      <td>2</td>
      <td>3</td>
      <td>female</td>
      <td>8</td>
      <td>0</td>
      <td>CA. 2343</td>
      <td>Miss</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>847</th>
      <td>29.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Mr. Douglas Bullen</td>
      <td>2</td>
      <td>3</td>
      <td>male</td>
      <td>8</td>
      <td>0</td>
      <td>CA. 2343</td>
      <td>Mr</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>864</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>2</td>
      <td>3</td>
      <td>female</td>
      <td>8</td>
      <td>0</td>
      <td>CA. 2343</td>
      <td>Miss</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Miss. Ada</td>
      <td>2</td>
      <td>3</td>
      <td>female</td>
      <td>8</td>
      <td>555</td>
      <td>CA. 2343</td>
      <td>Miss</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>29.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Mr. John George</td>
      <td>9</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>555</td>
      <td>CA. 2343</td>
      <td>Mr</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1252</th>
      <td>14.5</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Master. William Henry</td>
      <td>2</td>
      <td>3</td>
      <td>male</td>
      <td>8</td>
      <td>555</td>
      <td>CA. 2343</td>
      <td>Master</td>
      <td>very_expensive</td>
      <td>Child</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>69.55</td>
      <td>Sage, Mrs. John (Annie Bullen)</td>
      <td>9</td>
      <td>3</td>
      <td>female</td>
      <td>1</td>
      <td>555</td>
      <td>CA. 2343</td>
      <td>Mrs</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shrinking the result to columns of interest
df.loc[df.Size_family == df.Size_family.max(), ['Name','Age','Pclass','Size_family','Ticket', 'Survived']]
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
      <th>Name</th>
      <th>Age</th>
      <th>Pclass</th>
      <th>Size_family</th>
      <th>Ticket</th>
      <th>Survived</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>160</th>
      <td>Sage, Master. Thomas Henry</td>
      <td>4.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>181</th>
      <td>Sage, Miss. Constance Gladys</td>
      <td>22.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>202</th>
      <td>Sage, Mr. Frederick</td>
      <td>29.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>325</th>
      <td>Sage, Mr. George John Jr</td>
      <td>29.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>793</th>
      <td>Sage, Miss. Stella Anna</td>
      <td>22.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>847</th>
      <td>Sage, Mr. Douglas Bullen</td>
      <td>29.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>864</th>
      <td>Sage, Miss. Dorothy Edith "Dolly"</td>
      <td>22.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>Sage, Miss. Ada</td>
      <td>22.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>555</td>
    </tr>
    <tr>
      <th>1234</th>
      <td>Sage, Mr. John George</td>
      <td>29.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>555</td>
    </tr>
    <tr>
      <th>1252</th>
      <td>Sage, Master. William Henry</td>
      <td>14.5</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>555</td>
    </tr>
    <tr>
      <th>1257</th>
      <td>Sage, Mrs. John (Annie Bullen)</td>
      <td>35.0</td>
      <td>3</td>
      <td>11</td>
      <td>CA. 2343</td>
      <td>555</td>
    </tr>
  </tbody>
</table>
</div>



**Insights:** 
 - All above passengers are members of the same family. They are 11 with the same ticket number. The first name 'Sage' is also indicative. 
 - 7 of them did not survived to the catastrophy. the remaining 4 belong to the test file and their survival status is to predicted later on. They might also have not survived to the disaster.
     - This sounds as an interesting story that is being created here by digging into this 'Size_family' feature.
     > **How about analysing the effect of the Size_family on the survived rate of the passengers ?**


```python
# Correlation between Size_family and survival rate... only on existing survival values
pd.crosstab(df[df.Survived != 500].Survived, df[df.Survived != 500].Size_family)
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
      <th>Size_family</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>11</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>374</td>
      <td>72</td>
      <td>43</td>
      <td>8</td>
      <td>12</td>
      <td>19</td>
      <td>8</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>163</td>
      <td>89</td>
      <td>59</td>
      <td>21</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>555</th>
      <td>253</td>
      <td>74</td>
      <td>57</td>
      <td>14</td>
      <td>7</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



* **As a result**, 
   - more passengers survived in families with 2, 3, or 4 members. A plausible explanation could be that if one member succeeded to get in the lifeboat, the other members succeeded too.
   - On the other hand, the survival rate is really insignificant for dig families sized from 5 to 11 members.

* Glancing back at how the decision of who will get first into the lifeboats, raises the question of priority. Thinking first of mums with babies being prioritized over other passengers.
  * Wouldn't it be interesting to check the motherhood (or mum-baby instance) on the survival rate of the passengers ?
  * A new feature can be created and added to the data which will hold values about passengers in a **mum_baby** status.
  
      * the plausible focus will be on female passengers of more than 18 years old, having at leat one child on the titanic and who are possibly married (Though nowadays one can have a child without being married, in the year 1912, it was not the case).       

> ### **(3)- New feature: 'Mum_with_baby'** (based on combinations between the Sex and Parent features):


```python
# New feature with female passengers: 'Sex'=female, 'Age'>18, 'Parch'>0, 'Title' is not Missto say that she is married(so Lady and Mrs)
# ... If the passenger is a mum with baby we have a 1, otherwise the feature gets a 0.
df['Mum_with_baby'] = np.where(((df.Sex == 'female') & (df.Parch > 0) & (df.Age > 18) & (df.Title != 'Miss')), 1, 0)
```


```python
# Saving the dataframe as 'hightlight_mum_with_baby.xlsx' file.
df.to_excel('hightlight_mum_with_baby.xlsx', encoding='UTF-8');
# and display the first 10 rows highlighting the 'Mum_with_baby' feature.
```


```python
%%HTML
<img src='/images/hightlight_mum_with_baby.png'></img>
```


<img src='/images/hightlight_mum_with_baby.png'></img>


* Now crosstabing 'Survived' and 'Mum_with_baby' to check whether the fact of being a mother could affect the survival rate of passengers.


```python
# As usual crosstab on entries with survival information.
pd.crosstab(df[df.Survived != 500].Survived, df[df.Survived != 500].Mum_with_baby)
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
      <th>Mum_with_baby</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>533</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>303</td>
      <td>39</td>
    </tr>
    <tr>
      <th>555</th>
      <td>388</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>



* **As a result**, Of the 55 mothers in the ship, 39 survived to the catastrophy. This indicates that the fact of being a mother had an effect on the passenger's survival rate.

> ==> Another strategic feature to explore could be the **Deck** which can help in the following:
  - tell about the passenger location on the ship.
  - provide some orientation about the social ranking (wealthness is possible) of the passenger in relation to the availability and priority to the lifeboats.  
  
Dive in with me to explore the possibilityof  a Cabin information to affect the survival rate of its passengers !

> ### **(K)- New feature: 'Deck'** (based on 'Cabin' category):


```python
# extract Cabin information 
df.Cabin     # remember the max rows displayed in the notebook has been set to 15.
```




    PassengerId
    1        NaN
    2        C85
    3        NaN
    4       C123
    5        NaN
    6        NaN
    7        E46
            ... 
    1303     C78
    1304     NaN
    1305     NaN
    1306    C105
    1307     NaN
    1308     NaN
    1309     NaN
    Name: Cabin, Length: 1309, dtype: object



* **Infos on the Cabin features:  *Cabin = Deck...Room***
     - Cabin names follow a 2 parts pattern: alphabetic letter at the begining and a number at the end : 'C' - '85'.
     - As a guess, the letter might indicates the Deck while the numbers indicates the specific room of the passenger.
     - the fact of having a huge number of missings values for the Cabin column might indicate that those passengers were not given any specific Cabin.
     
> How many unique Cabins were in the ship ?     


```python
unique_cabins = df.Cabin.unique()
print(f'* There were {len(df.Cabin)} cabins in the ship among which {len(unique_cabins)} were unique: \n\n {unique_cabins}')
```

    * There were 1309 cabins in the ship among which 187 were unique: 
    
     [nan 'C85' 'C123' 'E46' 'G6' 'C103' 'D56' 'A6' 'C23 C25 C27' 'B78' 'D33'
     'B30' 'C52' 'B28' 'C83' 'F33' 'F G73' 'E31' 'A5' 'D10 D12' 'D26' 'C110'
     'B58 B60' 'E101' 'F E69' 'D47' 'B86' 'F2' 'C2' 'E33' 'B19' 'A7' 'C49'
     'F4' 'A32' 'B4' 'B80' 'A31' 'D36' 'D15' 'C93' 'C78' 'D35' 'C87' 'B77'
     'E67' 'B94' 'C125' 'C99' 'C118' 'D7' 'A19' 'B49' 'D' 'C22 C26' 'C106'
     'C65' 'E36' 'C54' 'B57 B59 B63 B66' 'C7' 'E34' 'C32' 'B18' 'C124' 'C91'
     'E40' 'T' 'C128' 'D37' 'B35' 'E50' 'C82' 'B96 B98' 'E10' 'E44' 'A34'
     'C104' 'C111' 'C92' 'E38' 'D21' 'E12' 'E63' 'A14' 'B37' 'C30' 'D20' 'B79'
     'E25' 'D46' 'B73' 'C95' 'B38' 'B39' 'B22' 'C86' 'C70' 'A16' 'C101' 'C68'
     'A10' 'E68' 'B41' 'A20' 'D19' 'D50' 'D9' 'A23' 'B50' 'A26' 'D48' 'E58'
     'C126' 'B71' 'B51 B53 B55' 'D49' 'B5' 'B20' 'F G63' 'C62 C64' 'E24' 'C90'
     'C45' 'E8' 'B101' 'D45' 'C46' 'D30' 'E121' 'D11' 'E77' 'F38' 'B3' 'D6'
     'B82 B84' 'D17' 'A36' 'B102' 'B69' 'E49' 'C47' 'D28' 'E17' 'A24' 'C50'
     'B42' 'C148' 'B45' 'B36' 'A21' 'D34' 'A9' 'C31' 'B61' 'C53' 'D43' 'C130'
     'C132' 'C55 C57' 'C116' 'F' 'A29' 'C6' 'C28' 'C51' 'C97' 'D22' 'B10'
     'E45' 'E52' 'A11' 'B11' 'C80' 'C89' 'F E46' 'B26' 'F E57' 'A18' 'E60'
     'E39 E41' 'B52 B54 B56' 'C39' 'B24' 'D40' 'D38' 'C105']


* **Remark:** If you look with me between the lines of the unique cabins list, we can extract few ones that do not follow the general 'letter-number' pattern. Notably Cabins 'T', 'F'.
 > What could exploring the passengers travelling in those cabins tell us ?


```python
# which passengers happened to be in the T Cabin
df.loc[df.Cabin == 'T']
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
      <th>Bin_Fare</th>
      <th>AgeDenomination</th>
      <th>Size_family</th>
      <th>Mum_with_baby</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>340</th>
      <td>45.0</td>
      <td>T</td>
      <td>S</td>
      <td>35.5</td>
      <td>Blackwell, Mr. Stephen Weart</td>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>113784</td>
      <td>Mr</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



* These 2 Cabins don't reveal any particular insight compared to the passengers in other Cabins. They can just be taken as mistakes and ignored. 
* May setting them as 'NaN' would be a better approach:


```python
# In the dataframe, gset the Cabin value to 'NaN' if the Cabin value is 'T'.
df.loc[(df.Cabin =='T') , 'Cabin'] = np.NaN
```

> Then looking again at the unique values of the Cabin...


```python
df.Cabin.unique()
```




    array([nan, 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6',
           'C23 C25 C27', 'B78', 'D33', 'B30', 'C52', 'B28', 'C83', 'F33',
           'F G73', 'E31', 'A5', 'D10 D12', 'D26', 'C110', 'B58 B60', 'E101',
           'F E69', 'D47', 'B86', 'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4',
           'A32', 'B4', 'B80', 'A31', 'D36', 'D15', 'C93', 'C78', 'D35',
           'C87', 'B77', 'E67', 'B94', 'C125', 'C99', 'C118', 'D7', 'A19',
           'B49', 'D', 'C22 C26', 'C106', 'C65', 'E36', 'C54',
           'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40',
           'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98', 'E10', 'E44', 'A34',
           'C104', 'C111', 'C92', 'E38', 'D21', 'E12', 'E63', 'A14', 'B37',
           'C30', 'D20', 'B79', 'E25', 'D46', 'B73', 'C95', 'B38', 'B39',
           'B22', 'C86', 'C70', 'A16', 'C101', 'C68', 'A10', 'E68', 'B41',
           'A20', 'D19', 'D50', 'D9', 'A23', 'B50', 'A26', 'D48', 'E58',
           'C126', 'B71', 'B51 B53 B55', 'D49', 'B5', 'B20', 'F G63',
           'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45', 'C46', 'D30',
           'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84', 'D17', 'A36',
           'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24', 'C50', 'B42',
           'C148', 'B45', 'B36', 'A21', 'D34', 'A9', 'C31', 'B61', 'C53',
           'D43', 'C130', 'C132', 'C55 C57', 'C116', 'F', 'A29', 'C6', 'C28',
           'C51', 'C97', 'D22', 'B10', 'E45', 'E52', 'A11', 'B11', 'C80',
           'C89', 'F E46', 'B26', 'F E57', 'A18', 'E60', 'E39 E41',
           'B52 B54 B56', 'C39', 'B24', 'D40', 'D38', 'C105'], dtype=object)



> Going deeper in exploring Deck information :: The deck is actually the first character in the Cabin value.
  - by extracting the Deck from the Cabin feature (1rst character).
  - create a separate Deck for the NaN Cabins.
    > by creating a little **'extract_deck'** which takes every Cabin as parameter, extract the deck and store it in a new 'Deck' feature.


```python
# function to extract decks from Cabins information.
def extract_deck(cab):
    # if the Cabin value is null, assign 'Deck NaN' to deck, 
    #.....otherwise first convert the Cabin value to string, then take the first character of the converted string. Changing the deck to uppercase is just a choice.
    deck = np.where(pd.notnull(cab), str(cab)[0].upper(), 'Deck NaN')
    return deck
```


```python
#ingUse the extract_deck() function to get the decks in the 'Deck' feature in the dataframe
df['deck'] = df.Cabin.map(lambda x : extract_deck(x))
```


```python
# Displaying the dataframe's 5 top rows...confirm the 'deck' feature.
df.head(5)
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
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
      <th>Bin_Fare</th>
      <th>AgeDenomination</th>
      <th>Size_family</th>
      <th>Mum_with_baby</th>
      <th>deck</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>22.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>Mr</td>
      <td>very_cheap</td>
      <td>Adult</td>
      <td>2</td>
      <td>0</td>
      <td>Deck NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>C85</td>
      <td>C</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>PC 17599</td>
      <td>Mrs</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>2</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>26.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>female</td>
      <td>0</td>
      <td>1</td>
      <td>STON/O2. 3101282</td>
      <td>Miss</td>
      <td>cheap</td>
      <td>Adult</td>
      <td>1</td>
      <td>0</td>
      <td>Deck NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>C123</td>
      <td>S</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>female</td>
      <td>1</td>
      <td>1</td>
      <td>113803</td>
      <td>Mrs</td>
      <td>very_expensive</td>
      <td>Adult</td>
      <td>2</td>
      <td>0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>5</th>
      <td>35.0</td>
      <td>NaN</td>
      <td>S</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>Mr</td>
      <td>cheap</td>
      <td>Adult</td>
      <td>1</td>
      <td>0</td>
      <td>Deck NaN</td>
    </tr>
  </tbody>
</table>
</div>



* The Deck information is now added as a feature in the dataframe.  - Let's check how passengers are distributed per deck


```python
# Counting passengers per deck
df['deck'].value_counts()
```




    Deck NaN    1015
    C             94
    B             65
    D             46
    E             41
    A             22
    F             21
    G              5
    Name: deck, dtype: int64



* Most of the passenger are in the 'Deck NaN' which is the deck we have imputed to those passengers with no plausible deck information.
> What could be the impact of the deck information on the survival rate ? Remember our goal in this study is at the end of the day, to predict Passenger survival status.


```python
# survival rate per deck... as usual appying to well defined 'Survived' values.
pd.crosstab(df[df.Survived != 555].Survived, df[df.Survived != 555].deck)
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
      <th>deck</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>Deck NaN</th>
      <th>E</th>
      <th>F</th>
      <th>G</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>12</td>
      <td>24</td>
      <td>8</td>
      <td>482</td>
      <td>8</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>35</td>
      <td>35</td>
      <td>25</td>
      <td>206</td>
      <td>24</td>
      <td>8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



**Results:**
  - Some decks have relatively quite high survival rates, such as   
  > Deck B with 35 passengers out of 47 survived.  
  > Deck C: out of 59 passengers, 35 survived.
  
  > Deck D: as the highest survival rate passengers deck: only 8 out of 33 passengers could not make it.  
  
**!!** So the Deck information could be another good indicator in predicting the survival rate of the titanic passengers.


```python
# current info on the dataframe to have a feel of all the features we have added to the dataframe.
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 17 columns):
    Age                1309 non-null float64
    Cabin              294 non-null object
    Embarked           1309 non-null object
    Fare               1309 non-null float64
    Name               1309 non-null object
    Parch              1309 non-null int64
    Pclass             1309 non-null int64
    Sex                1309 non-null object
    SibSp              1309 non-null int64
    Survived           1309 non-null int64
    Ticket             1309 non-null object
    Title              1309 non-null object
    Bin_Fare           1309 non-null category
    AgeDenomination    1309 non-null object
    Size_family        1309 non-null int64
    Mum_with_baby      1309 non-null int64
    deck               1309 non-null object
    dtypes: category(1), float64(2), int64(6), object(8)
    memory usage: 215.3+ KB


Starting with 11 columns, the titanic dataframe now possesses 17 columns of different data types:
  > Some of the features are of int64 type : Parch, Pclass, SibSp..  
  > two of them are of type float64: Fare and Age.
  
  > and others are of type object: these are CATEGORICAL features of the dataframe. They are usually in the form of string.
     - the categorical features are super useful in better understanding the data, but they mostly cannot be processed by most algorithms as strings as they are.
     - They need to be turned into numeric values. There is an efficient technique called 'Categorical Feature Encoding'
      (CFE) which just does that.
      
      - In the next next step of the disaster trip, categorical features of the titanic data will be turned into numerical form using CFE.
CFE can be done in multiple ways: 
  - **_binary encoding_**:  'Sex' column: male:1, female:0 or the other xway around.
  - **_label encoding_**: 'Bin_fare' feature: 4 categories in an intrinsic order (very_cheap:0, cheap:1, expensive:2, very_expensive:3)
  - **_one-hot encoding_**: 'Embarked' feature (C, S, Q) with no particular intrinsic order. Pretty safe method to turn categorical feature into numeric values.

> ### **(L)- Categorical Feature Encoding (CFE)**:

### * **Binary encoding the 'Gender' feature**
> by creating a new feature (IsFemale) on the dataframe which shall hold the nnumeric values encoded from the Sex column. 0 for female and 1 otherwise.


```python
# Sex feature binary encoding
df['IsFemale'] = np.where(df.Sex == 'female', 0, 1)
```


```python
# confirm the Sex binary encoding in 'IsFemale' column.
df[['Title', 'deck', 'Sex','IsFemale']].head()
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
      <th>Title</th>
      <th>deck</th>
      <th>Sex</th>
      <th>IsFemale</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mr</td>
      <td>Deck NaN</td>
      <td>male</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mrs</td>
      <td>C</td>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Miss</td>
      <td>Deck NaN</td>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mrs</td>
      <td>C</td>
      <td>female</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Mr</td>
      <td>Deck NaN</td>
      <td>male</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### * **Binary encoding the 'AgeDenomination' feature**


```python
# AgeDenomination feature binary encoding
df['IsChild'] = np.where(df.AgeDenomination == 'Child', 1, 0)

df[['Title', 'Age', 'AgeDenomination', 'IsChild']].head(3)
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
      <th>Title</th>
      <th>Age</th>
      <th>AgeDenomination</th>
      <th>IsChild</th>
    </tr>
    <tr>
      <th>PassengerId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Mr</td>
      <td>22.0</td>
      <td>Adult</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mrs</td>
      <td>38.0</td>
      <td>Adult</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Miss</td>
      <td>26.0</td>
      <td>Adult</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### * **One-hot encoding for several features**
> by encoding many features at once with the help of the pandas get_dummies() function. 
- The categorical columns to encode are passed to the column attribute of the **'get_dummies()'** function. - It is done like this:


```python
# One-hot encoding on several features and overwriting the dataframe df contents. 
df = pd.get_dummies(df, columns=['Embarked', 'Pclass', 'deck', 'Title', 'AgeDenomination', 'Bin_Fare']) 
```

* Note that although 'AgeDenomination' is a binary category it is accepted by the get_dummies() funtion for encoding.The get_dummies() is such a nice function that it will even accept binary features.

**==> The categorical variables are encoded. What is the actual state of the titanic dataset ?**


```python
# post CFE dataframe backup.
df.to_excel('df_postCFE.xlsx', encoding='UTF-8')
```


```python
# Current state of the titanic dataframe
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 40 columns):
    Age                        1309 non-null float64
    Cabin                      294 non-null object
    Fare                       1309 non-null float64
    Name                       1309 non-null object
    Parch                      1309 non-null int64
    Sex                        1309 non-null object
    SibSp                      1309 non-null int64
    Survived                   1309 non-null int64
    Ticket                     1309 non-null object
    Size_family                1309 non-null int64
    Mum_with_baby              1309 non-null int64
    IsFemale                   1309 non-null int64
    IsChild                    1309 non-null int64
    Embarked_C                 1309 non-null uint8
    Embarked_Q                 1309 non-null uint8
    Embarked_S                 1309 non-null uint8
    Pclass_1                   1309 non-null uint8
    Pclass_2                   1309 non-null uint8
    Pclass_3                   1309 non-null uint8
    deck_A                     1309 non-null uint8
    deck_B                     1309 non-null uint8
    deck_C                     1309 non-null uint8
    deck_D                     1309 non-null uint8
    deck_Deck NaN              1309 non-null uint8
    deck_E                     1309 non-null uint8
    deck_F                     1309 non-null uint8
    deck_G                     1309 non-null uint8
    Title_Lady                 1309 non-null uint8
    Title_Master               1309 non-null uint8
    Title_Miss                 1309 non-null uint8
    Title_Mr                   1309 non-null uint8
    Title_Mrs                  1309 non-null uint8
    Title_Officer              1309 non-null uint8
    Title_Sir                  1309 non-null uint8
    AgeDenomination_Adult      1309 non-null uint8
    AgeDenomination_Child      1309 non-null uint8
    Bin_Fare_very_cheap        1309 non-null uint8
    Bin_Fare_cheap             1309 non-null uint8
    Bin_Fare_expensive         1309 non-null uint8
    Bin_Fare_very_expensive    1309 non-null uint8
    dtypes: float64(2), int64(7), object(4), uint8(27)
    memory usage: 217.7+ KB


~~ **Observation:** From 11 columns in the original dataset, there is now 41 features describing titanic passengers. However some columns are not numerically encoded such as:

> Cabin: used to create the **'deck'** feature will not be of help anymore. It can then be removed from the dataset.

> Sex: used to generate the numeric column 'IsFemale', can also be neglected.

> Name: used to create the **'Title'** feature which will be of better help in predicting the passenger survival rate. Can then be dropped as well.

> Ticket: was not used to generate any new feature. It may be intersting to dig into this feature to see if it has an impact on the passenger's survival rate. 

> Parch and SibSp: weere used to generate the '**Size_family**' feature which is more interesting as pertain to the passenger survival prediction. These 2 can be dropeed as well.

#### ~~ **Question:** Is there any interesting story behind the 'Ticket' feature ?


```python
# exploring tickets 
df.Ticket
```




    PassengerId
    1                A/5 21171
    2                 PC 17599
    3         STON/O2. 3101282
    4                   113803
    5                   373450
    6                   330877
    7                    17463
                   ...        
    1303                 19928
    1304                347086
    1305             A.5. 3236
    1306              PC 17758
    1307    SOTON/O.Q. 3101262
    1308                359309
    1309                  2668
    Name: Ticket, Length: 1309, dtype: object




```python
# unique tickets
number_unique_tickets = len(df.Ticket.unique())
print(f'{number_unique_tickets} unique tickets.')
```

    929 unique tickets.


~~ **Observation:** 
  - The Ticket info show some general pattern which a number **'330877'**. 
  - However some other tickets show a slightly different pattern: a prefix before the number such as  **'STON/O2. 3101282'** or **'A.5. 3236'**

**~~ Question: **Is the prefix before the number significant in the ticket attribution. Is there any particular insight relating this types of ticket and the passenger's survival rate ??
> Checking that out...


```python
# Function to extract the prefix from the ticket pattern
#...if there is a space in the ticket value, split the ticket by that space(' ') and keep the first part ([0]) as the prefix.
#...in case of no space in the ticket value, return 'no exp' as prefix
def extract_prefix(ticket):
    if ' ' in ticket:  
        prefix = ticket.split(' ')[0]
        prefix = prefix.strip()
    elif ' ' not in ticket:
        prefix = 'no pref'
        
    return prefix
```


```python
# Create a new feature 'prefix' from the extracted prefixes from Ticket
df['Ticket_prefix'] = df.Ticket.map(lambda x : extract_prefix(x))
```


```python
# expressions in tickets
df.Ticket_prefix
```




    PassengerId
    1              A/5
    2               PC
    3         STON/O2.
    4          no pref
    5          no pref
    6          no pref
    7          no pref
               ...    
    1303       no pref
    1304       no pref
    1305          A.5.
    1306            PC
    1307    SOTON/O.Q.
    1308       no pref
    1309       no pref
    Name: Ticket_prefix, Length: 1309, dtype: object




```python
## number of unique prefixes in tickets
f'{len(df.Ticket_prefix.unique())} unique prefixes in the dataframe'
```




    '50 unique prefixes in the dataframe'




```python
df.Ticket_prefix.unique()
```




    array(['A/5', 'PC', 'STON/O2.', 'no pref', 'PP', 'A/5.', 'C.A.', 'A./5.',
           'SC/Paris', 'S.C./A.4.', 'A/4.', 'CA', 'S.P.', 'S.O.C.', 'SO/C',
           'W./C.', 'SOTON/OQ', 'W.E.P.', 'STON/O', 'A4.', 'C', 'SOTON/O.Q.',
           'SC/PARIS', 'S.O.P.', 'A.5.', 'Fa', 'CA.', 'F.C.C.', 'W/C',
           'SW/PP', 'SCO/W', 'P/PP', 'SC', 'SC/AH', 'A/S', 'A/4', 'WE/P',
           'S.W./PP', 'S.O./P.P.', 'F.C.', 'SOTON/O2', 'S.C./PARIS',
           'C.A./SOTON', 'SC/A.3', 'STON/OQ.', 'SC/A4', 'AQ/4', 'A.', 'LP',
           'AQ/3.'], dtype=object)



 * Having extracted the expressions from the tickets, some similarities can be sen between some of them. Grouping them in groups or bins according to their similarities looks like this:


```python
# extract prefixes from tickets and group them by name similarity.
def extract_prefix_bins(ticket):
    # dictionnary to map the prefixes to their repective groups
    prefix_bins = {
        'A/5':'A5', 'PC':'PC', 'STON/O2.':'STON', 'no pref':'No pref', 'PP':'PP', 'A/5.':'A5', 'C.A.':'CA', 'A./5.':'A5', 'SC/Paris':'SCParis', 'S.C./A.4.':'SCA4', 
        'A/4.':'A4', 'CA':'CA', 'S.P.':'SP', 'S.O.C.':'SOC', 'SO/C':'SOC', 'W./C.':'WC', 'SOTON/OQ':'SOTON', 'W.E.P.':'WEP', 'STON/O':'STON', 'A4.':'A4',
        'C':'C', 'SOTON/O.Q.':'SOTON', 'SC/PARIS':'SCParis', 'S.O.P.':'SOP', 'A.5.':'A5', 'Fa':'Fa', 'CA.':'CA', 'F.C.C.':'FCC', 'W/C':'WC', 'SW/PP':'SWPP',
        'SCO/W':'SCOW', 'P/PP':'PPP', 'SC':'SC', 'SC/AH':'SCAH', 'A/S':'AS', 'A/4':'A4', 'WE/P':'WEP', 'S.W./PP':'SWPP', 'S.O./P.P.':'SOPP', 'F.C.':'FC',
        'SOTON/O2':'SOTON02', 'S.C./PARIS':'SCParis', 'C.A./SOTON':'CASOTON', 'SC/A.3':'SCA3', 'STON/OQ.':'STON', 'SC/A4':'SCA4', 'AQ/4':'AQ4', 'A.':'A', 'LP':'LP',
        'AQ/3.':'AQ3',
    }
    
    if ' ' in ticket:
        prefix = ticket.split(' ')[0]
        prefix = prefix.strip()
    elif ' ' not in ticket:
        prefix = 'no pref'
               
    return prefix_bins[prefix]    
```


```python
df.Ticket.map(lambda x : extract_prefix_bins(x))
```




    PassengerId
    1            A5
    2            PC
    3          STON
    4       No pref
    5       No pref
    6       No pref
    7       No pref
             ...   
    1303    No pref
    1304    No pref
    1305         A5
    1306         PC
    1307      SOTON
    1308    No pref
    1309    No pref
    Name: Ticket, Length: 1309, dtype: object




```python
# apply the extract_prefix_bins function  to the dataset
df['bins_ticket_prefix'] = df.Ticket.map(lambda x : extract_prefix_bins(x))
```


```python
pd.options.display.max_columns = 50
```


```python
pd.crosstab(df[df.Survived != 555].Survived, df[df.Survived != 555].bins_ticket_prefix)
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
      <th>bins_ticket_prefix</th>
      <th>A4</th>
      <th>A5</th>
      <th>AS</th>
      <th>C</th>
      <th>CA</th>
      <th>CASOTON</th>
      <th>FC</th>
      <th>FCC</th>
      <th>Fa</th>
      <th>No pref</th>
      <th>PC</th>
      <th>PP</th>
      <th>PPP</th>
      <th>SC</th>
      <th>SCA4</th>
      <th>SCAH</th>
      <th>SCOW</th>
      <th>SCParis</th>
      <th>SOC</th>
      <th>SOP</th>
      <th>SOPP</th>
      <th>SOTON</th>
      <th>SOTON02</th>
      <th>SP</th>
      <th>STON</th>
      <th>SWPP</th>
      <th>WC</th>
      <th>WEP</th>
    </tr>
    <tr>
      <th>Survived</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>19</td>
      <td>1</td>
      <td>3</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>410</td>
      <td>21</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>13</td>
      <td>2</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>255</td>
      <td>39</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



**~~ Observation:** except for the passengers having ticket information staritng with 'PC' seem to show a higher survival rate (65%), there is no significant insight that can be drawn from the ticket type as pertain to the passengers' survival rate.
* So for now the ticket column can be removed as well from the dataframe.

**==> Let us remove the useless columns from the dataframe**

### * **Titanic dataframe: Dropping and re-arranging columns**


```python
# dropping colums 'Cabin', 'Sex', 'Name', 'Ticket', 'Parch' and 'SibSp' ; axis=1 for columns; inplace=True for overwriting the dataframe withoutmaking a copy of it.
df.drop(['Cabin', 'Sex', 'Name', 'Ticket', 'Parch', 'SibSp', 'bins_ticket_prefix', 'Ticket_prefix'],  axis=1, inplace=True)
```


```python
df.columns
```




    Index(['Age', 'Fare', 'Survived', 'Size_family', 'Mum_with_baby', 'IsFemale',
           'IsChild', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1',
           'Pclass_2', 'Pclass_3', 'deck_A', 'deck_B', 'deck_C', 'deck_D',
           'deck_Deck NaN', 'deck_E', 'deck_F', 'deck_G', 'Title_Lady',
           'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Officer',
           'Title_Sir', 'AgeDenomination_Adult', 'AgeDenomination_Child',
           'Bin_Fare_very_cheap', 'Bin_Fare_cheap', 'Bin_Fare_expensive',
           'Bin_Fare_very_expensive'],
          dtype='object')




```python
len(df.columns)
```




    34




```python
# re-ordering columns
columns = [column for column in df.columns if column != 'Survived']  # list of columns without the 'Survived' feature.
columns = ['Survived'] + columns  # add the 'Survived' column in frolt of the list, to directly see the prdiction....again, this is a personal thing.
df = df[columns]      # assiging the df with the reordered columns back to the dataframe
```


```python
#check the order of columns
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 1 to 1309
    Data columns (total 34 columns):
    Survived                   1309 non-null int64
    Age                        1309 non-null float64
    Fare                       1309 non-null float64
    Size_family                1309 non-null int64
    Mum_with_baby              1309 non-null int64
    IsFemale                   1309 non-null int64
    IsChild                    1309 non-null int64
    Embarked_C                 1309 non-null uint8
    Embarked_Q                 1309 non-null uint8
    Embarked_S                 1309 non-null uint8
    Pclass_1                   1309 non-null uint8
    Pclass_2                   1309 non-null uint8
    Pclass_3                   1309 non-null uint8
    deck_A                     1309 non-null uint8
    deck_B                     1309 non-null uint8
    deck_C                     1309 non-null uint8
    deck_D                     1309 non-null uint8
    deck_Deck NaN              1309 non-null uint8
    deck_E                     1309 non-null uint8
    deck_F                     1309 non-null uint8
    deck_G                     1309 non-null uint8
    Title_Lady                 1309 non-null uint8
    Title_Master               1309 non-null uint8
    Title_Miss                 1309 non-null uint8
    Title_Mr                   1309 non-null uint8
    Title_Mrs                  1309 non-null uint8
    Title_Officer              1309 non-null uint8
    Title_Sir                  1309 non-null uint8
    AgeDenomination_Adult      1309 non-null uint8
    AgeDenomination_Child      1309 non-null uint8
    Bin_Fare_very_cheap        1309 non-null uint8
    Bin_Fare_cheap             1309 non-null uint8
    Bin_Fare_expensive         1309 non-null uint8
    Bin_Fare_very_expensive    1309 non-null uint8
    dtypes: float64(2), int64(5), uint8(27)
    memory usage: 156.3 KB


* NOW, the raw titanic datasets are all converted to processed data (normally digestible by most algorithms), useful for building the model to predict passenger survival status. Let us save and write them to a file.

### * **Saving the processed datasets and wrting them to files**


```python
# path to save the processed files
processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')   # 'processed' subfolder was initiated when installing the cookiecutter environment.
write_train_path = os.path.join(processed_data_path, 'train.csv')         # path to the processed trainning file.
write_test_path = os.path.join(processed_data_path, 'test.csv')           # path to the processed test file.
```


```python
# extract specific rows from the dataframe and write them to 'train.csv'
df.loc[df.Survived != 555].to_csv(write_train_path)       # rows with Survived = 555 don't belong to the training set.

# extract specific rows from the dataframe and write them to 'test.csv'
 #...Because the raw test.csv had no 'Survived' column, 
 #.....and to keep the same structure for the processed test.csv file, 
 #.......create a list of columns without the 'Survived' feature; then used that list to extract all columns except the 'Survived' one
test_columns = [column for column in df.columns if column != 'Survived']
df.loc[df.Survived == 555, test_columns].to_csv(write_test_path)        # rows with Survived value of 555 belong to the test set (to be predicted).
```


```python
train_df = pd.read_csv(write_train_path)
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 35 columns):
    PassengerId                891 non-null int64
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
    dtypes: float64(2), int64(33)
    memory usage: 243.7 KB



```python
test_df = pd.read_csv(write_test_path)
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 34 columns):
    PassengerId                418 non-null int64
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
    dtypes: float64(2), int64(32)
    memory usage: 111.1 KB


## **4*- Few Advanced Visualizations using Matplotlib and Seaborn**: 
  > Matplotlib is an excellent library when it comes to customize some aspects of the visualizations.
  
  > Seaborn builds on top of Matplotlib and is used to visualize features combinations.


```python
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
plt.hist(df.Age)
```




    (array([ 80.,  62., 324., 426., 189., 108.,  66.,  41.,  10.,   3.]),
     array([ 0.17 ,  8.153, 16.136, 24.119, 32.102, 40.085, 48.068, 56.051,
            64.034, 72.017, 80.   ]),
     <a list of 10 Patch objects>)




![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_293_1.png)



```python
# plt.show() hides the information created internally by matplotlib
plt.hist(df.Age, bins=20, color='c')
plt.show() 
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_294_0.png)



```python
# Adding more components to the plot
plt.hist(df.Age, bins=20, color='c')
plt.title('Histogram for passengers age values')
plt.xlabel('Bins')
plt.ylabel('Counts')
plt.show()
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_295_0.png)


 ~~ **The exact same plot can be generated by using the axis object explicitely**
   - This can be very when adding subplots in a single visualization.


```python
# first extract the figure f and the axis object ax with plt.subplots()
f , ax = plt.subplots()

# Create the plot components using this time the axis object ax
ax.hist(df.Age, bins=20, color='c')
ax.set_title('Histogram: passenger age')
ax.set_xlabel('Bins')
ax.set_ylabel('Counts')
plt.show()
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_297_0.png)


 ~~ **Let us benefit from the ax object to create 2 subplots side by side**
   - This can be very when adding subplots in a single visualization.


```python
# Adding 2 subplots side by side: 2 axis: ax1 for the first subplot, ax2 for the second subplot; 1 row and 2 columns (1,2); figure of width =15 anf height=5 (15, 5)
f , (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# subplot1
ax1.hist(df.Fare, bins=20, color='darkslateblue')
ax1.set_title('Histogram: Passenger Fare')
ax1.set_xlabel('Bins')
ax2.set_ylabel('Counts')

#subplot2
ax2.hist(df.Age, bins=20, color='tomato')
ax2.set_title('Histogram: Passenger Age')
ax2.set_xlabel('Bins')
ax2.set_ylabel('Counts')

plt.show()
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_299_0.png)


 ~~ **Let us push it a bit further to create 5 subplots of different types in the same figure**
   - our figure hosting the subplots shall have 2 rows and 3 columns.


```python
f , axis_arr = plt.subplots(2, 3, figsize=(14, 7))

# subplot on the first row and first column of the figure
axis_arr[0, 0].hist(df.Fare, bins=20, color='darkslateblue')
axis_arr[0, 0].set_title('Histogram: Passenger Fare')
axis_arr[0, 0].set_xlabel('Bins')
axis_arr[0, 0].set_ylabel('Counts')

# subplot first row, second column
axis_arr[0, 1].hist(df.Age, bins=20, color='darkslateblue')
axis_arr[0, 1].set_title('Histogram: Passenger Age')
axis_arr[0, 1].set_xlabel('Bins')
axis_arr[0, 1].set_ylabel('Counts')

# boxplot Passenger's Fare on the first row and third column
axis_arr[0, 2].boxplot(df.Fare.values)
axis_arr[0, 2].set_title('Box plot: Passenger Fare')
axis_arr[0, 2].set_xlabel('Bins')
axis_arr[0, 2].set_ylabel('Counts')


# boxplot for Passenger's Age on the second row and first column
axis_arr[1, 0].boxplot(df.Age.values)
axis_arr[1, 0].set_title('Box plot: Passeger Age')
axis_arr[1, 0].set_xlabel('Bins')
axis_arr[1, 0].set_ylabel('Counts')

# scatter plot Passenger's Age vs Fare on the second row and the 2 column
axis_arr[1, 1].scatter(df.Age, df.Fare, color='darkslateblue', alpha=0.15)
axis_arr[1, 1].set_title('Scatter plot: Passenger: Age vs Fare')
axis_arr[1, 1].set_xlabel('Bins')
axis_arr[1, 1].set_ylabel('Counts')



plt.show()
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_301_0.png)


* Notice theoverlapings betwenn the subplots.  Easilyfix it with **'plt.tight_layout()'**


```python
f , axis_arr = plt.subplots(2, 3, figsize=(14, 7))

# subplot on the first row and first column of the figure
axis_arr[0, 0].hist(df.Fare, bins=20, color='darkslateblue')
axis_arr[0, 0].set_title('Histogram: Passenger Fare')
axis_arr[0, 0].set_xlabel('Bins')
axis_arr[0, 0].set_ylabel('Counts')

# subplot first row, second column
axis_arr[0, 1].hist(df.Age, bins=20, color='darkslateblue')
axis_arr[0, 1].set_title('Histogram: Passenger Age')
axis_arr[0, 1].set_xlabel('Bins')
axis_arr[0, 1].set_ylabel('Counts')

# boxplot Passenger's Fare on the first row and third column
axis_arr[0, 2].boxplot(df.Fare.values)
axis_arr[0, 2].set_title('Box plot: Passenger Fare')
axis_arr[0, 2].set_xlabel('Bins')
axis_arr[0, 2].set_ylabel('Counts')


# boxplot for Passenger's Age on the second row and first column
axis_arr[1, 0].boxplot(df.Age.values)
axis_arr[1, 0].set_title('Box plot: Passeger Age')
axis_arr[1, 0].set_xlabel('Bins')
axis_arr[1, 0].set_ylabel('Counts')

# scatter plot Passenger's Age vs Fare on the second row and the 2 column
axis_arr[1, 1].scatter(df.Age, df.Fare, color='darkslateblue', alpha=0.15)
axis_arr[1, 1].set_title('Scatter plot: Passenger: Age vs Fare')
axis_arr[1, 1].set_xlabel('Bins')
axis_arr[1, 1].set_ylabel('Counts')

# tight the figure layout to remove the overlaping between subplots
plt.tight_layout()


plt.show()
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_303_0.png)


* Notice that the subplot on the second row and third column has not been initialized. Setting its axes off with **axis_arr[1,2].axis('off')**


```python
f , axis_arr = plt.subplots(2, 3, figsize=(14, 7))

# subplot on the first row and first column of the figure
axis_arr[0, 0].hist(df.Fare, bins=20, color='darkslateblue')
axis_arr[0, 0].set_title('Histogram: Passenger Fare')
axis_arr[0, 0].set_xlabel('Bins')
axis_arr[0, 0].set_ylabel('Counts')

# subplot first row, second column
axis_arr[0, 1].hist(df.Age, bins=20, color='darkslateblue')
axis_arr[0, 1].set_title('Histogram: Passenger Age')
axis_arr[0, 1].set_xlabel('Bins')
axis_arr[0, 1].set_ylabel('Counts')

# boxplot Passenger's Fare on the first row and third column
axis_arr[0, 2].boxplot(df.Fare.values)
axis_arr[0, 2].set_title('Box plot: Passenger Fare')
axis_arr[0, 2].set_xlabel('Bins')
axis_arr[0, 2].set_ylabel('Counts')


# boxplot for Passenger's Age on the second row and first column
axis_arr[1, 0].boxplot(df.Age.values)
axis_arr[1, 0].set_title('Box plot: Passeger Age')
axis_arr[1, 0].set_xlabel('Bins')
axis_arr[1, 0].set_ylabel('Counts')

# scatter plot Passenger's Age vs Fare on the second row and the 2 column
axis_arr[1, 1].scatter(df.Age, df.Fare, color='darkslateblue', alpha=0.15)
axis_arr[1, 1].set_title('Scatter plot: Passenger: Age vs Fare')
axis_arr[1, 1].set_xlabel('Bins')
axis_arr[1, 1].set_ylabel('Counts')

# fixing the subplots' overlaping
plt.tight_layout()

#fixing the emptiness of the plot first row third column
axis_arr[1,2].axis('off')


plt.show()
```


![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_305_0.png)


#### * **Using Seaborn for visualization**


```python
import seaborn as sns
```


```python
sns.jointplot(df.Age, df.Fare,kind='hex', stat_func=None);
```

    /home/cv-dlbox/anaconda3/lib/python3.6/site-packages/matplotlib/axes/_axes.py:6462: UserWarning: The 'normed' kwarg is deprecated, and has been replaced by the 'density' kwarg.
      warnings.warn("The 'normed' kwarg is deprecated, and has been "



![png](/images/very_long_titanic_trip_files/very_long_titanic_trip_308_1.png)


#### Now that we have the titanic data processed and saved, the prediction model can be build. Join me on the next page to train our titanic prediction model.
