## ☂️🌂🌞 Train Test 🌞🌂☂️
- Standard Benchmarking: The Iris dataset is a built-in toy dataset within SKLearn, making it easily accessible for testing algorithms without external files. It contains measurements for 150 iris flowers across three different species: setosa, versicolor, and virginica.
- Feature Composition: Each sample in the dataset consists of four numerical features: sepal length, sepal width, petal length, and petal width. These features are stored in a NumPy array, allowing for efficient mathematical operations during the training process.
- Target Labels: The dataset includes a target array that maps each sample to an integer representing its specific species. This categorical data is what your machine learning model will attempt to predict based on the physical dimensions.
- Dictionary-Like Structure: When you call load_iris(), the data is returned as a "Bunch" object, which behaves similarly to a Python dictionary. You can access the raw data using keys like .data for features and .target for the classification labels.
- Data Exploration: Because the dataset is small and balanced, it is ideal for visualizing how different classes overlap in feature space. Analysts often use this data to practice scatter plots and pair plots before moving to more complex sets.
- Split the dataset using the Train Test Split function from the SKLearn library.

```
import sklearn
from sklearn import datasets
```

## ☂️🌂🌞 Load Iris Dataset 🌞🌂☂️
- The sklearn library provides the iris dataset, which is a dataset that is commonly used for classification problems.
- This dataset has 150 samples.

```
iris = datasets.load_iris()
```

## ☂️🌂🌞 Separate Attributes and Labels on Dataset Slices 🌞🌂☂️
- The iris dataset from the Sklearn library cannot be directly used by an ML model.
- In accordance with what was discussed in the previous module, we must separate attributes and labels in the dataset.
  
```
x=iris.data
y=iris.target

from sklearn.model_selection import train_test_split
```
 
## ☂️🌂🌞 Dividing the Dataset into Training and Testing 🌞🌂☂️
- To create a train set and test set we just call the train_test_split function.
- Train test split has parameters x, namely the attributes of the dataset, y, namely the target dataset, and test_size, namely the percentage of the test set from the complete dataset.
- Train test split returns 4 values namely, attributes of the train set, attributes of the test set, targets of the train set, and targets of the test set.
  
```
x_train,
x_test,
y_train,
y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```

## ☂️🌂🌞 Calculate The Length or Amount of Data in Each Split Directory 🌞🌂☂️
- When we print the length of x_test, we can see that the length of the test set attribute is 30 samples, according to the parameters we entered in the train_test_split function, namely 0.2 or 20% of 150 samples.
- The code to print the length of x_test is as below.
  
```
print(len(x))
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
```

## ☂️🌂🌞 Output 🌞🌂☂️
The following is the output of the code as shown below.
```
150
120
30
120
30
```

![image](https://github.com/diantyapitaloka/Sklearn-Traintest/assets/147487436/61ab5605-73b7-44fb-8d02-f58e6d0a0e7e)

## ☂️🌂🌞 Licences 🌞🌂☂️
Copyright by Diantya Pitaloka
