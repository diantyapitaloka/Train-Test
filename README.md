## ☂️🌂🌞 Train Test 🌞🌂☂️
- Standard Benchmarking: The Iris dataset is a built-in toy dataset within SKLearn, making it easily accessible for testing algorithms without external files. It contains measurements for 150 iris flowers across three different species: setosa, versicolor, and virginica.
- Feature Composition: Each sample in the dataset consists of four numerical features: sepal length, sepal width, petal length, and petal width. These features are stored in a NumPy array, allowing for efficient mathematical operations during the training process.
- Target Labels: The dataset includes a target array that maps each sample to an integer representing its specific species. This categorical data is what your machine learning model will attempt to predict based on the physical dimensions.
- Dictionary-Like Structure: When you call load_iris(), the data is returned as a "Bunch" object, which behaves similarly to a Python dictionary. You can access the raw data using keys like .data for features and .target for the classification labels.
- Data Exploration: Because the dataset is small and balanced, it is ideal for visualizing how different classes overlap in feature space. Analysts often use this data to practice scatter plots and pair plots before moving to more complex sets.
- Split the dataset using the Train Test Split function from the SKLearn library.
- Preventing Overfitting: The primary goal of splitting data is to ensure your model generalizes well to new, unseen data. If you train and test on the same data, the model might simply "memorize" the answers (overfitting) rather than learning the underlying patterns.
- The 80/20 Rule: A common convention is to use 80% of the data for training and 20% for testing. For a small dataset like Iris (150 samples), this gives you 120 samples to learn from and 30 samples to validate the model's accuracy.
- Reproducibility with random_state: Since the split is randomized, your results could change every time you run the code. By setting a random_state (e.g., random_state=42), you ensure the split is identical every time, which is vital for debugging and comparing different models.
- Stratification for Balance: Because the Iris dataset is perfectly balanced (50 samples per species), you should use the stratify parameter. This ensures that both your training and testing sets maintain the same proportion of each species as the original dataset, preventing a "biased" test set.

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
