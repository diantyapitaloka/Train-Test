'## ☂️🌂🌞 Train Test 🌞🌂☂️
- Standard Benchmarking: The Iris dataset is a built-in toy dataset within SKLearn, making it easily accessible for testing algorithms without external files. It contains measurements for 150 iris flowers across three different species: setosa, versicolor, and virginica.
- The 80/20 Rule: When performing a train-test split, practitioners typically reserve 20% to 30% of the data for testing. Since the dataset is small, using a random_state is crucial to ensure your results are reproducible and not just a "lucky" shuffle.
- Feature Composition: Each sample in the dataset consists of four numerical features: sepal length, sepal width, petal length, and petal width. These features are stored in a NumPy array, allowing for efficient mathematical operations during the training process.
- Fixed Dimensions: The dataset has a consistent shape of $(150, 4)$, meaning 150 rows (samples) and 4 columns (features). This predictability makes it perfect for debugging your data pipelines or reshaping logic.
- Target Labels: The dataset includes a target array that maps each sample to an integer representing its specific species. This categorical data is what your machine learning model will attempt to predict based on the physical dimensions.
- Dictionary-Like Structure: When you call load_iris(), the data is returned as a "Bunch" object, which behaves similarly to a Python dictionary. You can access the raw data using keys like .data for features and .target for the classification labels.
- Data Exploration: Because the dataset is small and balanced, it is ideal for visualizing how different classes overlap in feature space. Analysts often use this data to practice scatter plots and pair plots before moving to more complex sets.
- Model Evaluation: Developers frequently use this set to practice splitting data into training and testing subsets using functions like train_test_split. This helps in understanding how to measure accuracy, precision, and recall on a multiclass classification problem.
- Dimensionality Reduction: Because the dataset features four dimensions, it is a popular choice for practicing PCA (Principal Component Analysis) or t-SNE. These techniques allow you to compress the data into a 2D or 3D plot to visually see how distinct the species clusters are.
- Preprocessing Practice: The Iris dataset is excellent for learning about feature scaling and normalization. Since the measurements are all in centimeters and on a similar scale, it provides a safe environment to observe how different scaling methods affect model performance.
- Split the dataset using the Train Test Split function from the SKLearn library.
- Preventing Overfitting: The primary goal of splitting data is to ensure your model generalizes well to new, unseen data. If you train and test on the same data, the model might simply "memorize" the answers (overfitting) rather than learning the underlying patterns.
- The 80/20 Rule: A common convention is to use 80% of the data for training and 20% for testing. For a small dataset like Iris (150 samples), this gives you 120 samples to learn from and 30 samples to validate the model's accuracy.
- Reproducibility with random_state: Since the split is randomized, your results could change every time you run the code. By setting a random_state (e.g., random_state=42), you ensure the split is identical every time, which is vital for debugging and comparing different models.
- Stratification for Balance: Because the Iris dataset is perfectly balanced (50 samples per species), you should use the stratify parameter. This ensures that both your training and testing sets maintain the same proportion of each species as the original dataset, preventing a "biased" test set.
- Shuffle by Default: By default, train_test_split shuffles the data before splitting. This is crucial for Iris because the raw dataset is often ordered by species; without shuffling, your training set might contain only two species, while the test set contains only the third, leading to a total model failure.
- Four-Way Output Unpacking: The train_test_split function returns four distinct subsets in a specific order: X_train, X_test, y_train, and y_test. It is critical to unpack them in this exact order to avoid feeding your labels into the feature processor or vice-versa.
- Feature Scaling Sensitivity: If you plan to use algorithms like SVM or K-Nearest Neighbors, you must split your data before applying scalers (like StandardScaler). You should "fit" the scaler only on the training set to prevent "data leakage," where information from the test set sneaks into the training process.
- The Validation Set Extension: For more complex projects, a two-way split isn't enough. You often split the data into three parts: Training (to build the model), Validation (to tune hyperparameters), and Testing (the final "gold standard" check). With Iris, this might look like a 60/20/20 split.
- Memory Efficiency: While the Iris dataset is tiny, train_test_split creates copies of the data. For massive datasets (gigabytes in size), data scientists often use indices or "generators" to split data without duplicating the memory footprint, ensuring the system doesn't crash.
- Checking the Shape: A mandatory "sanity check" after splitting is to use the .shape attribute on your new arrays. For an 80/20 split on Iris, you should verify that X_train.shape returns (120, 4) and X_test.shape returns (30, 4), ensuring no rows were lost in the process.
- Multiclass Classification: Because there are three species to predict, it is a classic example of a multiclass classification problem rather than a binary one.

```
import sklearn
from sklearn import datasets
```

## ☂️🌂🌞 Load Iris Dataset 🌞🌂☂️
- The sklearn library provides the iris dataset, which is a dataset that is commonly used for classification problems.
- This dataset has 150 samples.
- Three Target Classes: The dataset contains three species of iris flowers: Iris setosa, Iris virginica, and Iris versicolor (50 samples each).
- Historical Significance: Introduced by British biologist Ronald Fisher in 1936, it remains one of the most famous datasets in the history of statistics and machine learning.

```
iris = datasets.load_iris()
```

## ☂️🌂🌞 Separate Attributes and Labels on Dataset Slices 🌞🌂☂️
- The iris dataset from the Sklearn library cannot be directly used by an ML model.
- In accordance with what was discussed in the previous module, we must separate attributes and labels in the dataset.
- No Missing Values: It is a "clean" dataset with no missing data points, making it perfect for beginners to practice model building immediately.
  
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
