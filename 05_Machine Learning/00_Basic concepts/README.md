# **MACHINE LEARNING PART 1**

## **1. Introduction to Machine Learning**

### **1.1 What is Machine Learning?**

- **Machine Learning** is the science (and art) of programming computers so they can **learn from data**.

- Some general definition:
  - **AI**: make machines mimic human behavior
  - **ML**: make machines learn from data
  - **DL**: learn from many data using neural network
  - **Data Science**: make value from data

![AI](https://studyopedia.com/wp-content/uploads/2022/12/Data-Science-VS-Machine-Learning-VS-Artificial-Intelligence-vs-Deep-Learning-Studyopedia.png)

### **1.2 Types of Machine Learning Systems**

#### **1.2.1 Supervised Learning**

- **Supervised learning**: the training data you feed to the algorithm includes the desired solutions, called **labels**.

- The two most common supervised tasks are **regression** and **classification**.

  - **Regression**: the goal is to predict a **continuous** value, such as the price of a car given a set of features (mileage, age, brand, etc.) called **predictors**.

  - **Classification**: the goal is to predict a **discrete** class, such as whether a people is cancer or not based on some attributes (age, heart rate, blood, etc.) called **features**.

- Some examples of supervised learning algorithms:
  - **k-Nearest Neighbors**
  - **Linear Regression**
  - **Logistic Regression**
  - **Support Vector Machines (SVMs)**
  - **Decision Trees and Random Forests**
  - **Neural networks** (some NN are unsupervised)

![Suppervised](https://cdn.labellerr.com/Supervised%20vs.%20Unsupervised%20Learning/supervised1.webp)

#### **1.2.2 Unsupervised Learning**

- **Unsupervised learning**: the training data is **unlabeled**. The system tries to learn without a teacher.

- Some examples of unsupervised learning algorithms:
  - **Clustering**: the algorithm tries to group similar instances together. For example: **k-Means**
  - **Anomaly detection**: the algorithm learns what "normal" data looks like, and can use that to detect abnormal instances, such as bad items on a production line or a new trend in a time series.
  - **Visualization and dimensionality reduction**: the algorithm tries to simplify the data without losing too much information. (This can be called **feature extraction**)
  - **Association rule learning**: the algorithm tries to dig into large amounts of data and discover interesting relations between attributes. For example: recommend system.

![Unsuppervised](https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2020/07/Unsupervised-Learning-in-ML-1.jpg)

#### **1.2.3 Semisupervised Learning**
  
- **Semisupervised learning**: some algorithms can deal with partially labeled training data, usually a lot of unlabeled data and a little bit of labeled data. Most semisupervised learning algorithms are combinations of unsupervised and supervised algorithms.

- For example, Google Photos, are good at clustering together pictures of the same person: this is a **clustering** algorithm, but it first uses a **face recognition** algorithm to label the training data, then it clusters the same face in the unlabeled data.

![Semisuppervised](https://miro.medium.com/v2/resize:fit:1358/1*CzVENZj3bWrwhRBN4hQq7Q.png)

#### **1.2.4 Reinforcement Learning**

- **Reinforcement learning**: the learning system, called an **agent** in this context, can observe the environment, select and perform actions, and get **rewards** in return (or **penalties** in the form of negative rewards). It must then learn by itself what is the best strategy, called a **policy**, to get the most reward over time. A policy defines what action the agent should choose when it is in a given situation.

![Reinforcement](https://www.visteon.com/wp-content/uploads/2016/11/as2.jpg)

### **1.3 Challenges of Machine Learning**

- In short, the only two things that can go wrong are **bad algorithm** and **bad data**.

#### **1.3.1 Insufficient Quantity of Training Data**

- The more data you have, the better. When using a lot of data, it is usually easier to train a more complex model without overfitting than it is with less data.
- For example, to predict the price of a house, we need many features such as: location, size, number of rooms, etc. But if we only have 10 houses, we don't have enough data to train a good model.
- Deep neural networks, which can learn features hierarchically and often generalize well even with little training data.

#### **1.3.2 Nonrepresentative Training Data**

- In order to generalize well, it is crucial that your training data be representative of the new cases you want to generalize to.
- For example, a laptop prediction is implemented in Vietnam, but the training data is collected from USA, so the model will not work well in Vietnam.

#### **1.3.3 Poor-Quality Data**

- If your training data is full of errors, outliers, and noise (e.g., due to poor-quality measurements), your system is less likely to perform well.

- It is often well worth the effort to spend time cleaning up your training data.
  - If some instances are clearly outliers, it may help to simply discard them or try to fix the errors manually.
  - If some instances are missing a few features (e.g., 5% of your customers did not specify their age), you must decide whether you want to ignore this attribute altogether, ignore these instances, fill in the missing values (e.g., with the median age), or train one model with the feature and one model without it, and so on.

#### **1.3.4 Irrelevant Features**

- A system will only be capable of learning if the training data contains enough relevant features and not too many irrelevant ones.
- A critical part of the success of a Machine Learning project is coming up with a good set of features to train on. This process, called **feature engineering**, involves:
  - **Feature selection**: selecting the most useful features to train on among existing features.
  - **Feature extraction**: combining existing features to produce a more useful one (dimensionality reduction algorithms).
  - Creating new features by gathering new data.

#### **1.3.5 Overfitting the Training Data**

- **Overfitting** means that the model performs well on the training data, but it does not generalize well. It is often the result of a model being too complex relative to the amount and noisiness of the training data. The possible solutions are:
  - To simplify the model by selecting one with fewer parameters (e.g., a linear model rather than a high-degree polynomial model), by reducing the number of attributes in the training data or by constraining the model.
  - To gather more training data.
  - To reduce the noise in the training data (e.g., fix data errors and remove outliers).

- Constraining a model to make it simpler and reduce the risk of overfitting is called **regularization**.
- The amount of regularization to apply during learning can be controlled by a **hyperparameter**. A hyperparameter is a parameter of a learning algorithm (not of the model). As such, it is not affected by the learning algorithm itself; it must be set prior to training and remains constant during training.

#### **1.3.6 Underfitting the Training Data**

- **Underfitting** is the opposite of overfitting; it occurs when your model is too simple to learn the underlying structure of the data. The main options to fix this problem are:
  - Selecting a more powerful model, with more parameters.
  - Feeding better features to the learning algorithm (feature engineering).
  - Reducing the constraints on the model (e.g., reducing the regularization hyperparameter).

### **1.4 Testing and Validating**

- To know how well a model will generalize to new cases is to actually try it out on new cases.

- Common option is to split your data into two sets: the training set and the test set. Train your model using the training set, and test it using the test set. By evaluating model on the test set, you get an estimation of error, that tells you how well your model will perform on instances it has never seen before.

- If the training error is low (i.e., your model makes few mistakes on the training set) but the generalization error is high, it means that your model is **overfitting** the training data.

## **2. End-to-end Machine Learning Project**

### **2.1 Prepare data**

- **Get the data**:
  - From open datasets: e.g., [UCI Machine Learning Repository](https://archive.ics.uci.edu/), [Kaggle datasets](https://www.kaggle.com/datasets), [Amazon's AWS datasets](https://registry.opendata.aws/), [Google's Dataset Search](https://datasetsearch.research.google.com/), etc.
  - Crawling the data from Internet: e.g. Laptop price from Amazon.
  - Collect it from people (e.g., MITBIH Arrhythmia Database collect from people wearing ECG sensors).

- **Take a quick look at the data structure**:
  - Use some functions from Pandas, e.g., head(), info(), describe(), value_counts(), etc.
  - Use some libraries to plot the data, e.g., matplotlib, seaborn, etc.

- **Split data**: for ussualy, the data is split into 3 sets: training set, validation set, and test set. The test set is used to test the model after training, the validation set is used to validate the model during training, and the training set is used to train the model.
  - **Random sampling**: the simplest option is to use Scikit-Learn's `train_test_split()` function to split the dataset randomly.
  - **Stratified sampling**: is a way to split the dataset in order to make the distribution of the data in the training set and the test set is similar to the distribution of the data in the whole dataset.
  - **Random seed**: if you set the random seed (e.g., `np.random.seed(42)`) in order to make the output stable across multiple runs.

- **Visualize the data to gain insights**: use some libraries to plot the data, e.g., matplotlib, seaborn, etc. to gain the insights of data.

- **Look for correlations**: use some functions to look for correlations, e.g., corr(), scatter_matrix(), etc. to view the standard correlation coefficient (also called Pearson's r) between every pair of attributes.

- **Attribute Combinations**: we can create new attributes by combining existing attributes in order to get more useful attributes.

#### **2.1.1 Data cleaning**

- **Missing values**:
  - Most Machine Learning algorithms cannot work with missing features, so let's create a few functions to take care of them.
  - We have three options:
    - Drop missing value -> using `dropna()`
    - Drop the whole attribute -> using `drop()`
    - Fill missing values with some values (zero, the mean, the median, etc.) -> using `fillna()` or some imputer class in Scikit-Learn
  - With categorical attributes, most of the time we just fill missing values with the most frequent value.

- **Catrgorical data**: ussually divide into 2 types: nominal data and ordinal data.
  - **Nominal data** is just a set of categories, e.g., "A", "B", "C", "D", or "red", "green", "blue", country name, etc. The order does not matter.
  - **Ordinal data** is a set of categories that have some order, e.g., "bad", "average", "good", "excellent", or "C", "B", "A+", etc. The order matters.

- **Handling text and categorical attributes**:
  - **OrdinalEncoder**: convert each categorical value to a different integer, start from 0, e.g., "red" -> 0, "green" -> 1, "blue" -> 2, etc.
  - **OneHotEncoder**: create one binary attribute per category (a one-hot vector), e.g., "red" -> (1, 0, 0), "green" -> (0, 1, 0), "blue" -> (0, 0, 1), etc.
  - **Hash encoding**: convert each categorical value to a hash value, that will never change with the same input value.
  - **Word embedding**: convert each categorical value to a vector, this vector can present the relationship between each categorical value, e.g., "red" -> (1, 0, 0), "green" -> (0, 1, 0), "blue" -> (0, 0, 1), "yellow" -> (0.5, 0.5, 0), etc.

- **Feature scaling**: model won't perform well when the input numerical attributes have very different scales. There are two common ways to get all attributes to have the same scale:
  - **Min-max scaling** (many call this normalization): values are shifted and rescaled so that they end up ranging from 0 to 1. Scikit-Learn provides a transformer called `MinMaxScaler` for this. The equation is: $`x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}`$
  - **Standardization**: the values are centered around 0 and have a unit variance, that mean Mean of data is 0 and standard deviation is 1. Standardization is much less affected by outliers. Scikit-Learn provides a transformer called `StandardScaler` for standardization. The equation is: $`x_{std} = \frac{x - \mu}{\sigma}`$ where $`\mu = \frac{sum(x)}{count(x)}`$ is the mean of the training set and $`\sigma = \frac{\sqrt{sum((x - mean)^{2})}}{count(x)}`$ is the standard deviation of data.

#### **2.1.2 Transformation pipelines**

- **Pipeline**: to ensure that all the transformations are performed in the right order, Scikit-Learn provides the provides the `Pipeline` class to help with such sequences of transformations. For example:

```python
from sklearn.pipeline import Pipeline

# Get the list of numerical and categorical columns
cat_names = list(X_train.select_dtypes('object').columns)
num_names = list(X_train.select_dtypes(['float', 'int']).columns)

# Pipeline for categorical data
cat_pl= Pipeline(
    steps=[
      # Handle missing data by filling with the most frequent value
      ('imputer', SimpleImputer(strategy='most_frequent')),
      # Encode categorical data by OneHotEncoder
      ('onehot', OneHotEncoder()),
    ]
)

# Pipeline for numerical data
num_pl = Pipeline(
    steps=[
      # Handle missing data by filling with the mean value
      ('imputer', SimpleImputer(strategy='mean')),
      # Scale data by StandardScaler
      ('scaler', StandardScaler()),
    ]
)

from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pl, num_names), # apply for numerical data
        ('cat', cat_pl, cat_names), # apply for categorical data
    ]
)

# Completed training pipeline
completed_pl = Pipeline(
    steps=[
            ("preprocessor", preprocessor), 
            ("classifier", RandomForestClassifier())
    ]
)

# training
completed_pl.fit(X_train, y_train)

# accuracy
y_train_pred = completed_pl.predict(X_train)
print(f"Accuracy on train: {accuracy_score(list(y_train), list(y_train_pred)):.2f}")

y_pred = completed_pl.predict(X_test)
print(f"Accuracy on test: {accuracy_score(list(y_test), list(y_pred)):.2f}")
```

### **2.3 Visualize Data**

- Use some libraries to plot the data, e.g., matplotlib, seaborn, etc. to gain the insights of data.
- Some common plots:
  - Histograms
  - Box plots
  - Scatter plots
  - Bar/Pie/Line plots
  - Correlation matrix
  - ...

### **2.4 Select & train model**

- **Training and evaluating on the training set**: Import the model from Scikit-Learn, e.g., `from sklearn.linear_model import LinearRegression` then fit the model to the training data using the `fit()` method.

- **Cross-Validation**: help us to evaluate the model by split the training data into smaller training set and validation set. There are some ways to split the data:
  - **K-fold cross-validation**: split the training set into K-folds (e.g., K = 5 or K = 10), K-1 folds for training and 1 fold for validation. Then train the model K times and evaluate it on each of the K folds using the model trained on the remaining folds. The result is an array containing the K evaluation scores.
  - **Stratified K-fold cross-validation**: the folds are made by preserving the percentage of samples for each class.

- That why Cross-Validation evaluate more objectively and accurately.

![Cross-validation](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)

### **2.5 Fine-tune model**

- **Model parameters** are the configuration variable of the model estimated from the training data that help to show the relationship between the quantities in the data. For example, the coefficients of the `Linear Regression` model.

- **Hypperparameter** is a parameter of a learning algorithm (**not of the model**). It is set prior to the start of the learning process and remains constant throughout the learning process. For example, number of hidden layers in neural network or `C` and `gamma` of the `Support Vector Machine` algorithm are hyperparameters.

#### **2.5.1. Grid Search**

- **Grid Search** is a technique for finding the optimal hyperparameters for a model.
- It works by exhaustively searching through a set of predefined hyperparameters and evaluating the model for each combination of hyperparameters.
  - You define a set of hyperparameters and their possible values.
  - Grid Search will train your model for each possible combination of these hyperparameters.
  - For each combination of hyperparameters, Grid Search evaluates your model using cross-validation. It calculates a metric like accuracy or loss and chooses the hyperparameters that resulted in the best performance based on the evaluation metric. Those hyperparameters are then considered optimal for your model.
  
```python
from sklearn.model_selection import GridSearchCV
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}
]
model = RandomForestRegressor()
# train across 5 folds, that's a total of 12*5=60 rounds of training
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```

#### **2.5.2. Randomized Search**

- **Randomized search** is an optimization technique for hyperparameter tuning that aims to overcome some of the drawbacks of grid search. It works as follows:
  - Like grid search, you define a search space of possible hyperparameters and their value ranges.
  - Randomized search then samples random hyperparameters from this search space.
  - It trains and evaluates your model using these randomly chosen hyperparameters. It calculates a performance metric like accuracy or loss.
  - It repeats steps 2 and 3 for a predefined number of iterations (e.g. 100 iterations). It records the best performing set of hyperparameters found so far. At the end, it returns the hyperparameters that resulted in the best performance during the search process.

#### **2.5.3. Ensemble Methods**

- Another way to fine-tune system is to try to combine the models that perform best. The group (or "ensemble") will often perform better than the best individual model (just like Random Forests perform better than the individual Decision Trees they rely on), especially if the individual models make very different types of errors.

#### **2.5.4. Analyze the best models and their errors**

- You will often gain good insights on the problem by inspecting the best models. For example, the `RandomForestRegressor` can indicate the relative importance of each attribute for making accurate predictions:

```python
feature_importances = grid_search.best_estimator_.feature_importances_
```

- With this information, you may want to try dropping some of the less useful features before training.

#### **2.5.5. Evaluate your system on the test set**

- After tweaking your models for a while, we evaluate the final model on the test set. We get the predictors and the labels from your test set, run your `full_pipeline` to transform the data (call `transform()`, not `fit_transform()`—you do not want to fit the test set!), and evaluate the final model on the test set (call `predict()` and `evaluate()`).

- With the result from test set, we can calculate some metrics to evaluate the model, e.g., RMSE, MAE, etc.

### **2.6 Launch, monitor, and maintain your system**

- After developing the model, we need to deploy it to production environment. There are some ways to do that:
  - Save the model and load it in the production environment.
  - Use the cloud service to deploy the model.
  - ...

- **MLOps (Machine Learning Operations)** is practices and tools used to deploy machine learning models into production and maintain them over time.

![MLOps](https://images.viblo.asia/2067b594-4731-4074-80ab-13a537e3d054.png)

## **3. Classification**

### **3.1. Definition**

- Classification is a type of machine learning where the model learns from labeled training data to accurately predict the class or category of new data.

### **3.2 Performance measures**

#### **3.2.1 Confusion matrix**

- **Confusion matrix**: a matrix that count the number of times instances of class A are classified correct as class A and the number of times instances of class A are classified incorrectly as class B. This is done for all classes.

![TP, FP, TN, FN](https://github.com/thangnch/photos/blob/master/Screen%20Shot%202020-06-16%20at%2014.00.30.png?raw=true&a=1)

#### **3.2.2 Precision and recall**

![Precision and recall](https://www.digital-mr.com/media/cache/5e/b4/5eb4dbc50024c306e5f707736fd79c1e.png)

- **Precision**: is the accuracy of positive predictions. It answers the question: "Of all the items labeled as belonging to the positive class, what percentage did the classifier actually classify correctly?"

```math
Precision = \frac{TP}{TP + FP}
```

- **Recall**: is the ratio of positive instance that are correctly detected. It answers the question: "Of all the items that truly belong to the positive class, what percentage did the classifier classify correctly?"

```math
Recall = \frac{TP}{TP + FN}
```

- **Precision/Recall Tradeoff**: unfortunately, you can't have it both ways: increasing precision reduces recall, and vice versa. This is called the precision/recall tradeoff.

- Ussually, the classifier computes a score based on a decision function, and if that score is greater than a threshold, it assigns the instance to the positive class, or else it assigns it to the negative class. The threshold is a parameter that we can set to increase or decrease precision and recall.
  - Increasing the threshold increases precision and reduces recall.
  - Decreasing the threshold increases recall and reduces precision.

- Base on the equation of precision and recall, we can see that if we want to increase precision, we need to decrease the number of false positives. However, this also increase number of false negative and reduces the recall.

![Precision/Recall Tradeoff](https://miro.medium.com/v2/resize:fit:640/format:webp/1*R_eSIAc-1wjArHbACy1z9w.png)

#### **3.2.3. ROC**

- Two new definitions:
  - **True Positive Rate (TPR)**: is the ratio of positive instances that are correctly detected by the classifier. It is also called **recall** or **sensitivity**.
  - **False Positive Rate (FPR)**: is the ratio of negative instances that are incorrectly detected as positive.

```math
\large True Positive Rate = \frac{True Positive}{True Positive+False Negative}
```
  
```math
\large False Positive Rate = \frac{False Positive}{False Positive+True Negative}
```

- `The receiver operating characteristic (ROC) curve` showing the relationship between the true positive rate (TPR) and the false positive rate (FPR) for every possible threshold value. The dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).

- **AUC (Area Under the Curve)**: is the area under the ROC curve. A perfect classifier will have a ROC AUC equal to 1, whereas a purely random classifier will have a ROC AUC equal to 0.5. Better model has higher AUC.

![ROC](https://i0.wp.com/sefiks.com/wp-content/uploads/2020/12/roc-curve-original.png?fit=726%2C576&ssl=1)

### **3.3 Multiclass classification**

- Wheareas binary classifiers distinguish between two classes, multiclass classifiers (also called multinomial classifiers) can distinguish between more than two classes.

- Some algorithms (such as Random Forest classifiers or naive Bayes classifiers) are capable of handling multiple classes directly.

- Others (such as Support Vector Machine classifiers or Linear classifiers) are strictly binary classifiers. However, there are various strategies that you can use to perform multiclass classification using multiple binary classifiers

- **One-versus-the-rest (OvR)** strategy (also called one-versus-all): train a binary classifier for each class. When you want to classify an instance, you get the decision score from each classifier for that instance and you select the class whose classifier outputs the highest score.

- **One-versus-one (OvO)** strategy: train a binary classifier for every pair of classes. If there are N classes, you need to train N × (N – 1) / 2 classifiers. When you want to classify an instance, you have to run all N × (N – 1) / 2 classifiers and see which class wins the most duels. The main advantage of OvO is that each classifier only needs to be trained on the part of the training set for the two classes that it must distinguish.
