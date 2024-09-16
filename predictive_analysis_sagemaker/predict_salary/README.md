# Predicting Employee Salaries Based on Years of Experience: A Case Study
This project illustrates the use of a simple linear regression model to estimate salaries using a fundamental yet powerful statistical technique. In this blog post, we'll walk through the problem, methodology, and outcomes of this analysis.

## Problem Statement
The goal of this case study is to develop a predictive model that estimates an employee's salary based on their years of experience. Specifically, the aim is to quantify the relationship between the number of years an employee has worked and their corresponding salary, using a Simple Linear Regression model.

**Key Details:**
- **Model Type:** Simple Linear Regression
- **Independent Variable (X):** Number of Years of Experience
- **Dependent Variable (Y):** Employee Salary

The Simple Linear Regression model is chosen due to the linear relationship we expect between years of experience and salary. This model will help us understand how salary changes with varying levels of experience.

## Project Setup
**Environment Setup:** To start, the environment is set up to ensure that the Python path is correctly configured to access necessary  odules and datasets.  

**Importing Libraries and Data:** Popular libraries such as pandas, numpy, seaborn, and matplotlib to handle data manipulation, visualization, and modeling.  

The loaded dataset contains employee salaries and years of experience.

## Exploratory Data Analysis (EDA)
Before diving into modeling, an Exploratory Data Analysis is conducted to understand the dataset better:

**Checking for Null Values:** The dataset is clean, with no missing values.  

**Descriptive Statistics:** Provides insights into the distribution of years of experience and salary.

**Distribution Analysis**
- Years of Experience
    - Mean: ~6.31 years
    - Range: 1.1 to 13.5 years
- Salary
    - Mean: ~83,946
    - Range: 37,731 to 139,465

**Observations:**
The data reveals a bimodal distribution in salaries, suggesting that different salary clusters correspond to different levels of experience. 

## Data Visualization
To gain further insights, we visualize the data using scatter plots, histograms, and pair plots:
```python
sns.pairplot(df_salary)
sns.histplot(df_salary["Salary"], bins=30, kde=False)
```
These visualizations help us understand the relationships between experience and salary and identify any outliers or anomalies.

## Local Model Development using `scikit-learn`
### Splitting the Dataset
We split the dataset into training and testing sets to evaluate the performance of our model with 80% of all data for training and the rest 20% for testing.
```python
from sklearn.model_selection import train_test_split

X = df_salary[["YearsExperience"]]
y = df_salary["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
### Training the Model
We use scikit-learn's LinearRegression to train our model:
```python
from sklearn.linear_model import LinearRegression

lin_regress_sklearn = LinearRegression()
lin_regress_sklearn.fit(X_train, y_train)
```

### Model Evaluation
The model's performance is evaluated using the coefficient of determination (R<sup>2</sup>) and the regression line:
```python
lin_regress_sklearn_accuracy = lin_regress_sklearn.score(X_test, y_test)
print(f"Model Accuracy: {lin_regress_sklearn_accuracy}")

# Coefficients
print(f"Coefficient (m): {lin_regress_sklearn.coef_}")
print(f"Intercept (b): {lin_regress_sklearn.intercept_}")
```
### Predictions
Finally, we use the trained model to predict salaries for employees with specific years of experience:
```python
experience_yrs = pd.DataFrame({"YearsExperience": [6, 2]})
salary_yrs = lin_regress_sklearn.predict(experience_yrs)
print(f"Predicted Salaries: {salary_yrs}")
```
### Concluding Remark:
This case study demonstrates how a simple linear regression model can effectively predict employee salaries based on years of experience. The model shows a high accuracy of approximately 89%, indicating a strong linear relationship between the two variables. By understanding this relationship, businesses can make informed decisions regarding salary structures and employee compensation.


## Linear Learner Model Deployment with AWS SageMake
This part demonstrates how to deploy and test a machine learning model using AWS SageMaker, focusing on the Linear Learner algorithm. This section highlights the process of model management, deployment, and inference using AWS cloud services. The project involves the end-to-end process of model deployment, from loading pre-trained artifacts to making predictions and visualizing results.

### Objective
The main goal of this project was to train and deploy a Linear Learner model on AWS SageMaker and perform inference on a test dataset. The project scope also then expanded to encompass the deployment of pre-trained model using saved model artifacts and hence inferencing using deployed model. The steps included:

- **Training the Model:** Number of models is trained in parallel and finally saved the best one.
- **Loading the Saved Model Artifacts:** Retrieving the pre-trained model from an S3 bucket.
- **Deploying the Model:** Setting up an endpoint to facilitate real-time predictions.
- **Making Predictions:** Using the deployed endpoint to generate predictions on new data.
- **Visualizing Results:** Presenting the predictions alongside the actual test data for analysis.
- **Cleaning Up:** Deleting the endpoint to avoid incurring costs for idle resources.

### Methodology
**1. Training the Model:** 
- Converts numpty array to RecordIO format. RecordIO is a data format used by SageMaker to efficiently handle large datasets and is required by some SageMaker algorithms.
- Saves the data to S3 bucket
- A total of 32 models was trained in parallel where one model was trained according to the given training parameter (regularization, optimizer, loss), and the rest by close parameters. Additionally, the `huber_loss` was used for loss calculation during the model training.
- Finally the trained model was saved in a S3 bucket.

**2. Loading the Saved Model:**
The model artifacts were stored in an S3 bucket, which was accessed to load the pre-trained Linear Learner model. The model was initialized using the SageMaker Model class with the specified S3 path and Docker image URI for Linear Learner.

**3. Deploying the Model:**
The model was deployed on an ml.m4.xlarge instance type. The deployment process involved creating a model, endpoint configuration, and the endpoint itself.

**4. Making Predictions:**
Using the Predictor class from SageMaker, the deployed model was used to make predictions on the test dataset. The input data was serialized in CSV format, and the response was deserialized from JSON.

**5. Visualizing Results:**
The predictions were visualized using matplotlib, comparing the predicted values against the actual test data. This step provided insights into the model's performance and accuracy.

**6. Cleaning Up:**
After completing the predictions and analysis, the endpoint was deleted to prevent unnecessary costs.
```python
linear_regressor_deploy.delete_endpoint()
```
### Outcomes  

**Successful Deployment:** The model was successfully deployed and an endpoint was created for inference.  
**Accurate Predictions:** The model provided predictions that were visualized effectively, showing a good fit with the actual data.  
**Cost Management:** The endpoint was properly cleaned up post-analysis, demonstrating good resource management.