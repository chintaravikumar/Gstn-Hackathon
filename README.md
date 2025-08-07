# Gstn_Hackathon-2024
My Team was able to achieve a **97.58%** accuracy on a predictive model using **Random Forest** training algorithm for the problem statement given for the [GST Hackathon](https://innovateindia.mygov.in/online-challenge-for-developing-a-predictive-model-in-gst/ "About gst hackathon")
<br>Please find the detailed solution in the above python files. Also, there is a python notebook [Google Colab](https://colab.research.google.com/drive/1Dj2ZciIako1es8NtCRhEg2wrr6J_goL8?usp=sharing "Drive") which contains preliminary work done by My Team.<br>
# Get Started
This repository contains code and data analysis for the submission of **Analytics Hackathon on Developing a Predictive Model in GST** by **My team**.
<br>Please follow the below instructions to validate the results of the predictive model.
## Pre-requisites
#### Core Technologies 
1)Python: The primary programming language used in this project.<br>
2)Google colabs: For interactive development and documentation.<br><br>
**Data Processing and Analysis**
<br>1)pandas: For data manipulation and analysis.<br>
2)scikit-learn: For machine learning algorithms and tools.<br><br>
**Visualization**
<br>1)Matplotlib For creating static, animated, and interactive visualizations.<br>
2)PyQt6: For displaying matplotlib graphs in CLI.<br><br>
**Machine Learning Models**<br>
This project utilizes the following machine learning model:<br><br>
Random Forest: For both classification and regression tasks, providing ensemble learning capabilities.
<br><br>Please ensure you have the appropriate versions of these libraries installed before running the project. Refer to the requirements.txt file for specific version information.
## Model Construction
|Aspect|Logistic Regression|Random Forest|
|---|---|---|
|Model Type |<ul><li>Linear model for classification</li></ul><ul><li>Estimates probability of class membership</li></ul>|<ul><li>Ensemble of decision trees</li></ul><ul><li>Combines multiple trees for classification</li></ul>|
|Pros|<ul><li>Simple and interpretable</li></ul><ul><li>Efficient with large datasets</li></ul><ul><li>Performs well with linearly separable classes</li></ul><ul><li>Provides probability scores</li></ul><ul><li>Less prone to overfitting on small datasets</li></ul>|<ul><li>Handles non-linear relationships well</li></ul><ul><li> Robust toutliers and noise</li></ul><ul><li>Provides feature importance rankings</li></ul><ul><li> Reduces overfitting through ensemble learning</li></ul><ul><li>Can capture complex interactions between features</li></ul>|
|Cones|<ul><li>Assumes linear relationship between features and log-adds of the outcome</li></ul><ul><li> May underperform with complex, non-linear relationships</li></ul><ul><li>Sensitive to outliers</li></ul><ul><li>Limited ability to handle feature interactions without explicit engineering</li></ul>|<ul><li>Less interpretable than logistic regression</li></ul><ul><li>Can be computationally expensive for very large datasets</li></ul><ul><li>May overfit on small, noisy datasets if not tuned properly</li></ul><ul><li>Predictions are not as easily probabilistic as logistic regression</li></ul>|
|Use Cases|<ul><li>When interpretability is crucial (e.g.. healthcare, finance)</li></ul><ul><li>Large-scale linear problems</li></ul><ul><li>As a baseline model</li></ul>|*<ul><li> Complex datasets with non-linear relationships</li></ul><ul><li>When feature importance is needed</li></ul><ul><li>Problems where predictive accuracy is more important than interpretability</li></ul>|
## Methodology
Basically methodology means a set of guidelines, rules, and process that help project teams manage and complete projects.<br>
In this, The methodology we follow is...
#### 1. Data PreProcessing
Data preprocessing is a critical step in preparing the raw dataset for analysis. Proper preprocessing ensures data quality, reduces noise, and transforms variables into formats suitable for modeling.<br>
**Step 1:** **Data cleaning**<br>
In this step we clean the given gst datasets. Initially Datasets contains Nan values(Null), so we replace them with mean/average values and filtering irrelevant inforamation.<br> 
**Step 2:** **Encoding Categorical Variables**<br>
Converting categorical features into numerical representations using methods such as One-Hot Encoding or Label Encoding.<br>
**Step 3:** **Normalization/Standardization**<br>
Scaling numerical values to a consistent range (e.g., between 0 and 1) to prevent any particular attribute from disproportionately influencing the model.<br>
**Step 4:** **Feature Engineering**<br>
Creating new features or transforming existing ones to better capture the underlying patterns in the data.
#### 2. Splitting the Dataset
After preprocessing, the dataset is split into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate its performance. This step ensures that the model’s ability to generalize is assessed on unseen data.<br><br>
- *Training Set*: Typically 70-80% of the dataset.<br>
- *Testing Set*: The remaining 20-30% of the dataset.<br>
#### 3.Model Selection and Training
Select appropriate classifiers to train and evaluate. Typical models for classification include:<br>
- *Logistic Regression*: A linear model for binary classification.<br>
- *Decision Tree*: A tree-structured model that splits data based on feature values.<br>
- *Random Forest*: An ensemble of decision trees that reduces overfitting.<br>
- *Support Vector Machine (SVM)*: Finds an optimal hyperplane for separating classes.<br>
- *k-Nearest Neighbors (k-NN)*: Uses distances to the nearest neighbors to classify data.<br>
- *Neural Networks*: A deep learning model for capturing complex patterns.<br>
- *Naive Bayes*: To Predict a target variable based on feature probabilities, assuming independence between features.<br>
Each model is trained on the training data using fit() and then evaluated on the test data using predict().<br>
#### 4.Model Evaluation
Evaluate the performance of each classifier using metrics such as:<br>
- *Accuracy*: The proportion of correctly classified instances.<br>
- *Precision*: The proportion of true positives among the predicted positives.<br>
- *Recall*: The proportion of true positives identified from all actual positives.<br>
- *F1 Score*: The harmonic mean of precision and recall.<br>
- *ROC AUC*: is a metric evaluating binary classification model performance, measuring the model's ability to distinguish between positive and negative classes, ranging from 0 (random guessing) to 1 (perfect classification).
### Performance Comparison
The peformance of these models can vary depending on the specific characteristics of your datasets<br>
1. **Dataset Size**<br>
* Logistic Regression often performs well on large datasets.
* Random Forest can excel with medium sized datasets but might struggle with very large ones due to Computational costs.
2. **Accuracy**<br>
- Logistic Regression: Medium<br>
- Decision Trees: Medium<br>
- Random Forest: High<br>
- SVM: High<br>
- Neural Networks: Very High<br>
- Naive Bayes: Medium<br>
3. **Handling Imbalanced Data**<br>
- Logistic Regression: Good<br>
- Decision Trees: Fair<br>
- Random Forest: Good<br>
- Neural Networks: Excellent<br>
- Naive Bayes: Fair.<br>
## Result
|Prototype|Accuracy|Precision|Recall|F1 Score|
|---|---|---|---|---|
|Logistic Regression|0.968|0.802|0.882|0.8405|
|Decision Tree|0.9673|0.8266|0.8306|0.8286|
|Standard Scaler|0.9688|0.8028|0.8813|0.8403|
|Naive Bayes|0.9589|0.7304|0.8949|0.8043|
|Artificial Neural Networks|0.9756|0.8202|0.9496|0.8802|
|Ada Boost|0.9754|0.8240|0.9418|0.8790|
|Random Forest|0.9758|0.8437|0.9129|0.8769|
## Final Submission
### Modelling Approach
Since we are not given the relationship between the columns and their definitions, it sounds fit to use the **Random Forest Classifier** to train our model. Moreover the results from Random Forest Classifier were far exceeding the models from other techniques.
### Metrics used for evaluation
|Metric |Value |
|---|---|
|Accuracy|0.9758|
|Training Accuracy|0.9975|
|Testing Accuracy|0.9765|
|Precision|0.8437|
|Recall|0.9129|
|F1 Score|0.8769|
|ROC AUC|0.9925|
#### Confusion Matrix
| |Predictive Negative|Predictive Positive|
|---|---|---|
|Actual Negative|21796|409|
|Actual Positive|228|2055|
### Conclusion
Our comparative analysis reveals that **Random Forest** achieved the highest accuracy of **97.58%**, outperforming other algorithms, making it the optimal choice for Predictive Model in GST.
