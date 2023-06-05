# Predicting Cardiovascular Disease in Framington, MA. 

The World Health Organization has estimated that there are approximately 12 million deaths annually related to heart diseases, with half of the deaths in the United States and other developed countries attributed to cardiovascular diseases. The early prognosis of cardiovascular diseases can aid in decision making realted to lifestyle changes in high-risk patients and reduce the complications of heart diease. The intent of this research is to determine the most relevant risk factors of heart disease and forecast the overall risk using machine learning and predictive modeling.

## Introduction to the data
The dataset is publically available on the Kaggle website: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset, and is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts. The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).The dataset provides the patients’ information, includes over 4,200 records and 16 of the following attributes:

•	male: 0 = Female; 1 = Male

•	age: Age at exam time.

•	education: 1 = Some High School; 2 = High School or GED; 3 = Some College or Vocational School; 4 = College

•	currentSmoker: 0 = nonsmoker; 1 = smoker

•	cigsPerDay: number of cigarettes smoked per day (estimated average)

•	BPMeds: 0 = Not on Blood Pressure medications; 1 = Is on Blood Pressure medications

•	prevalentStroke: 0 = Stroke not prevalent in family history; 1 = Stroke prevalent in family history

•	prevalentHyp: 0 = Hypertension not prevalent in family history; 1 = Hypertension prevalent in family history

•	diabetes: 0 = No; 1 = Yes

•	totChol: total cholesterol (mg/dL)

•	sysBP: systolic blood pressure (mmHg)

•	diaBP: diastolic blood pressure (mmHg)

•	BMI: BodyMass Index calculated as: Weight (kg) / Height(meter-squared)

•	heartRate Beats/Min (Ventricular)

•	glucose: total glucose mg/dL

•	TenYearCHD: 0 = Patient doesn’t have 10-year risk of future coronary heart disease; 1 = Patient has 10-year risk of future coronary heart disease

## Methods of Investigation
The project involves several key steps. First, the dataset was preprocessed to handle missing values and convert categorical variables into numerical representations. Data exploration and visualization techniques were then used to gain insights into the distribution of features and their relationships with the target variable, “TenYearCHD”. We then used feature selection methods to identify the most relevant features for heart disease prediction.

Next, the dataset was split into training and testing sets, and a variety of machine learning algorithms were considered for model selection, including PCA Method, Decision Tree Modeling, Random Forest, K-Neighbors Model, and Keras Neural network. The models were trained on the training set and evaluated using metrics such as accuracy, precision, recall and F1-score. The trained models were then interpreted to understand the significant features contributing to heart disease prediction. Finally, the best-performing model can be deployed for real-time heart disease prediction on new, unseen data.

## Predictive Modeling
Supervised Machine Learning-

First we ran the Decision tree model against all features of our dataset.


![image](https://github.com/Gorbulin1989/Project-4/blob/main/Supervised%20Methods%20-%20DTM.png)

Provided the low accuracy from the DT model, we decided to build a Random Forest model again using all features, and looking into the order of importance of these features.


![image](https://github.com/Gorbulin1989/Project-4/blob/main/RF%20model.PNG)

Next we are interested in seeing how a model would perform dropping the 4 least important features, namely; ‘Prevalent_Stroke’, ‘Diabetes’, ‘Blood_Pressure_Medications’ and ‘Current_Smoker’ And build a K-nearest Neighbor (KNN) and Keras Neural Network (Keras NN) models.
Neither of these 2 models yielded an accuracy higher than 84% (RF accuracy using all features)
Therefore we put those all 4 features back in the dataset and re-run the models.

1) KNN model-


![image](https://github.com/Gorbulin1989/Project-4/blob/main/Supervised%20Methods%20-%20KNN.png)

2) Keras NN model-
Initially, we start with only 1 hidden layer, but seen the promising performance of this model (84.69% accuracy), that prompted us to use the hyperparameter auto-optimizer to tune up the model. In doing so we were able to reach 86% in the end, after increasing epochs from 20 to 30 and then 50.


Subsequently, we tried to optimized our accuracy by trying to get a better training data, using Random OverSampler;


 In the end the accuracy did not improve.

Principal Component Analysis-

Another thing, we explored is Principal Component Analysis. With the 15 features we have in our dataset, we examined if there is a plausible cause to group some features.

![image](https://github.com/Gorbulin1989/Project-4/blob/main/Unsupervised%20Methods%20-%20PCA.png)

As we can see from the image above, it may be worth while running our models against PCA.
We got the following:

1) RF model with PCA

![image](https://github.com/Gorbulin1989/Project-4/blob/main/Supervised%20Methods%20-%20RF%20-%20PCA.png)

2) K-NN model with PCA

![image](https://github.com/Gorbulin1989/Project-4/blob/main/Supervised%20Methods%20-%20KNN%20-%20PCA.png)

3) Keras NN with PCA



## Conclusion

Below illustrates the final accuracy of our models, with Keras NN having the highest degree of accuracy for prediction.
![image](https://github.com/Gorbulin1989/Project-4/blob/main/Model%20Accuracy.png)



