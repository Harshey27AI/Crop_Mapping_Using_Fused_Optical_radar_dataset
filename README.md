# Crop_Mapping_Using_Fused_Optical_radar_dataset
Implement neural network and one other classification algorithm and compare the performance for the dataset you choose.  Apply (PCA) to the dataset.  Apply any feature selection method .






PART B

DATA ANALYTICS

PART 1:

7.0 DEFINING TRAINING AND TESTING DATA

In machine learning applications, for "Crop mapping using fused optical-radar dataset", defining training and testing data is a crucial step. Machine learning models are trained using training data, and their effectiveness is assessed using testing data.

7.1. Training Data: 

Examples that are typical of the various crop classes prevalent in the research region should make up the training data. For each sample, the training dataset should contain both optical and radar data. The appropriate ground truth or reference data identifying the proper crop type for each sample should also be labeled on the data.

7.2. Testing Data: 

We made a  choice to select  the Crop mapping using fused optical-radar dataset combines information from optical and radar sensors to accurately identify and map different types of crops, in this case dataset consist of 175 features and 1 label column which contains 7 classes i.e 'Corn', 'Pea', 'Canola', 'Soy', 'Oat', 'Wheat', 'Broadleaf' which are different types of crops.
175 features includes:

1.	Class. 
2.	f1 to f49: Polarimetric features.
3.	f50 to f98: Polarimetric features.
4.	f99 to f136: Optical features.
5.	f137 to f174: Optical features.

This Large data collection consists of bi-temporal optical-radar data that has been merged for agricultural categorization. RapidEye satellites (optical) and the Unmanned Aerial Vehicle Synthetic Aperture Radar (UAVSAR) system (Radar) gathered the photos in 2012 over an agricultural region near Winnipeg, Manitoba, Canada, there are 2 * 49 radar features and 2 * 38 optical features.








7.3. Data Preprocessing:

The complete list of variables of each feature vector shows a total of 561 attributes. The given dataset is split into: 
1. 80% Training Set
2. 20% Test Set 

For the Decision Tree classification:

1.	Feature selection: Highly intercorrelated features have been eliminated from the dataset.

2.	Splitting the data:
●	Training set: 80% of the data will be allocated for training the Decision Tree model.
●	Testing set: 20% of the data will be reserved for evaluating the model's performance.

For the Multi-Layer Perceptron (MLP) classification, the steps mentioned earlier can be modified as follows:

●	Feature selection: Eliminate highly intercorrelated features from the dataset.
●	Label tensor:
i.	MLP Classification: Encode the label column using one-hot encoding with the help of Pandas' get_dummies method. This will transform the labels into arrays with seven binary elements, representing each crop class.
●	Data splitting:
i.	Training set: Assign 80% of the data for training the MLP model.
ii.	Testing set: Keep 20% of the data for evaluating the model's performance.
iii.	The split proportions are chosen arbitrarily.









 
7.4. Data Analysis & Visualization:

In this part of Data Analysis & Visualization we used Jupyter Notebook which is  a web-based interactive computing platform based on python language. During analysis we found different relationships in terms of correlation between classes.
























Fig 1.1 Class Representation

As mentioned, there are altogether 7 types of labeled classes which are 'Corn', 'Pea', 'Canola', 'Soy', 'Oat', 'Wheat', 'Broadleaf they are basically different types of crops.

7.5. Machine Learning Approach:

There are several approaches in machine learning (ML) that can be used to solve problems. These approaches include classical learning (supervised and unsupervised), reinforcement learning, ensemble methods, and neural networks and deep learning (DL).

●	Classical Learning (Supervised Learning & Unsupervised Learning)
●	Reinforcement Learning
●	Ensemble Methods
●	Neural Networks and Deep Learning (DL)

Artificial Intelligence and Machine Learning (AI/ML) is a broad field that is constantly developing. Given its broad scope, it is difficult to cover every aspect of it in a single report. We have thus gone to several literature sources that are in the public domain in order to give insights into the present state-of-the-art models, classifications, and learning methodologies. (Fig 1.4)


 

Fig 1.2 ML Model and Types (Researchgate)


We performed a preliminary study utilizing the Convolution Neural Network (CNN) in accordance with the need to investigate a neural network and another classification technique. On our training and testing datasets, we also applied the classification algorithm Decision Tree Using this method, we were able to examine the underlying powers of these algorithms and collect initial data for subsequent investigation.











PART 2: IMPLEMENTATION OF NEURAL & CLASSIFICATION ALORITHM

Background: 
We performed this experimental comparison on the loaded dataset using 
1. Full training set i.e. all 174 features
2. No Feature Selection Algorithm.

Implementing Machine Learning Model:

Our Chosen Classification and Neural Algorithms: 

7.6.1. Neural Network - Multilayer Perceptron (MLP): 
A multilayer perceptron (MLP) is a type of artificial neural network that mimics the human brain. It consists of layers of nodes or neurons that process information. Each node takes inputs from the previous layer, applies weights to them, and passes the result through an activation function to produce an output. This process continues through multiple layers until a final output is obtained. MLPs can be trained using a technique called backpropagation, which adjusts the network weights to minimize errors. MLPs are commonly used for numeric data, but they can also process image inputs by flattening the pixel values. The performance of an MLP improves with more training data, as it can learn better patterns and make more accurate predictions (Khoshgoftaar et al., 2010).




















		
Fig 1.3 Multilayer Layer Perceptron(Medium) 
Training Outcome for Multilayer Perceptron Model:

Accuracy of Model		
●	Accuracy: 99.77 %					
●	Precision: 99.22 %  			
●	Recall: 99.54 % 			
●	F-Score: 99.38 % 




						
















Fig 1.4 Training Outcome for Multilayer Perceptron Model

Fig 1.5 Denotes Accuracy using Multilayer Perceptron Modeling (MLP) model was implemented successfully with accuracy of 99.77% and Precision of 99.22% which are a great result using MLP later on we calculated Accuracy per class i.e., different crops available in dataset.
Accuracy calculated for per class seems to be interesting results Corn: 99.92 %, Pea: 98.77 %, Canola: 99.92 %, Soy: 99.89 %, Oat: 99.58 %, Wheat: 99.79 %, Broadleaf: 96.86 % all classes were predicted with overall good accuracy. The following is a confusion matrix which represents different classes present.











Fig 1.5 Confusion Matrix (MLP)

7.6.2. Classification Algorithm - Decision Tree:

In ML, the DT algorithm is a popular classification method. It builds a model that resembles a tree, with each internal node standing in for a feature or attribute and each leaf node standing in for a class label or projected value. To arrive at judgements and build branches for the tree, the algorithm iteratively divides the data into groups according to several qualities.
Decision trees provide the benefit of interpretability since they are simple to understand and visualize. They can manage category and numerical data and capture complicated connections. By allocating instances to the most prevalent class or value in each node, decision trees may also manage missing data. 
(Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and Regression Trees. CRC Press)

Training Outcome for Decision Tree:

Accuracy of Model:

●	Accuracy: 98.90 %
●	Precision: 98.29 %
●	Recall: 97.63 %
●	F-Score: 97.95 %




















Fig 1.6 Accuracy Decision Tree

As shown in Fig 1.6 Denotes Accuracy using Decision Tree(DT), model was implemented successfully with accuracy of 98.90%  and Precision of 98.29% which is a great result using DT later on we calculated Accuracy per class i.e. labeled class Accuracy calculated for each class seems to be quite accurate results Corn: 98.91 %, Pea: 98.91 %, Canola: 99.65 %, Soy: 99.13 %, Oat: 97.63 %, Wheat: 98.73 %, Broadleaf: 95.03 % all classes where predicted with overall good accuracy. The result was slightly lower as compared to MLP model. The following is a confusion matrix which represents different classes present.

 

Fig 1.7 Confusion Matrix DT


The above result visualizes the confusion matrix for the predictions made by a decision tree classifier. The confusion matrix is a useful tool for evaluating the performance of a classifier by showing the counts of true positive, true negative, false positive, and false negative predictions for each class.











7.6.3 Comparative Evaluation of Performance Results of Multilayer Perceptron Model & Decision Tree 

The accuracy outcomes of the three models are tabulated below and a corresponding visualization of the performance differentials is shown in figure 2.0. Table 1. Model Accuracy Performance Comparison.

MODEL	ACCURACY	RECALL	PRECISION	F1-SCORE
Multilayer Perceptron (MLP)	99.77%	99.54%	99.22%	99.38%
Decision Tree (DT)	98.90%	97.63%	98.29%	97.95%
     
Table 1.8 Comparative Evaluation of Performance Results  

Fig 1.9 Comparison DT and MLP
Hence, we found out that Multilayer Perceptron Model performs better than the classification algorithm i.e. Decision Tree where the accuracy i.e. overall correctness of the model’s predictions are higher in the case of MLP compared to DT, The proportion of correctly predicted positive instance is higher in MLP in this case also MLP wins.



PART 3:

8. PRINCIPAL COMPONENT ANALYSIS (PCA):

Reduce the number of variables in datasets (particularly huge datasets) while retaining as much meaningful information as you can is the underlying tenet of PCA. Although PCA sacrifices some accuracy for simplicity, it removes the need for extraneous variables, which simplifies data processing for machine learning algorithms.
Generalization:
The capacity of a learning system to perform effectively on a novel, untrained data is known as generalization, and it is a key component of machine learning. In order to produce correct predictions, a machine learning model must be able to learn patterns from the training data and apply them to fresh, unexplored data. This is known as excellent generalisation performance.

Holdout method:
 

Fig 2.0 Split Data

The train_size variable is set to 80% of the length of the X data frame, and the iloc method is used to select the first train_size rows for the training set and the remaining rows for the testing set. The values method is used to convert the data frames to NumPy arrays.
After splitting the dataset, the training data is used to fit the decision tree classifier, and the testing data is used to evaluate the performance of the model. The holdout method is a common way to evaluate the performance of a machine learning model, and it involves splitting the dataset into training and testing sets in order to simulate how the model will perform on new, unseen data.

Cross Validation:
Here we in this code dropped highly correlated features from a dataset. The first line identifies the features that have a correlation coefficient greater than 0.95 with any other feature, using the upper_matrix DataFrame that contains the upper triangle of the correlation matrix. The second line then drops these features from the original dataset. This is done to remove redundancy in the data and prevent multicollinearity, which can affect the performance of certain machine learning algorithms. By removing highly correlated features, we can reduce the dimensionality of the dataset and potentially improve the accuracy and efficiency of our models.

 

Fig.2.1 Cross Validation

These above scores indicate the accuracy of the model on each fold of the cross-validation process. It is common to report these scores to assess the consistency of the model's performance across different subsets of the data. The average of these scores shows an overall estimate of the model's performance during cross-validation.

Implementation of Principal Component Analysis(PCA):

 

Fig 2.2 Implementation of PCA



Impact of PCA on the percentage of variance:

Fig 2.3 displays the eigenvalues and the proportion of variance covered by each principal component. PC1 has a proportion of 0.19, which means it can explain 19% of the variance of the dataset. That is the summation of PC1, PC2…PC10  cover 66% approx percentage of the variance. Fig 2.3: Eigenvalue and Proportion of the first 3 Principal Components.

 

Fig 2.3 Eigenvalue and Proportion of the First 10 Principal Components 


 
Fig 7.4 Eigenvalue and Proportion of the First 10 Principal Components





Decision Tree Performance After Applying PCA 
As illustrated in Figures 7.5 and 7.6, the accuracy results are as stated below. Training Accuracy (post-PCA) 
• Using Full Training Set: 99.99% 
• Using 10-fold Cross-Validation: 97.04%

 

Fig 7.5 Accuracy Post PCA

 
	Fig 7.6 Accuracy Comparison Post and Pre PCA on Decision Tree 

PART 4:

9.0 ALGORITHM PERFORMANCE EVALUATION BEFORE AND AFTER APPLYING FEATURE SELECTION

9.1 Feature Selection Methods 
According to Guyon and Elisseeff (2003), feature selection is a crucial step in machine learning and data analysis, where the goal is to select a subset of relevant features from a larger set of features that are available in a dataset. The purpose of feature selection is to reduce the dimensionality of the dataset, remove irrelevant or redundant features, and improve the accuracy and efficiency of the machine learning models. Several feature selection methods have been proposed in the literature in this context, each with its advantages and limitations.

There are several methods available for feature selection, including filter methods, wrapper methods, embedded methods, and hybrid methods (Saeys et al., 2007):-


•	Filter methods use statistical measures to rank the features and select the top-ranked ones, 
•	wrapper methods select features by training and evaluating a machine learning model repeatedly on different subsets of features. 

•	Embedded methods select features during the training of the machine learning model itself, and hybrid methods combine two or more feature selection methods to improve the accuracy and efficiency of the machine learning models (Li et al., 2017).

The choice of the feature selection method depends on the characteristics of the dataset and the goals of the analysis (Dash and Liu, 1997). Each of these methods has its advantages and limitations, and it is important to choose the most appropriate method for the particular application. So we choose the Filter method in which comes correlation.

9.1.1 Correlation:
The correlation-based feature selection (CFS) method is a filter approach, it is unaffected by the final classification model. It solely considers feature subsets depending on the data's intrinsic qualities.

Correlation is a statistical measure of the relationship between two variables, indicating how much they tend to vary together. In the field of statistics, correlation plays a significant role in analysing and interpreting data, particularly in identifying patterns and relationships that can help in making predictions and informed decisions. According to Howell (2002), correlation
coefficients can range from -1 to 1, with negative values indicating an inverse relationship between variables and positive values indicating a direct relationship. 
 
				Figure 7.7- Correlation (Mcleod, S. (2022, November 3)
Finding Correlation: here we need to find a correlation between different variables it measures the extent to which two variables are related and the direction and strength of that relationship. Correlation may be used to find characteristics that have a high degree of relationship to the target variable in the context of feature selection. A feature may be a good predictor of the goal and hence a viable candidate for inclusion in the model if it has a high correlation with the target variable. On the other hand, a feature may not be a strong predictor and can be dropped from the model if it has a weak correlation with the target variable.

 
				Figure 7.8- Finding Correlation
Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable. So, when two features have a high correlation, we can drop one of the two features.

 
				Figure 7.9 Dropping Features

After eliminating strongly linked characteristics, the number of features decreased from 174 to 102. As a result, more efficient models are produced that use less processing power and memory, lowering collinearity and enhancing machine learning performance. a correlation chart with the dependent variable (the "label") included



9.2 NEURAL NETWORK (Multilayer Perceptron) With Feature Selection

 
Figure 8.0 (MLP) Neural Network
 
This suggests that during the feature selection process, the model's Accuracy and Precision were increased somewhat by 0.04% and 0.03%  but its F-score, and Recall fell little. The selected subset of features was able to capture more of the relevant information in the dataset, resulting in a slight improvement in the model's ability to correctly classify instances.

The F-score is a measure of the model's balance between precision and recall, while recall measures the model's ability to correctly identify positive instances. A decrease in F-score and recall means that the model is now less effective at correctly ident a selected subset of features
and was able to capture more of the relevant information in the dataset, resulting in a slight improvement in the model's ability to correctly classify instances and satisfy positive instances.

 
			Figure 8.1 Confusion matrix MLP(Feature Selection)

A confusion matrix is shown in the table, and it is used to assess the effectiveness of a classification model. The actual classes are represented by the rows of the matrix, while the anticipated classes are represented by the columns.

The first row, for example, shows that there were 7733 instances of the "Corn" class in the dataset, and the model correctly predicted 7733 of these as "Corn." There were three cases of "Pea" misclassified as "Corn," and seven occurrences of "Soy" misclassified as "Corn."

Similarly, the second row shows that the dataset had 722 occurrences of the "Pea" class, and the model correctly predicted all 722 cases as "Pea." There were no instances of "Pea" being misclassified as a different class.


 
9.3 Performance Comparison of MultiLayer Perceptron(MLP) Before and After Applying Feature Selection

We performed feature selection on MLP by following the steps listed earlier.
  
			Figure 8.2 Outcome Accuracy of MLP Before Feature Selection 
 
			Figure 8.3 Outcome Accuracy of MLP After Feature Selection 

The Accuracy Outcome of MLP and Decision Tree are:

                Method 	Before Feature Selection	AfterFeature Selection
                  MLP	99.77%	99.81%

			


       
        Figure 8.4 Comparison Chart for MLP 

The feature selection process has had a positive impact on the performance of the model. The accuracy has improved from 99.77% to 99.81% and recall has also increased from 99.54% to 99.47%. However, the precision has slightly decreased from 99.22% to 99.25% and the F-score has also seen a slight dip from 99.38% to 99.36%.
Overall, the feature selection process seems to have improved the performance of the model. By selecting the most important features, the model can avoid overfitting and focus on the relevant information in the data. This can lead to better generalization and more accurate predictions of new, unseen data.

















PART 5:

10.0 Discuss the challenges and implications regarding the time required to build the required models. Compare the times with and without the feature selection method.

We try implementing 4 different classification Algorithm of which we found the Decision Tree more efficient and optimized in terms of all factors in the existing case study author used Random Forest for the same dataset and had a great result time of execution was high in decision tree which was a drawback for this algorithm whereas other algorithms performed well in terms of Time complexity. Naive Bayes, Random Forest, performed well in case of time complexity rendering time was quite high for this algorithm. Later on, using the same algorithm after feature selection, there was not much fluctuation in the accuracy of the algorithm, f1score was quite challenging later we try using another feature selection method i.e. Correlation which gave better results in terms of accuracy.
Challenges for selecting Neural Network Model: Neural Network model selection was quite challenging very first attempt we used Multi-Layer Perceptron Model which gave an excellent output and quite high accuracy compared to the classification algorithm.
•	Adjusting Number of Layers
•	Batch_size and Ephocs

The Data shape was quite high to fit in this was one big challenge here we used Scaler to scale the data we used Standard scaler for this which gave outstanding results.
A number of layer selections were challenging and adjusting it to the perfect number to optimize time and space complexity was the challenge to tackle it we executed the algorithm using different parameters.

Selecting the appropriate architecture: might be challenging because there are so many different neural network topologies available. The number of layers, the number of neurons per layer, and the activation functions are all significant choices that may impact the network's performance.

Pre-processing of the data: Neural networks need a lot of data to train properly. To make sure that the network can learn useful patterns, the data may need to be pre-processed in a variety of methods, such as normalisation, scaling, or one-hot encoding.

Selecting the appropriate design: might be challenging because there are so many different neural network topologies available. The number of layers, the number of neurons per layer, and the activation functions are all significant choices that may impact the network's performance.
Pre-processing of the data: Neural networks need a lot of data to train properly. To make sure that the network can learn useful patterns, the data may need to be pre-processed in a variety of methods, such as normalisation, scaling, or one-hot encoding.
Computing power and Results interpretation: Neural networks may consume a lot of computing power, especially when working with huge datasets or intricate topologies. It might not be possible to train a network on a single computer, and specialised hardware like GPUs or TPUs would be required.  
