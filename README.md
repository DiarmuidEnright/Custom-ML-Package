# ml_lib

- Custom machine learning library written entirely in C for efficient processing of large datasets
- Implements commonly used machine learning algorithms:
- **Support Vector Machines (SVM)**
  - Binary classification using linear SVM with configurable hyperparameters.
  - Model persistence: Save and load trained SVM models for future use.

- **Decision Trees**
  - Recursive binary decision tree implementation for classification.
  - Supports both numeric and binary target variables.
  
- **Random Forest**
  - Ensemble learning method using multiple decision trees for classification.
  - Includes bootstrap sampling and majority voting for final prediction.
  
- **Gradient Boosting Machines (GBM)**
  - Implements boosting of decision trees using residual-based learning.
  - Configurable learning rate and number of trees for control over training.
  
- **Preprocessing**
  - **Min-Max Scaling**: Scales features to a specified range.
  - **Standardization (Z-score normalization)**: Scales features based on mean and standard deviation.
  
- **Model Persistence**
  - Ability to save and load trained models for SVM.
  
- **Utilities**
  - K-fold cross-validation for robust model evaluation.
  - Grid search for hyperparameter tuning.


#Expected output:

Neural Network Example:
Neural Network Output: 0.716561

- PCA Example:
- PCA Components:
0.262453 0.047465 0.736082 
0.247039 0.982550 0.722660 
- Performing 5-fold cross-validation on Decision Tree:
Fold 1 accuracy: 1.00
Fold 2 accuracy: 1.00
Fold 3 accuracy: 0.00
Fold 4 accuracy: 0.00
Fold 5 accuracy: 1.00
Average accuracy: 0.60
- Performing 5-fold cross-validation on KNN:
Fold 1 accuracy: 1.00
Fold 2 accuracy: 1.00
Fold 3 accuracy: 0.00
Fold 4 accuracy: 0.00
Fold 5 accuracy: 0.00
Average accuracy: 0.40
- Performing 5-fold cross-validation on SVM:
Fold 1 accuracy: 1.00
Fold 2 accuracy: 1.00
Fold 3 accuracy: 0.00
Fold 4 accuracy: 1.00
Fold 5 accuracy: 1.00
Average accuracy: 0.80
- Applying Bagging Ensemble Method:
Bagging Accuracy: 1.00
- Applying Stacking Ensemble Method:
Stacking Accuracy: 1.00
