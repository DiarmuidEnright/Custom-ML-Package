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
