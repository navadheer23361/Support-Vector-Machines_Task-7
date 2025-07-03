# Support-Vector-Machines_Task-7

📌 Dataset

Breast Cancer Wisconsin Diagnostic Dataset

diagnosis: Target label (M = malignant, B = benign)

30 numeric features extracted from breast mass ima

🧪 Workflow

1. Data Preprocessing

  Drop irrelevant id column

  Encode diagnosis: M → 1, B → 0

  Standardize features using StandardScaler

2. Model Training
   
Train SVM with Linear Kernel

Train SVM with RBF Kernel for non-linear boundaries

3. Dimensionality Reduction for Visualization
   
Apply PCA to reduce 30 features to 2 components

Plot decision boundaries for both kernels using test set

4. Hyperparameter Tuning
   
Use GridSearchCV for:

C: [0.1, 1, 10, 100]

gamma: [0.001, 0.01, 0.1, 1]

Use 5-fold cross-validation

Evaluate best model on test set

5. Evaluation
   
Accuracy

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)


📊 Sample Visualization

PCA-reduced 2D scatter plot with SVM decision boundary:

Linear kernel gives straight boundary

RBF kernel gives curved/non-linear boundary

