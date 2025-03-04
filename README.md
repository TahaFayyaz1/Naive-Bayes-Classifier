# Naive Bayes Classifier for Student Performance Prediction

**Course**: COMP 2211 Exploring Artificial Intelligence  
**Lab**: 2  
**Objective**: Implement a Naive Bayes Classifier from scratch to predict whether a student will pass or fail a course based on study habits and other features.

## Dataset
The dataset contains categorical features related to student behavior:
- **Features**: Hours Studied, Sleep Hours, Interest Level, Attendance, Social Media Usage, Extracurricular Activities.
- **Target**: Course Result (0 = Fail, 1 = Pass).
- **Files**:
  - `X_train.csv`, `X_test.csv`: Feature matrices for training/testing.
  - `y_train.csv`, `y_test.csv`: Labels for training/testing.

## Implementation
The classifier was built from scratch, focusing on:
1. **Prior Probabilities**: Calculated class probabilities from training data.
2. **Likelihoods**: Computed with Laplace smoothing to avoid zero probabilities.
3. **Posterior Probabilities**: Applied Bayes' theorem for predictions.
4. **Metrics**: Accuracy, Precision, Recall, F1-score.

### Key Code Components
- `OurImplementedNaiveBayesCategorical`: Custom Naive Bayes class.
- `compute_priors()`: Calculates class priors.
- `compute_likelihoods()`: Handles feature likelihoods with smoothing.
- `compute_posteriors()`: Combines priors and likelihoods for predictions.
- `compute_metrics()`: Evaluates model performance.

## Results
- **Accuracy**: 84%
- **F1-Score**: 0.90
- **Precision**: 0.88
- **Recall**: 0.92

The custom implementation matches Scikit-learn's `CategoricalNB`, verified by identical predictions.

## Dependencies
- Python 3.x
- Libraries: NumPy, Pandas

## Usage
1. **Upload Data**: Place `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` in the working directory.
2. **Run Code**:
   ```python
   classifier = OurImplementedNaiveBayesCategorical(alpha=1e-10)
   classifier.fit(X_train, y_train)
   predictions = classifier.predict(X_test)
   accuracy, f1, precision, recall = compute_metrics(y_test, predictions)
