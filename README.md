# ml-linear-regression-20241001
`🎓` Machine Learning Series: Understand Logistic Regression

---

# Index 
1. [Introduction to Logistic Regression](#introduction-to-logistic-regression)
    - Overview and Definition
    - Classification vs. Regression
    - Applications of Logistic Regression
2. [Theoretical Foundations](#theoretical-foundations)
    - Sigmoid Function and Log-Odds
    - Logistic Regression Process
    - Decision Boundaries
    - Visual Flow
3. [Mathematical Formulation](#mathematical-formulation)
    - Logistic Regression Equation
    - Log-Likelihood and Cost Function
    - Gradient Descent for Logistic Regression
4. [Model Assumptions](#model-assumptions)
    - Independence of Observations
    - Linearity of Independent Variables and Log-Odds
    - Handling Binary and Multiclass Problems
5. [Fitting the Model](#fitting-the-model)
    - Maximum Likelihood Estimation (MLE)
    - Training and Optimization Techniques
    - Python Code Example
6. [Model Evaluation](#model-evaluation)
    - Confusion Matrix
    - Accuracy, Precision, Recall, F1-Score
    - ROC Curve and AUC (Area Under Curve)
7. [Dealing with Overfitting](#dealing-with-overfitting)
    - Regularization Techniques (L1, L2)
    - Cross-Validation and Model Selection
8. [Interpretation of Coefficients](#interpretation-of-coefficients)
    - Odds Ratios and Coefficients
    - Interpreting Categorical and Continuous Variables
9. [Conclusion](#conclusion)
    - Summary of Key Points
    - Insights on Logistic Regression's Importance in Machine Learning
    - Next Steps in the Series

---

➥ _**Practical Example**_:
    > [`⛁` Dataset Overview]()
    > [`☠` Logistic Regression on Real-World Data (e.g., Disease Prediction)]()

---

# [Introduction to Logistic Regression](#introduction-to-logistic-regression)

- Logistic regression predicts the *likelihood of an event happening*, like whether someone voted or didn’t, based on a dataset of independent variables.
- This type of statistical model (also known as `logit` model) is often used for classification and predictive analytics.
  
- `💡` Types of logistic regression:
    - Binary (eg. Pass/Fail)
    - Multiclass (eg. Cats, Dogs, Bunnies)
    - Ordinal (eg. Low, Medium, High)

<p align="center">
  <img src="https://fabbookreviews.com/wp-content/uploads/2014/11/img-340595-1-futurama-fry-meme-generator-not-sure-if-real-or-spam-a925b1.jpg?w=640" alt="meme" width="300" />
</p>

`🗲` logistic regression is a supervised learning algorithm. It is used for classification tasks where the model is trained on labeled data, meaning that both the input features and the corresponding output labels (e.g., whether an event occurred or not) are provided during the training process.

## Classification vs. Regression

| **Regression Algorithms**                                               | **Classification Algorithms**                                               |
|----------------------------------------------------------------------|--------------------------------------------------------------------------|
| The output variable must be either continuous nature or real value. | The output variable has to be a discrete value.                         |
| The regression algorithm’s task is mapping input value (x) with continuous output variable (y). | The classification algorithm’s task is mapping the input value of x with the discrete output variable of y. |
| They are used with continuous data.                                   | They are used with discrete data.                                       |
| It attempts to find the best fit line, which predicts the output more accurately. | Classification tries to find the decision boundary, which divides the dataset into different classes. |
| Regression algorithms solve regression problems such as house price predictions and weather predictions. | Classification algorithms solve classification problems like identifying spam emails, spotting cancer cells, and speech recognition. |
| We can further divide regression algorithms into Linear and Non-linear Regression. | We can further divide classification algorithms into Binary Classifiers and Multi-class Classifiers. |

### What's meant by Discrete here?
In the context of data, "discrete" refers to values that can be counted and have distinct, separate categories. For example, the outcome of a classification problem, such as whether an email is spam or not, represents a discrete value since it falls into one of the two categories. In contrast, continuous values can take any value within a range, such as height or temperature.

## 🗲 Applications of Logistic Regression

- **Binary Classification**:
  - **Spam Detection**: Classifies emails as "spam" or "not spam."
  - **Credit Approval**: Determines if a loan application is "approved" or "denied."

- **Multiclass Classification**:
  - **Image Recognition**: Identifies objects in images, such as "cat," "dog," or "car."
  - **Document Classification**: Categorizes documents into topics like "sports," "politics," or "entertainment."

- **Ordinal Classification**:
  - **Customer Satisfaction Surveys**: Ranks responses as "poor," "fair," "good," or "excellent."
  - **Movie Ratings**: Classifies films on a scale from "1 star" to "5 stars," indicating varying levels of preference.

# [Theoretical Foundations](#theoretical-foundations)
ⓘ While regression models predict continuous variables, they can also predict probabilities. Logistic regression specifically uses a probability-predicting model to classify outcomes by applying a decision boundary.

## Sigmoid Function and Log-Odds
### - Probabilities
- **Definition**: The likelihood of an event occurring, ranging from 0 to 1.
- **Example**: A 60% chance of rain is a probability of 0.6.

### - Odds
- **Definition**: The ratio of the probability of an event occurring to the probability of it not occurring.
- **Formula**: $\text{Odds} = \frac{P(\text{event})}{1 - P(\text{event})}$
- **Example**: If the probability of rain is $\( P(\text{rain}) = 0.6 \)$, the odds can be calculated as follows:

$$
\text{Odds} = \frac{0.6}{1 - 0.6} = \frac{0.6}{0.4} = 1.5
$$

### - Log-Odds (Logit Function)
- **Definition**: The natural logarithm of the odds.
- **Purpose**: Transforms probabilities to a scale from negative infinity to positive infinity.
- **Advantage**: Creates a linear relationship with predictor variables.
- It's more accurate to say that log-odds **transform odds to an unbounded scale**.

### - Sigmoid Function
- **Definition**: A function that maps any real-valued number to a value between 0 and 1.
- **Formula**: $f(x) = \frac{1}{1 + e^{-x}}$
- **Purpose**: Converts log-odds back into probabilities.

## Logistic Regression Process
**❶** Calculate log-odds using a linear combination of predictors.

**❷** Use the sigmoid function to convert log-odds to probabilities.

**❸** Apply a decision boundary to classify predictions into discrete categories.

## Decision Boundaries
- **Definition**: The threshold used to classify predictions into discrete categories.
- **Example**: If probability > 0.5, predict 1 (event occurs); otherwise, predict 0.

## Visual Flow
<p align="center">
    <img src="https://github.com/user-attachments/assets/9a9bf111-fd6e-4da5-9b67-8931b5301d89" alt="lr-flow" width="500"/>
</p>

**`⌕`** This schema illustrates the flow from input features to the final classification in logistic regression. It shows how the model uses a linear combination of features to calculate log-odds, transforms these to probabilities using the sigmoid function, and then applies a decision boundary to make the final classification.

_**I hope this gives you a big picture before diving into the mathematical formulation :)**_

## [Mathematical Formulation](#mathematical-formulation)


### Logistic Regression Equation


### Log-Likelihood and Cost Function


### Gradient Descent for Logistic Regression



4. [Model Assumptions](#model-assumptions)
    - Independence of Observations
    - Linearity of Independent Variables and Log-Odds
    - Handling Binary and Multiclass Problems
5. [Fitting the Model](#fitting-the-model)
    - Maximum Likelihood Estimation (MLE)
    - Training and Optimization Techniques
    - Python Code Example
6. [Model Evaluation](#model-evaluation)
    - Confusion Matrix
    - Accuracy, Precision, Recall, F1-Score
    - ROC Curve and AUC (Area Under Curve)
7. [Dealing with Overfitting](#dealing-with-overfitting)
    - Regularization Techniques (L1, L2)
    - Cross-Validation and Model Selection
8. [Interpretation of Coefficients](#interpretation-of-coefficients)
    - Odds Ratios and Coefficients
    - Interpreting Categorical and Continuous Variables
9. [Conclusion](#conclusion)
    - Summary of Key Points
    - Insights on Logistic Regression's Importance in Machine Learning
    - Next Steps in the Series


