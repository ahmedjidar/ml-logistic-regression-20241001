# ml-logistic-regression-20241001
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
    - Logistic Function Expression (Sigmoid)
    - General Logistic Regression Equation & Result
    - Log-Likelihood and Cost Function
    - Gradient Descent for Logistic Regression
5. [Model Assumptions](#model-assumptions)
    - Binary or Ordinal Dependent Variable
    - Independence of Observations
    - No Multicollinearity
    - Linearity of Independent Variables and Log-Odds
    - Handling Binary and Multiclass Problems
6. [Fitting the Model](#fitting-the-model)
    - Maximum Likelihood Estimation (MLE)
    - Training and Optimization Techniques
    - Python Code Example
7. [Model Evaluation](#model-evaluation)
    - Confusion Matrix
    - Accuracy, Precision, Recall, F1-Score
    - ROC Curve and AUC (Area Under Curve)
8. [Dealing with Overfitting](#dealing-with-overfitting)
    - Regularization Techniques (L1, L2)
    - Cross-Validation and Model Selection
9. [Interpretation of Coefficients](#interpretation-of-coefficients)
    - Odds Ratios and Coefficients
    - Interpreting Categorical and Continuous Variables
10. [Conclusion](#conclusion)
    - Insights on Logistic Regression's Importance in Machine Learning
    - Next Steps in the Series
11. [Further Resources](#further-resources)

---

➥ _**Practical Example**_:
    > [`⛁` Dataset Overview](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
    > [`☠` Logistic Regression (Binary) on real-world data example (e.g., Heart Disease Prediction)](ml-logistic-regression-20241001.ipynb)

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
<p align="center">
    <img src="https://github.com/user-attachments/assets/6b9a8766-6c07-47d7-bbe2-3020c14f4152" alt="dcb"/>
</p>

- **Definition**: The threshold used to classify predictions into discrete categories.
- **Example**: If probability > 0.5, predict 1 (event occurs); otherwise, predict 0.

## Visual Flow
<p align="center">
    <img src="https://github.com/user-attachments/assets/9a9bf111-fd6e-4da5-9b67-8931b5301d89" alt="lr-flow" width="500"/>
</p>

**`⌕`** This schema illustrates the flow from input features to the final classification in logistic regression. It shows how the model uses a linear combination of features to calculate log-odds, transforms these to probabilities using the sigmoid function, and then applies a decision boundary to make the final classification.

_**I hope this gives you a big picture before diving into the mathematical formulation :)**_

# [Mathematical Formulation](#mathematical-formulation)
🔥 Now, let us get into the math behind involvement of log odds in logistic regression.

## Logistic Function Expression (Sigmoid)

The **logistic function** expresses the probability of success in a logistic regression model. It transforms a linear combination of input features into a value between 0 and 1, representing a probability.

$$
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}
$$

Where:
- $\( p \)$ = probability of success (dependent variable = 1)
- $\( \beta_0 \)$ = intercept
- $\( \beta_1 \)$ = coefficient of the independent variable $\( x \)$
- $\( x \)$ = independent variable

> Simple Interpretation:

- $\( \beta_0 \)$ is the intercept (bias term), and $\( \beta_1, \beta_2, \)$ etc., are the coefficients that determine the influence of the input variables $\( x_1, x_2, \)$ etc., on the outcome.

- $\( e^{-(\beta_0 + \beta_1 x)} \)$: This controls how quickly the probability changes as the input features change. Large positive values in $\( (\beta_0 + \beta_1 x) \)$ make the probability of success approach 1, while large negative values make the probability approach 0.


## General Logistic Regression Equation & Result
- As we said, the odds of an event occurring (success) is defined as the ratio of the probability of success $(\( p \))$ to the probability of failure $(\( 1 - p \))$.

$$
  \text{Odds} = \frac{p}{1 - p}
$$

In `logistic regression`, based on the `Logistic Function definition`, the probability of the independent variable corresponding to a success is given by:

$$
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}} = \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}}
$$

Where:
- $\( p \)$ = probability of success
- $\( \beta_0, \beta_1 \)$ = coefficients (weights) of the logistic regression model
- $\( x \)$ = independent variable

> The probability of the independent variable corresponding to a failure is:

$$
1 - p = 1 - \frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}} = \frac{1}{1 + e^{\beta_0 + \beta_1 x}}
$$

> Therefore, the odds ratio is:

$$
\frac{p}{1 - p} = \frac{\frac{e^{\beta_0 + \beta_1 x}}{1 + e^{\beta_0 + \beta_1 x}}}{\frac{1}{1 + e^{\beta_0 + \beta_1 x}}} = e^{\beta_0 + \beta_1 x}
$$

> Then, to linearize the relationship, we take the natural logarithm of the odds, which is called the **log-odds** or **logit** function:

$$
\text{Logit}(p) = \log \left( \frac{p}{1 - p} \right) = \beta_0 + \beta_1 x
$$

- Just as we mentionned, the logit introduced an unbounded interval and effectively linearized the relationship!

🗲 This is the general equation of logistic regression.

## ⚠ IMPORTANT ⚠
### ✎ Transformation to unbounded scale:
- The log-odds transform the odds from a scale of (0, +∞) to (-∞, +∞).

- This unbounded scale is indeed compatible with the real number line (ℝ), which is what algebraic polynomials work with.

### ✎ Compatibility with algebraic polynomials:
- The linear combination of predictors (β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ) can result in any real number.

- This matches perfectly with the range of the log-odds function, which is also any real number.

### ✎ Application of threshold:
- Because we're working on an unbounded scale, we can indeed apply any threshold we want.

- Typically, we use a threshold of 0 in log-odds space, which corresponds to a probability of 0.5.

- But we could choose different thresholds if needed for specific applications.

- Therefore, in logistic regression, **WE ASSUME** that the log-odds of the probability of success is a linear combination of the input features $(\( x \))$ and their corresponding coefficients $(\( \beta \))$.

$$
  \text{Logit}(p) = \beta_0 + \beta_1 x
$$

- Or (β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ) for multiple predictors.
  
- _**Final Result**_:
 By applying the logistic function to the **logit function's linear expression assumption**, you return to the original probability 𝑝 of success, on which you can apply a decision boundary based on your use case and classify the outcome.

$$
p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}} 
$$

🗲 That's why the Logistic Function is known as the inverse of the Logit Function.


### Log-Likelihood and Cost Function

#### 1. Binary Classification Likelihood and Log-Likelihood Function in Logistic Regression

| **Concept**               | **Likelihood Function**                                                                                                                       | **Log-Likelihood Function**                                                                                                         |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Definition**             | The likelihood function represents the probability of observing the given data, based on a set of model parameters ($\beta$ coefficients).      | The natural logarithm of the likelihood function. Simplifies calculations by converting a product into a sum.                       |
| **Mathematical Expression**| $L(\beta) = \prod \left( p_i^{y_i} \cdot (1 - p_i)^{1 - y_i} \right)$ <br> Where $p_i$ is the predicted probability and $y_i$ is the actual outcome. | $\log L(\beta) = \sum \left( y_i \cdot \log(p_i) + (1 - y_i) \cdot \log(1 - p_i) \right)$                                            |
| **Purpose**                | - Helps find the optimal $\beta$ coefficients. <br> - Determines how well the model fits the data.                                              | - Easier computation (sum instead of product). <br> - Avoids underflow issues with small probabilities.                             |
| **Relationship**           | Works with predicted probabilities using the logistic (sigmoid) function.                                                                     | The log-likelihood is used because it is computationally easier and has nicer mathematical properties for maximization.              |

ⓘ In practice, we usually work with the log-likelihood, but the terms are sometimes used interchangeably because maximizing either will give the same parameter estimates.


#### 2. Binary Classification Cost function for a Logistic Regression in comparaison with Linear Regression

| **Property**            | **Logistic Regression**                              | **Linear Regression**                                      |
|-------------------------|------------------------------------------------------|------------------------------------------------------------|
| **Cost Function**        | Cross-Entropy (Log-Loss)                             | Mean Squared Error (MSE)                                    |
| **Why It's Used**        | Directly relates to the likelihood function          | Assumes normally distributed errors; leads to maximum likelihood estimation |
| **Formula**              | **Cross-Entropy = -1 \* (Log-Likelihood)** = $$-\sum_{i=1}^N \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$ |$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$ |
| **Flexibility**          | Can use other cost functions, but **Cross-Entropy** is most common and suitable for **classification** problems | Can also use **MAE**, **Huber Loss**, or others depending on the problem; best for **regression** with normally distributed errors |

#### 3. N-Multiclass Generalization
| Concept               | Likelihood                                                                 | Log-Likelihood                                                           | Cross-Entropy (Log Loss)                                                 |
|-----------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **Definition**         | Likelihood is the probability of observing the actual outcomes for all classes. | The natural logarithm of the likelihood. Simplifies multiplication into addition. | Negative log-likelihood. Measures how well the predicted probabilities match the actual outcomes. |
| **Mathematical Form**  | $L(\beta) = \prod_{i=1}^{N} \prod_{c=1}^{C} p_{ic}^{y_{ic}}$          | $\log L(\beta) = \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(p_{ic})$ | $- \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(p_{ic})$               |
| **Explanation**        | - For each observation $\( i \)$, multiply the predicted probabilities $\( p_{ic} \)$ raised to the power of the actual outcome $\( y_{ic} \)$ (1 if class is correct, 0 otherwise). | - Take the natural log of the likelihood to simplify calculations and prevent issues with very small probabilities. | - This is the negative sum of log-likelihood, which penalizes incorrect predictions more severely. |
| **Purpose**            | Helps to find the best model parameters (like weights $\( \beta \))$ that make the observed data most likely. | Log transformation makes it easier to compute and work with. | Used as the cost function in logistic regression for classification tasks. |

#### 4. Explaining the consfusing Transition 😵‍💫

➜ **Going from Bernoulli to Multiclass Classification**

> 1️⃣ Bernoulli Distribution (Binary Case)

In binary classification, we have two outcomes: success (1) and failure (0). The likelihood for a single observation can be represented as:

> Likelihood Function

$$
L(\beta) = p^y (1 - p)^{(1 - y)}
$$

Where:
- $\( p \)$ is the predicted probability of success.
- $\( y \)$ is the actual outcome (0 or 1).

> **Explanation**:
- Here, we model the probability of the actual outcome $\( y \)$ given the predicted probability $\( p \)$.
- The likelihood function calculates how likely the observed outcome is, based on the predicted probability.

> 2️⃣ Extending to Multiclass Classification

When we move to multiclass classification (where there are more than two classes), we have $\( C \)$ classes instead of just two. For each observation $\( i \)$, we want to consider the probability of each class $\( c \)$.

> Likelihood in Multiclass

For each observation, we now calculate the likelihood as a product of probabilities for all classes:

$$
L(\beta) = \prod_{i=1}^{N} \prod_{c=1}^{C} p_{i}^{y_{ic}}
$$

Where:
- $\( p_{ic} \)$ is the predicted probability for observation $\( i \)$ belonging to class $\( c \)$.
- $\( y_{ic} \)$ is 1 if observation $\( i \)$ belongs to class $\( c \)$, otherwise it is 0.

> **Explanation**:
- In multiclass, we compute the likelihood for each observation by multiplying the probabilities of the actual class across all classes.
- The outer product runs over all observations, while the inner product runs over all classes.

> 3️⃣ Why the Double Sum?

> **Log-Likelihood**

To make calculations easier, we take the natural log of the likelihood. When you take the log of a product, it turns into a sum:

$$
\log L(\beta) = \log \left( \prod_{i=1}^{N} \prod_{c=1}^{C} p_{i}^{y_{ic}} \right) = \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(p_{ic})
$$

> **Explanation**:
- The outer sum iterates over each observation, while the inner sum iterates over each class for that observation.
- By using the log-likelihood, we can simplify the optimization process as it turns multiplicative relationships into additive ones, making computations more manageable.

> **Now we got**:

- **Single Product**: For each observation, you multiply the probabilities for each class (hence the inner product).
- **Double Sum**: When computing the log-likelihood for multiple classes across multiple observations, you sum over each class for each observation, leading to the double summation.

> **Why Is This Useful?**

This structure allows us to capture the contribution of all classes to the likelihood of observing the actual outcomes, making it very suitable for multiclass classification problems.


### Gradient Descent for Logistic Regression

ⓘ In logistic regression, we use gradient descent to optimize our model parameters by minimizing the cost function, which is derived from the log-likelihood.

#### Steps of Gradient Descent:

1. **Define the Cost Function**: 
   We start with the cost function derived from the log-likelihood for binary classification. The log-likelihood can be expressed as:
   
$$
\ell(\beta) = \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

   To minimize the negative log-likelihood, we define the cost function as:
   
$$
J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]
$$

3. **Calculate the Gradient**: 
   The gradient of the cost function with respect to the model parameters is computed as:
   
$$
\frac{\partial J(\beta)}{\partial \beta_j} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) x_{ij}
$$

   Where:
   - $\( p_i \)$ is the predicted probability for the $\( i \)$-th observation.
   - $\( y_i \)$ is the actual label for the $\( i \)$-th observation.
   - $\( x_{ij} \)$ is the value of the $\( j \)$-th feature for the $\( i \)$-th observation.

5. **Update the Parameters**:
   We update each parameter using the calculated gradient:
   
$$
\beta_j := \beta_j - \alpha \frac{\partial J(\beta)}{\partial \beta_j}
$$

   Where:
   - $\( \alpha \)$ is the **learning rate**.

7. **Iterate**:
   Repeat the process of calculating the gradient and updating the parameters until convergence, meaning the cost function stabilizes.

---

#### ↔ Specificity of the Gradient Expression

| Cost Function       | Gradient Expression                          | Description                                    |
|---------------------|---------------------------------------------|------------------------------------------------|
| Cross-Entropy (Logistic Regression) | $\frac{\partial J(\beta)}{\partial \beta_j} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) x_{ij}$ | Measures the difference between predicted probabilities and actual labels. |
| Mean Squared Error (Linear Regression) | <img src="https://latex.codecogs.com/svg.latex?\color{white}\frac{\partial J(\beta)}{\partial \beta_j} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) x_{ij}" alt="MSE Formula" /> | Measures the difference between predicted values and actual values. |

---

This structure clarifies how gradient descent is used in logistic regression, showing its derivation from the cost function and log-likelihood, while the table highlights the specificity of the gradient expressions for different cost functions and all...


## [Model Assumptions](#model-assumptions)

Unlike linear regression models, logistic regression does not impose strict assumptions regarding the relationship between the independent and dependent variables, nor does it require normality, homoscedasticity, or measurement level constraints often associated with linear regression models using ordinary least squares (OLS) algorithms.

### Key Assumptions in Logistic Regression:

1. **No Strict Linearity Between Independent and Dependent Variables**  
   Logistic regression does **not** require a linear relationship between the dependent variable and the independent variables. Instead, it models the relationship using the log-odds (logit) transformation.

2. **No Requirement for Normally Distributed Errors**  
   In logistic regression, the error terms (residuals) do **not** need to follow a normal distribution. This is different from OLS, where normally distributed residuals are often a critical assumption.

3. **No Homoscedasticity Requirement**  
   Homoscedasticity, where the variance of the residuals is constant across values of the independent variables, is **not** a requirement for logistic regression models.

4. **Dependent Variable Scale**  
   The dependent variable in logistic regression is **not** measured on an interval or ratio scale; rather, it is categorical, typically binary (0 or 1), or ordinal for ordered logistic regression.

---

**However, several key assumptions still hold true:**

### 1. Binary or Ordinal Dependent Variable
- In **binary logistic regression**, the dependent variable must be **binary** (i.e., it takes two possible values, such as 0 or 1). For **ordinal logistic regression**, the dependent variable must be **ordinal**, meaning it has a natural order (e.g., "low," "medium," and "high").

  Mathematically, for binary logistic regression:

$$
Y \in \{0, 1\}
$$

### 2. Independence of Observations
- The model assumes that observations are **independent** of each other, which means no repeated measurements or matched-pair data can be used without special handling. Violating this assumption can lead to biased standard errors.

  In mathematical terms:

$$
P(y_i | y_j) = P(y_i) \quad \text{for} \quad i \neq j
$$

> In other words, the probability of yiyi​ occurring should remain the same, regardless of whether yjyj​ has occurred or not.

### 3. No Multicollinearity
- Logistic regression assumes that there is little or no **multicollinearity** among the independent variables, meaning the predictors should not be highly correlated with each other. Multicollinearity can inflate the standard errors of the coefficients and lead to unstable estimates.

  To detect multicollinearity, you can use the **Variance Inflation Factor (VIF)**:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

  Where $R_j^2$ is the R-squared value of the regression of the $j$-th independent variable on the remaining variables.

### 4. Linearity of Independent Variables and Log-Odds
- Although logistic regression does not require a linear relationship between the independent variables and the dependent variable, it assumes a linear relationship between the independent variables and the **log-odds** (logit) of the dependent variable.

  The log-odds are represented by the following equation:

$$
\log \left( \frac{p}{1 - p} \right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n
$$

  Where:
  - $p$ is the probability of the outcome (e.g., success or 1),
  - $x_1, \dots, x_n$ are the independent variables, and
  - $\beta_0, \beta_1, \dots, \beta_n$ are the coefficients of the model.

---

### 5. Handling Binary and Multiclass Problems
- Logistic regression is primarily used for **binary classification**, where the dependent variable takes on two possible outcomes (e.g., 0 and 1). However, the method can also be extended to handle **multiclass problems** using techniques like **One-vs-All (OvA)** and **Multinomial Logistic Regression**.

  In **One-vs-All** classification, a series of binary logistic regressions is run, each treating one class as the positive class and all other classes as the negative class.

  The **multinomial logistic regression** extends the model to multiple categories (i.e., for more than two classes):

$$
P(y_i = j) = \frac{e^{\beta_j^T x_i}}{1 + \sum_{k=1}^{K-1} e^{\beta_k^T x_i}}
$$

  Where:
  - $K$ is the number of classes,
  - $x_i$ is the vector of independent variables,
  - $\beta_j$ are the coefficients for class $j$.

ℹ️ **Multinomial logistic regression** is known by a variety of other names, including _**`polytomous LR`**_, _**`multiclass LR`**_, _**`softmax regression`**_, _**`multinomial logit` (mlogit)**_, _**`the maximum entropy (MaxEnt) classifier`**_, and _**the `conditional maximum entropy model`**_.

➥ _Check [multinomial-regression-brief-10042024.md](multinomial-regression-brief-10042024.md) for basic explanation._

---

### Summary Table of Assumptions:

| Assumption                        | Description                                                                                                | Mathematical Expression                                           |
|------------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| Binary or Ordinal Dependent Variable | For binary logistic regression, the outcome must be binary; for ordinal logistic regression, it must be ordinal. | $Y \in \{0, 1\}$ |
| Independence of Observations        | Observations must be independent of one another.                                                           | $P(y_i│y_j) = P(y_i) \quad \text{for} \quad i \neq j$       |
| No Multicollinearity               | Independent variables should not be highly correlated.                                                      | $\text{VIF}_j = \frac{1}{1 - R_j^2}$                          |
| Linearity of Log-Odds              | The log-odds of the dependent variable should be linearly related to the independent variables.              | $\log \left( \frac{p}{1 - p} \right) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$ |
| Handling Multiclass Problems       | Logistic regression can be extended to handle multiclass problems using techniques like One-vs-All and Multinomial Logistic Regression. | $P(y_i = j) = \frac{e^{\beta_j^T x_i}}{1 + \sum_{k=1}^{K-1} e^{\beta_k^T x_i}}$ |

---

# [Fitting the Model](#fitting-the-model)

## 1. **Maximum Likelihood Estimation (MLE)**

In logistic regression, the goal is to estimate the model parameters that best fit the data. This is done through **Maximum Likelihood Estimation (MLE)**. The MLE method estimates the coefficients by finding the parameter values that maximize the likelihood function.

- **For Binary Logistic Regression**:
  
  In binary classification, the likelihood function measures the probability of the observed binary outcomes, given the model's predicted probabilities.

  As we discussed eariler, the **log-likelihood** function for binary logistic regression is expressed as:

$$
\text{LL}(\beta) = \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
$$

  Where:
  - $y_i$ is the observed value (0 or 1) for the $i^{th}$ instance.
  - $\hat{p}_i$ is the predicted probability that the $i^{th}$ instance belongs to the positive class.

**For Multinomial Logistic Regression**:

  In multinomial logistic regression, the outcome can take more than two categories (e.g., $K$ classes). The likelihood function is extended to handle multiple categories:

$$
\text{LL}(\beta) = \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})
$$

  Where:
  - $y_{ik}$ is 1 if the $i^{th}$ instance belongs to class $k$, otherwise 0.
  - $\hat{p}_{ik}$ is the predicted probability that the $i^{th}$ instance belongs to class $k$.

  💡 The goal is still to maximize the log-likelihood function to find the best parameters for the model.

## 2. **Training and Optimization Techniques**

Once we define the log-likelihood function, we use optimization techniques to estimate the parameters.

- **For Binary Logistic Regression**:

  In binary logistic regression, one common approach is to use **Gradient Descent** to maximize the log-likelihood. The parameter update formula is:

$$
\beta_j = \beta_j - \alpha \frac{\partial}{\partial \beta_j} \text{LL}(\beta)
$$

  Where:
  - $\alpha$ is the learning rate (step size).
  - $\frac{\partial}{\partial \beta_j} \text{LL}(\beta)$ is the gradient of the log-likelihood with respect to the parameter $\beta_j$.

**For Multinomial Logistic Regression**:

  > In the multinomial case, the update rule for **Gradient Descent** is similar but applies to each class separately:

$$
\beta_{kj} = \beta_{kj} - \alpha \frac{\partial}{\partial \beta_{kj}} \text{LL}(\beta)
$$

  Here, the model estimates $K - 1$ sets of coefficients (relative to a reference class).

- **Alternative Optimization Methods** (for both binary and multinomial):
  - **Newton-Raphson Method**: A second-order method that uses both the gradient and the Hessian matrix (second derivative) to find optimal parameter values more quickly. This method converges faster than gradient descent but is computationally expensive for large datasets.
  - **Stochastic Gradient Descent (SGD)**: A variant of gradient descent that updates the parameters using one or a few examples at a time, which can be faster for very large datasets.

## 3. **Python Code Example**

> Below is a Python code example using `scikit-learn` to fit a logistic regression model:

```python
# Importing required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example dataset: X (independent variables) and y (dependent variable)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a logistic regression model
model = LogisticRegression(solver="liblinear")

# Fitting the model
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

> For Multinomial Logistic Regression, the code is quite similar, but you need to specify the multi_class='multinomial' parameter in the LogisticRegression() function:

```python
# Multinomial Logistic Regression Example
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
```

### `❗` IMPORTANT
Solver Choice in Logistic Regression.

<p align="center">
    <img src="https://github.com/user-attachments/assets/7162fa80-004f-485d-9ed4-2e5eb20946a0" alt="warn" width="500"/>
</p>

When selecting the appropriate **solver** for logistic regression, it's important to consider the following aspects:

- For **small datasets**, the `'liblinear'` solver is efficient.
- For **large datasets**, `'sag'` and `'saga'` solvers are generally faster.
- For **multiclass problems**, only the following solvers handle the **multinomial loss**:
  - `'newton-cg'`
  - `'sag'`
  - `'saga'`
  - `'lbfgs'` (default)
  
  **Note:** `'liblinear'` and `'newton-cholesky'` are limited to **binary classification** by default. To apply them to multiclass problems, you can wrap the model with `OneVsRestClassifier`.

- The **'newton-cholesky'** solver is particularly suited for **n_samples >> n_features**, especially when dealing with **one-hot encoded** categorical features. However, be aware that it has a **quadratic memory dependency on n_features**, as it explicitly computes the Hessian matrix.


### `✍️` Key Points:

- MLE is used to estimate parameters in both binary and multinomial logistic regression.
    
- Gradient Descent and the Newton-Raphson Method are common optimization techniques for fitting the model.
    
- Python's scikit-learn provides an easy interface to fit and evaluate logistic regression models for both binary and multinomial cases.

### Sample Table for Optimization Methods:
  
| **Optimization Method**          | **Description**                                                                                 | **Pros**                            | **Cons**                                   |
|----------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------|--------------------------------------------|
| **Gradient Descent**              | Iterative method that updates parameters by moving in the direction of the gradient             | Simple to implement                 | Can be slow, sensitive to learning rate    |
| **Newton-Raphson/IRLS**           | Second-order method that uses both the gradient and Hessian for faster convergence              | Faster convergence                  | Computationally expensive for large data   |
| **Stochastic Gradient Descent**   | Variant of gradient descent that updates parameters using one or a few examples at a time       | Faster for large datasets           | More variance in updates, noisier          |
| **L-BFGS (Limited memory BFGS)**  | Quasi-Newton method for approximating the Hessian, used for large-scale optimization problems   | Efficient for large datasets        | More complex to implement                  |





# [Model Evaluation](#model-evaluation)

## 1. Confusion Matrix

- The **Confusion Matrix** is a table used to evaluate the performance of a classification model by comparing the predicted and actual values.
- It contains four elements:
  - **True Positives (TP):** Correctly predicted positive cases.
  - **True Negatives (TN):** Correctly predicted negative cases.
  - **False Positives (FP):** Incorrectly predicted positive cases (Type I error).
  - **False Negatives (FN):** Incorrectly predicted negative cases (Type II error).

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive  | True Positive (TP)  | False Negative (FN) |
| Actual Negative  | False Positive (FP) | True Negative (TN)  |

<p align="center">
    <img src="https://github.com/user-attachments/assets/48a19453-f96e-4ba5-9d13-bdd3ada66270" alt="tptnfpfnlr" width="500"/>
</p>

- True Positives (TP): These tend to be in the top right of the graph, where the predicted probability is high and the actual class is positive.
  
- True Negatives (TN): These tend to be in the bottom left of the graph, where the predicted probability is low and the actual class is negative.
  
- False Positives (FP): These are typically found in the top left quadrant. The model predicts a high probability (so they're towards the top), but they're actually negative cases (so they're on the left side).

- False Negatives (FN): These are typically found in the bottom right quadrant. The model predicts a low probability (so they're towards the bottom), but they're actually positive cases (so they're on the right side).

## 2. Accuracy, Precision, Recall, F1-Score

### *Accuracy*:
- Accuracy measures the overall correctness of the model:
  
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

### *Precision*:
- Precision is the ratio of correctly predicted positive observations to the total predicted positives:

$$ 
\text{Precision} = \frac{TP}{TP + FP} 
$$

### *Recall (Sensitivity)*:
- Recall is the ratio of correctly predicted positive observations to all actual positives:
- *Out of all the actual positive cases, how many did we correctly identify?*
  
$$ 
\text{Recall} = \frac{TP}{TP + FN} 
$$

### *F1-Score:*
- The F1-Score is the harmonic mean of precision and recall, providing a balance between the two:
  
$$ 
\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

<p align="center">
    <img src="https://images.prismic.io/encord/edfa849b-03fb-43d2-aba5-1f53a8884e6f_image5.png?auto=compress,format" alt="cm" width="700" />
</p>

## 3. ROC Curve and AUC (Area Under the Curve)

### *ROC Curve:*
- The **Receiver Operating Characteristic (ROC)** curve plots **True Positive Rate (TPR)** against **False Positive Rate (FPR)** at different classification thresholds.
  - **True Positive Rate (TPR)** = Recall = $\frac{TP}{TP + FN}$
  - **False Positive Rate (FPR)** = $\frac{FP}{FP + TN}$

  - True Positive Rate (Recall): How many actual positives did we correctly identify? (i.e., out of all actual positives, how many true positives?)
  - False Positive Rate: How many false positives did we predict? (i.e., out of all actual negatives, how many did we wrongly classify as positive?)

_**In other words**_:

  - Y-axis (True Positive Rate / Recall): How well we identify the positive cases.
    
  - X-axis (False Positive Rate): How many negatives we accidentally labeled as positive.

<p align="center">
    <img src="https://github.com/user-attachments/assets/8e37b621-f18e-4982-8b0a-eef34e41e0b7" alt="roc" width="400"/>
</p>

**ⓘ A perfect model would have high recall (catching all true positives) and a low false positive rate (not mistakenly identifying negatives as positives).**

**ⓘ You want a curve that hugs the top-left corner of the graph, indicating high true positive rate and low false positive rate.**

### *AUC (Area Under the Curve):*
- **AUC** represents the area under the ROC curve and gives a single scalar value to evaluate the model's performance.
  - A **higher AUC** means the model is better at distinguishing between positive and negative classes.

<p align="center">
    <img src="https://github.com/user-attachments/assets/10cdfb36-a632-4c93-843e-7fcc87c1bf52" alt="auc" width="400"/>
</p>

> AUC > 0.5 => Towards good performance.

# [Dealing with Overfitting](#dealing-with-overfitting)

## Regularization Techniques (L1, L2)
Overfitting occurs when a model fits the training data too well, capturing noise and leading to poor generalization on unseen data. Regularization is a technique used to prevent overfitting by adding a penalty to the model's complexity, encouraging simpler models that generalize better.

### L1 Regularization (Lasso)
L1 regularization adds the absolute values of the coefficients as a penalty term to the loss function.

**L1 Regularization Formula:**

$$
J(\theta) = \text{Cost Function} + \lambda \sum_{i=1}^{n} |\theta_i|
$$

Here, $\( \lambda \)$ is the regularization strength, and $\( \theta_i \)$ are the model coefficients. L1 regularization can shrink some coefficients to zero, effectively performing feature selection by eliminating less important variables.

### L2 Regularization (Ridge)
L2 regularization adds the squared values of the coefficients as a penalty term to the loss function.

**L2 Regularization Formula:**

$$
J(\theta) = \text{Cost Function} + \lambda \sum_{i=1}^{n} \theta_i^2
$$

L2 regularization prevents large coefficients and forces the model to spread out the importance more evenly across all features.

In **logistic regression**, regularization helps control the complexity of the model, preventing overfitting while improving the stability of the coefficients.

### Implementing Regularization in Logistic Regression
The logistic regression model can apply both L1 and L2 regularization through the `penalty` parameter, which we used in our example model with `GridSearchCV`. By adjusting the regularization strength $(\( C \)$ in `scikit-learn`), you can find the optimal balance between fitting the data well and keeping the model simple.

## Cross-Validation and Model Selection (GridSearchCV)
Cross-validation is essential in evaluating model performance and selecting the best model without overfitting. **K-fold cross-validation** splits the data into K subsets and evaluates the model on each subset, ensuring good model performance.

In our model, `GridSearchCV` performs cross-validation over a grid of hyperparameters (e.g., regularization strength `C`, penalties `L1`/`L2`, solvers). It helps select the best combination of hyperparameters that maximizes model performance while avoiding overfitting.

# [Interpretation of Coefficients](#interpretation-of-coefficients)

## Odds Ratios and Coefficients
In logistic regression, the coefficients $(\( \beta \))$ represent the log-odds of the outcome for a one-unit increase in the predictor. The relationship between the coefficient and the outcome is not linear like in linear regression but is instead linked through the logistic (sigmoid) function.

- **Log-Odds**: The log of the odds ratio. For a predictor $\( X_i \)$, the log-odds of the target being 1 (positive class) are:
 
$$
\text{log-odds} = \log \left( \frac{P(Y=1)}{P(Y=0)} \right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n
$$

- **Odds Ratio**: To interpret the coefficients in terms of odds ratios, exponentiate the coefficients:

$$
\text{Odds Ratio} = e^{\beta}
$$

  The odds ratio tells us how much the odds of the target increase (if $\( \text{OR} > 1 \)$ ) or decrease (if $\( \text{OR} < 1 \)$ ) for a one-unit increase in the predictor.

## Interpreting Categorical and Continuous Variables
- **Categorical Variables**: The coefficients of categorical variables show how the odds of the target change compared to the reference category. For example, if `sex` is a binary variable (0: female, 1: male), the coefficient of `sex` indicates how the odds of heart disease change for males compared to females.
  
  Example: If $\( \beta_{\text{sex}} = 0.7 \)$, then the odds ratio is $\( e^{0.7} \approx 2.01 \)$, meaning males are twice as likely as females to have heart disease.

- **Continuous Variables**: For continuous variables like `age`, the coefficient tells us how the odds of heart disease change for each one-year increase in age. If the coefficient for age is $\( \beta_{\text{age}} = 0.05 \)$, then the odds ratio is $\( e^{0.05} \approx 1.05 \)$, meaning the odds increase by 5% for each additional year of age.

# [Conclusion](#conclusion)

## Insights on Logistic Regression's Importance in Machine Learning
 Logistic regression is a fundamental algorithm in machine learning, especially useful for binary classification problems, providing interpretable models. It’s also a foundation for more complex techniques like **`neural networks`** and generalized linear models (**`GLMs`**). Its ease of implementation and mathematical soundness make it a staple in both academia and industry.

## Next Steps in the Series
 This series on machine learning is far from over! Now that you've got it almost all about logistic regression, we'll build on this foundation by exploring more advanced topics, such as:
1. **Non-linear Models**: Explore `decision trees`, `random forests`, and `gradient boosting` to capture complex patterns.
2. **Support Vector Machines (`SVM`)**: A powerful classifier for linear and non-linear data.
3. **Neural Networks**: Deepen your understanding of how `neural networks` generalize logistic regression.
4. **Model Evaluation and Improvement**: Dive into `cross-validation` techniques, `hyperparameter tuning`, and `model selection` using more advanced metrics.

## _Thank you for reading & feel free to ask!_ 😁

# [Further Resources](#further-resources)

Implementation_

1. **Scikit-learn Documentation: Logistic Regression**
   - [Scikit-learn: Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
   - This is the official documentation for Scikit-learn’s logistic regression implementation. It explains the parameters, usage, and examples of how to apply logistic regression in Python using Scikit-learn.

2. **Scikit-learn User Guide: Logistic Regression**
   - [Scikit-learn: Logistic Regression (User Guide)](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
   - The user guide dives deeper into the theory and practical applications of logistic regression, covering topics like regularization and solvers.

3. **Kaggle: Logistic Regression Tutorial**
   - [Kaggle: Logistic Regression Tutorial](https://www.kaggle.com/learn/machine-learning-explainability)
   - Kaggle provides a hands-on tutorial with real-world datasets that guide you through logistic regression and interpreting the coefficients and odds ratios.

4. **Google Developers: Logistic Regression in ML Crash Course**
   - [Google Developers: Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression)
   - This is part of Google’s Machine Learning Crash Course, and it provides a concise but clear explanation of logistic regression, its usage, and how it works.

5. **Towards Data Science: Comprehensive Guide on Logistic Regression**
   - [Towards Data Science: Logistic Regression](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)
   - This article on Towards Data Science offers an in-depth overview of logistic regression, including the mathematical intuition behind it and practical Python examples.

6. **Coursera: Machine Learning with Andrew Ng (Logistic Regression)**
   - [Coursera: Machine Learning by Andrew Ng (Logistic Regression)](https://www.coursera.org/learn/machine-learning/lecture/IuA7Y/logistic-regression)
   - Andrew Ng's course on machine learning, which includes a section dedicated to logistic regression. It’s an excellent resource for understanding both the simple math and implementation in various contexts.

7. **Kaggle: Classification Problem Datasets**
   - [Kaggle Datasets](https://www.kaggle.com/datasets)
   - Use Kaggle’s rich dataset repository to practice logistic regression on different classification problems, with plenty of datasets available for real-world applications.


