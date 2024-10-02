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
    - Independence of Observations
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


# [Model Assumptions](#model-assumptions)

## Independence of Observations



## Linearity of Independent Variables and Log-Odds



## Handling Binary and Multiclass Problems









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


