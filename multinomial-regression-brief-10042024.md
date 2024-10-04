# Understanding Multinomial Regression

This README provides a big-picture breakdown of the components involved in multinomial regression, specifically focusing on **logits**, **softmax**, and **log-likelihood**. Each of these components serves a distinct purpose in modeling the relationship between predictors and outcomes.

## Components Breakdown

### 1. **Logit Function (Log-Odds)**

- The **logit** function is a **linear function** of the predictors (independent variables). This linearity is important because it allows us to model the relationship between the predictors and the outcome in an **unbounded** manner. Unlike probabilities (which are bounded between 0 and 1), logits can take any real value, making them much easier to model using linear equations.

- The logit transforms probabilities into a linear form (log-odds). For each class $\( j \)$, we model the **log-odds** of the class relative to the baseline category:

$$
\log \left( \frac{P(Y = j)}{P(Y = \text{baseline})} \right) = \beta_{j0} + \beta_{j1} X_1 + \dots + \beta_{jp} X_p
$$

  This transformation makes it easier to **learn** the relationship between the predictors and the classes using linear methods.

### 2. **Softmax Function**

- Once we compute the **logit** (linear predictor), we need to **convert** these logits back into **probabilities**, which is where the **softmax** function comes in. The softmax function takes the unbounded logit values and transforms them into probabilities that sum to 1.

$$
P(y = j) = \frac{e^{logit_j}}{\sum_{k=1}^{K} e^{logit_k}}
$$

- The softmax function operates on the **exponentiated logits** (i.e., $e^{logit_j}$) and ensures the probabilities are bounded between 0 and 1. This means:
  - The logits are used to calculate the **relative likelihood** of each class (through the exponentiated values).
  - The softmax function ensures that these likelihoods are normalized into a proper probability distribution.


### 3. **Log-Likelihood Function**

- After we compute the class probabilities from softmax, we use these probabilities to **optimize the model** by maximizing the **log-likelihood**.

$$
\log L(\beta) = \sum_{i=1}^{N} \sum_{j=1}^{k} y_{ij} \log(P(Y = j))
$$

- The **log-likelihood** helps to find the best-fitting parameters $\( \beta \)$ by comparing the predicted probabilities to the actual observed classes. The goal of the optimization process is to maximize the likelihood that the observed data came from the model's predicted probabilities.

### Order of Flow:

1. **Logit Calculation**: 
   - We compute the logits for each class as a linear combination of the predictor variables $\( X_1, X_2, \dots, X_p \)$. These logits represent the unbounded relationship between the predictors and the outcome class.

2. **Softmax**:
   - The softmax function takes the exponentiated logits and converts them into class probabilities. This ensures that the output is a valid probability distribution (i.e., all probabilities sum to 1 and are between 0 and 1).

3. **Log-Likelihood**:
   - The probabilities from the softmax function are then used in the log-likelihood function. This function evaluates how well the model's predicted probabilities match the actual outcomes, and the optimization process works to adjust the model parameters $\( \beta \)$ to maximize the likelihood of the observed data.

---

## Key Clarification

- **Logits** are not just a way to receive already-calculated probabilities. They are linear combinations of the predictors, and their unbounded nature is what makes them suitable for modeling in a linear way. The logits allow us to express the relationship between the predictors and the outcome in terms of log-odds.
  
- **Softmax** takes these unbounded logits and transforms them into probabilities, which we can then use in the log-likelihood.

- **Log-likelihood** is used to optimize the model by adjusting the parameters such that the predicted probabilities from the softmax match the actual observed class labels.



