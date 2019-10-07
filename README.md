# Salary Prediction Project (Python)

## Dependencies: 
pandas, sklearn, matplotlib, scipy, seaborn, numpy

## Files:
The jupyternotebook is here. 

## 4'D' Data Science Framework: 
I use the 4D framework for my data science projects. It is a simple and reliable method of solving a business problem with data.

# 1. Define the goal: 
This is a simple project with the goal of predicting salaries based on structured data from job postings. The requirement is to get the mean squared error (MSE) down below 360 (RMSE < $60K), or 320 for bonus points!

The MSE is a good metric for this task because it is a regression problem, the loss function will be smoothly differentiable, and we care most about prediction accuracy rather than the explanatory power of the model.

# 2. Discover:
I do not have permission to share the raw data but here is what I found:
* There are 999995 training samples after cleaning the data (dropping invalid salaries)
* There are __ samples in the test set
* The distribution of salary in the train set is mildly rightly skewed (0.3465)

* The **job type** **'janitor'** offers the highest negative correlation to salary (-0.442) 
* **'CEO'** is weakly positively correlated with salary (0.285) followed by most other levels of **job type** and salary distributions shift higher with more senior/executive job types
* The ordinal feature **degree code** offers the next greatest (weak) correlation with salary (0.384) and salary distributions shift higher with increasing level of degree, as would be expected
* **Years of experience** is weakly correlated with salary (0.375)
* Having **'no major'** is weakly negatively correlated to salary (-0.371)
* **Miles from metropolis** is weakly negatively correlated to salary (-0.298)
* The levels of **industry** and **major** are very weakly correlated with salary (<0.18, except for having 'no major')
* Finance and oil industries have significantly higher salary distributions than other industries, but do not vary significantly with each other
* From the ANOVA results, we see that all categorical features contain at least some levels that vary significantly, however for the **company**  feature this is fairly rare and it is not correlated with salary (<0.003)


# 3. Develop:
## Features:
- I tried a square-root transformation of the target since the distribution was right-skewed
- Because salary did not vary significantly with each individual company, I tried binning the distribution of the mean of each company salary into quartiles, such that each bin varied signficantly from the others.
- I was interested in the max salary of each categorical level, since intuitively I would guess that companies or industries with extremely high paid executives MIGHT pay their employees more. So I **engineered features** that included group statistics. 

## Models:
* Linear regression (there were several weak linear correlations with salary)
* Polynomial regression (perhaps for job type field this will help since the relationship of the groups to salary looks a bit curved)
* Random Forest (some of the relationships were not clearly linear and I also wanted to know feature importance)
* Gradient Boosting (high risk of overfitting, but want to get that MSE down as far as possible)

## Results (MSE):
* The engineered features helped
* Linear regression: 384.409 (the target transformation did not help)
* Polynomial regression: 505.017 using hand-picked features from correlations in EDA
* Random Forest: 390.268
* Gradient Boosting: 

## Feature Importance:


# 4. Deploy:


