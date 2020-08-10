#!/usr/bin/env python
# coding: utf-8

# # # Simple Linear Regression With scikit-learn
# 
# Powered by: Dr. Hermann Völlinger, DHBW Stuttgart(Germany); July 2020
#     
# Following ideas from: 
# "Linear Regression in Python" by Mirko Stojiljkovic, 28.4.2020 
# (see details: https://realpython.com/linear-regression-in-python/#what-is-regression)     
#     
# Let’s start with the simplest case, which is simple linear regression.
# There are five basic steps when you’re implementing linear regression:
#     
# 1. Import the packages and classes you need.
# 2. Provide data to work with and eventually do appropriate transformations.
# 3. Create a regression model and fit it with existing data.
# 4. Check the results of model fitting to know whether the model is satisfactory.
# 5. Apply the model for predictions.
# These steps are more or less general for most of the regression approaches and implementations.

# # Step 1: Import packages and classes
# The first step is to import the package numpy and the class LinearRegression from sklearn.linear_model:

# In[1]:


# Step 1: Import packages and classes

import numpy as np
from sklearn.linear_model import LinearRegression

# import time module
import time


# Now, you have all the functionalities you need to implement linear regression.
# 
# The fundamental data type of NumPy is the array type called numpy.ndarray. The rest of this article uses the term array to refer to instances of the type numpy.ndarray.
# 
# The class sklearn.linear_model.LinearRegression will be used to perform linear and polynomial regression and make predictions accordingly.

# # Step 2: Provide data
# The second step is defining data to work with. The inputs (regressors, 𝑥) and output (predictor, 𝑦) should be arrays
# (the instances of the class numpy.ndarray) or similar objects. This is the simplest way of providing data for regression:

# In[2]:


# Step 2: Provide data

x = np.array([ 5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([ 5, 20, 14, 32, 22, 38])


# Now, you have two arrays: the input x and output y. You should call .reshape() on x because this array is required to be two-dimensional, or to be more precise, to have one column and as many rows as necessary. That’s exactly what the argument (-1, 1) of .reshape() specifies.

# In[3]:


print ("This is how x and y look now:")
print("x=",x)
print("y=",y)


# As you can see, x has two dimensions, and x.shape is (6, 1), while y has a single dimension, and y.shape is (6,).

# # Step 3: Create a model and fit it
# 
# The next step is to create a linear regression model and fit it using the existing data.
# Let’s create an instance of the class LinearRegression, which will represent the regression model:

# In[4]:


model = LinearRegression()


# This statement creates the variable model as the instance of LinearRegression. You can provide several optional
# parameters to LinearRegression:
#     
# ----> fit_intercept is a Boolean (True by default) that decides whether to calculate the intercept 𝑏₀ (True) or consider
# it equal to zero (False).
# 
# ----> normalize is a Boolean (False by default) that decides whether to normalize the input variables (True) or not
# (False).
# 
# ----> copy_X is a Boolean (True by default) that decides whether to copy (True) or overwrite the input variables (False).
# 
# ----> n_jobs is an integer or None (default) and represents the number of jobs used in parallel computation. None
# usually means one job and -1 to use all processors.
# 
# This example uses the default values of all parameters.
# 
# It’s time to start using the model. First, you need to call .fit() on model:
#     
# 

# In[5]:


model.fit(x, y)


# With .fit(), you calculate the optimal values of the weights 𝑏₀ and 𝑏₁, using the existing input and output (x and y) as
# the arguments. In other words, .fit() fits the model. It returns self, which is the variable model itself. That’s why you
# can replace the last two statements with this one:

# In[6]:


# model = LinearRegression().fit(x, y)


# This statement does the same thing as the previous two. It’s just shorter.

# # Step 4: Get results
# 
# Once you have your model fitted, you can get the results to check whether the model works satisfactorily and
# interpret it.
# 
# You can obtain the coefficient of determination (𝑅²) with .score() called on model:

# In[7]:


r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)


# When you’re applying .score(), the arguments are also the predictor x and regressor y, and the return value is 𝑅².
# 
# The attributes of model are .intercept_, which represents the coefficient,𝑏₀ and .coef_, which represents 𝑏₁:

# In[8]:


print('intercept:', model.intercept_)
print('slope:', model.coef_)


# The code above illustrates how to get 𝑏₀ and 𝑏₁. You can notice that .intercept_ is a scalar, while .coef_ is an array.
# 
# The value 𝑏₀ = 5.63 (approximately) illustrates that your model predicts the response 5.63 when 𝑥 is zero. The value 𝑏₁
# = 0.54 means that the predicted response rises by 0.54 when 𝑥 is increased by one.
# 
# You should notice that you can provide y as a two-dimensional array as well. In this case, you’ll get a similar result.
# This is how it might look:

# In[9]:


new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)


# As you can see, this example is very similar to the previous one, but in this case, .intercept_ is a one-dimensional array with the single element 𝑏₀, and .coef_ is a two-dimensional array with the single element 𝑏₁.

# # Step 5: Predict response
# 
# Once there is a satisfactory model, you can use it for predictions with either existing or new data.
# 
# To obtain the predicted response, use .predict():

# In[10]:


y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')


# When applying .predict(), you pass the regressor as the argument and get the corresponding predicted response.
# 
# This is a nearly identical way to predict the response:

# In[11]:


y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')


# In this case, you multiply each element of x with model.coef_ and add model.intercept_ to the product.
# 
# The output here differs from the previous example only in dimensions. The predicted response is now a twodimensional
# array, while in the previous case, it had one dimension.
# 
# If you reduce the number of dimensions of x to one, these two approaches will yield the same result. You can do this
# by replacing x with x.reshape(-1), x.flatten(), or x.ravel() when multiplying it with model.coef_.
# 
# In practice, regression models are o􀁸en applied for forecasts. This means that you can use fitted models to calculate
# the outputs based on some other, new inputs:

# x_new = np.arange(5).reshape((-1, 1))
# print(x_new)
# y_new = model.predict(x_new)
# print(y_new)

# Here .predict() is applied to the new regressor x_new and yields the response y_new. This example conveniently uses
# arange() from numpy to generate an array with the elements from 0 (inclusive) to 5 (exclusive), that is 0, 1, 2, 3, and 4.
# 
# You can find more information about LinearRegression on the official documentation page.

# In[12]:


# print current date and time
print("date",time.strftime("%d.%m.%Y %H:%M:%S"))
print ("end")

