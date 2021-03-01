#!/usr/bin/env python
# coding: utf-8

# # multiple Linear Regression (mLR) with scikit-learn

# ### Provided by Nora Baitinger, Antonia Wermerskirch, Paul Jaworski 
# Location: DHBW Stuttgart, Date: 2.11.2020
#     
# Extented by H. Völlinger; DHBW; 2.11.2020

# The implementation of mLR is very similar to that of sLR:
# 1. Import all needed packages
# 2. Provide data to work with
# 3. Create and fit regression model with data from previous step
# 4. Check the fitted model for statisfaction
# 5. Apply model for predicitions

# # Step 1: Import all needed dependencies
# 
# numpy - uses numerical mathematics
# 
# IPython - uses display, Math and Latex to for printing the formula
# 
# sklearn - Use/call the LinearRegression module
# 
# sys - version information to pythonImport of libraries

# In[1]:


# For mLR we use the numpy and scikit-learn python library. Ladder provides the LinearRegression module we need.
import numpy as np
from sklearn.linear_model import LinearRegression
from IPython.display import display, Math, Latex
import sys

# check version of numpy an sklearn

# version of numpy:
print("numpy {}".format(np.__version__))
# version of sklearn
import sklearn as sk
print("sklearn {}".format(sk.__version__))
# version of python
# print python version, for some imports this version number is viewed as theirs. 
print("python {}".format(sys.version))


# # Step 2: Provide data and Check data
# Provide the data given in the task

# In[2]:


x = [[1, 2], [3, 3], [2, 2], [4, 3]]
x1 = [1, 2]; x2 = [3, 3]; x3 = [2, 2]; x4 =[4, 3]
y = [3, 4, 4, 6]
x, y = np.array(x), np.array(y)


# ### Print the data and check it

# In[3]:


# You can print x and y to check the correctness:
print('(x11,x21) looks like:     ', x1)
print('(x12,x22) looks like:     ', x2)
print('(x13,x23) looks like:     ', x3)
print('(x14,x24) looks like:     ', x4)
print('(y1,y2,y3,y4) looks like:', y)


# # Step 3: Create and  fit the model
# Now we need to create and fit the regression model. To do that we need to create an instance of LinearRegression and call the .fit()-method on it. 

# In[4]:


#The variable fitted_model represents in this case the fitted model according to the given data.
fitted_model = LinearRegression().fit(x, y)


# # Step 4: Get needed results

# In[5]:


# 𝑅² can be obtained by calling the score() method on the model
r_square = fitted_model.score(x, y) 
print('coefficient of determination:', r_square)
# The values of the estimators of regression coefficients can be obtained by:
print('intercept:', fitted_model.intercept_)
print('coefficients:', fitted_model.coef_)
# intercept holds the bias b0, coef holds b1 and b2 in an array


# ### Visualization of the Result
# 
# Here I imported an hand-done sketch.
# 
# This will be replaced later by a python graphics

# In[6]:


from IPython.display import Image
Image('Images/Homework_H5_4-mLR_plane-Sketch.jpg')


# # Step 5: Predict response using the model

# The process of predicting values in mLR is analogous to that in sLR. The prediction is performed by predict() :

# In[7]:


prediction = fitted_model.predict(x)
print('predicted response:', prediction, sep='\n')


# Another way to predict y-values is to multiply each column of the input with the appropriate weight, summing the results up and adding the intercept to the sum.

# In[8]:


prediction = fitted_model.intercept_ + np.sum(fitted_model.coef_ * x, axis=1)
print('predicted response:', prediction, sep='\n')


# In[9]:


# print current date and time
import time
print("date+time",time.strftime("%d.%m.%Y %H:%M:%S"))
print ("***** End of Homework H5.4c *******")

