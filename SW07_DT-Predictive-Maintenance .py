#!/usr/bin/env python
# coding: utf-8

# # Decision Trees (with GINI) for Predictive Maintenance
# Supplementary Jupyter Notebook for the seminar paper of Lucas Krauter and Richard Mader.
# 
# This Notebook implements the procedure to build a decision-tree based on Gini-index metric.  
# It has been setup to build a Decision-Tree for the predictive-maintance example of the lecture.
# 
# See Homework 3.2: Calculate the Decision Tree for UseCase “Predictive Maintenance” on slide p.77. Do the following steps: 
# 1.	Calculate the Frequency Matrices for the features „Temperatur“, „Druck“ and „Füllstand“
# 2.	Define the Root-node by calculating the GINI-Index for all values of the three features. Define the optimal split-value for the root-node (see slide p.67)
# 3.	Finalize the decision tree by calculation of GINI-Index for the remaining features “Temp.” and “Füllst.” 
# Task: Create and describe the algorithms to automate the calculation of the steps 1. to 3. 
# 

# In[1]:


import platform
import datetime
import numpy as np
import pandas as pd
import graphviz as gv

print(f"This notebook was launched at: {datetime.datetime.now()}")
print()
print("Versions of the used runtime and libraries:")
print(f"- python {platform.python_version()}")
print(f"- pandas {pd.__version__}")
print(f"- numpy {np.__version__}")
print(f"- graphviz {gv.__version__}")


# In[2]:


# The name of the feature that should get predicted
predict_feature = 'Fehler'
# A set of possible values that the feature to predict might have
predict_values = [True, False]

# A set of value-sets on which to base the decision tree
data = pd.DataFrame(np.array([
    [244, 140, 4600, False],
    [200, 130, 4300, False],
    [245, 108, 4100, True],
    [250, 112, 4100, False],
    [200, 107, 4200, False],
    [272, 170, 4400, True],
    [265, 105, 4100, False],
    [248, 138, 4800, True],
    [200, 194, 4500, True],
]), columns = ['Temperatur', 'Druck', 'Füllstand', 'Fehler'])


# ## Datastructures and utilities
# Before we start to build the decision-tree, we first define datastructures that can be used later to represent a decision tree.  
# Further utility functions are defined to render a decision-tree in a graphical, human-understandable way.

# In[3]:


# A list of all features that will be considered when building the decision tree.
# This is any feature except the feature to predict.
input_features = list(filter(lambda f: f != predict_feature, data.columns))

# A question of a decision tree that defines on which feature at which threshold to divide.
#
# If futher distinction between the data is possible, a further question for the values
# below and above the threshold can be provided. Otherwise the tree will contain the 
# prediction result for these cases as generated from the `calculate_prediction` function.
class Decision:
    def __init__(self, feature, threshold, below, above):
        self.feature = feature
        self.threshold = threshold
        self.below = below
        self.above = above


# In[4]:


# Represent a prediction for the "predict_feature" as it
# will be present at each leaves of the decision-tree
class Prediction:
    # Create a prediction by consuming the values, that are
    # predicted at one leaf of the decision-tree.
    def __init__(self, values):
        total = len(values)
        self.props = {
            value: values.count(value) / total
            for value in predict_values
        }

    # If there is only one value with a propability of 100% predicted, then get that value
    def single_value(self):
        single_value = [value for value, prop in self.props.items() if prop == 1.00]
        return single_value[0] if len(single_value) > 0 else None

    # Build a humanreadable string that describes the propability of the
    # occurrence of each predicted value in percent.
    def multi_label(self):
        return ", ".join([
            f"{int(percentage * 100)}% {value}"
            for [value, percentage] in self.props.items()
            if percentage > 0.0
        ])

# Example:
print(f"Prediction: {Prediction([True, False, False]).multi_label()}")


# In[5]:


# Visualize a calculated decision-tree using Graphviz
def render_tree(tree):
    i = 0
    def render_node(dot, tree):
        nonlocal i
        if isinstance(tree, Decision):
            # Render a decision on a feature with its subtrees
            treeId = i
            dot.node(str(treeId), tree.feature)
            i += 1

            dot.edge(str(treeId), str(i), f"<= {tree.threshold}")
            render_node(dot, tree.below)

            dot.edge(str(treeId), str(i), f"> {tree.threshold}")
            render_node(dot, tree.above)

        elif isinstance(tree, Prediction):
            # Render a prediction which can be one single result which
            # has a propability of 100% or multiple weighted values.
            val = tree.single_value()
            if val != None:
                dot.node(str(i), str(val))
            else:
                dot.node(str(i), tree.multi_label())
            i += 1

    dot = gv.Digraph()
    render_node(dot, tree)
    return dot


# ---

# # Gini index
# In the next section we implement the algorithms to calculate the Gini-Index for given value-sets.  
# These metrics are later used to make optimal decisions when choosing the questions for the decision-tree.

# ### Calculate the Gini-index
# Calculate the gini-index (also commonly referred to as Gini-impurity) for a set of predicted values as follows:
# 
# $ 1 - \sum^{j}_{i=1} {p_i}^{2} $
# 
# ${j}$ is the amount of `predict_values`, beeing `len(predict_values)`  
# ${p_i}$ is the propability that the `predict_feature` of a random value from the input value-set is equal to `predict_values[i]`
# 
# This metric describes how homogeneous a set of values is.  
# Datasets with much variation between the values have high index values,  
# while a set of very similar values has a low index.

# In[6]:


def gini_index(values):
    index = 1
    total = len(values)

    # If theres are no values, then this early return
    # prevents a division by zero exception.
    if total == 0: return 0

    for predict_value in predict_values:
        # How many of the values match the predict_value
        count = len(list(filter(lambda val: val == predict_value, values)))
        index -= (count / total) ** 2

    return index

# Examples:
mixed_index = gini_index([False, False, True, True])
print(f"The Gini-index of maximum mixed values is: {mixed_index}")

homogeneous_index = gini_index([True, True])
print(f"The Gini-index of homogeneous values is: {homogeneous_index}")


# ### Calculate the Gini-index for a split on a feature
# Calculate a Gini-index for a split thresold on a feature.  
# It is a weighted average of the gini-index for the values below and above a defined threshold.
# 
# The gini-index therefore describes how well a split for a feature partitions a dataset into two subdatasets as evenly as possible.  
# A gini-index of zero is therefore an ideal split into two equal categories.

# In[7]:


def split_gini_index(data, feature_name, threshold):
    def frq(value_set):
        return len(value_set) / len(data)

    # Collect the values predicted by the value-sets below/above the threshold
    below = data[data[feature_name] <= threshold][predict_feature]
    above = data[data[feature_name] > threshold][predict_feature]

    return frq(below) * gini_index(below) + frq(above) * gini_index(above)

# Examples:
idea_threshold = split_gini_index(pd.DataFrame(np.array([
    [200, False],
    [200, False],
    [200, False],
    [244, True],
    [245, True],
]), columns = ['Temperatur', 'Fehler']), 'Temperatur', 210)
print(f"Gini-index of a split that divides the dataset perfectly: {idea_threshold}")

non_perfect_threshold = split_gini_index(pd.DataFrame(np.array([
    [200, True],
    [200, True],
    [244, True],
    [245, False],
]), columns = ['Temperatur', 'Fehler']), 'Temperatur', 210)
print(f"Gini-index of a non-ideal split: {non_perfect_threshold}")


# ---

# # Choosing the best question to ask
# For each feature we will define a decision rule that partitions the dataset into values above and below a certain threshold.  
# To choose an optimum threshold which provides the best separation, we are using the gini-index as metric.
# 
# For each feature the optimum threshold to consider when asking a question is the one whose gini-index is the lowest.
# 
# Given a dataset, the overall best question to ask is therefore, to partition over the feature whose optimum threshold has the overall lowest gini-index.

# ### Calculate all thresholds which can be considered for a feature.  
# We therefore take all mean values between the numbers into account.
# 
# Additional thresholds below the smallest and above the largest number have been considered in the lecture.  
# They are implemented for completeness but should be unattainable in practice.
# 
# 
# 
# <table style="float: right;">
#     <tr>
#         <td width="100px" style="text-align: left;">Input-values</td>
#         <td width="50px"></td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">200</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">244</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">245</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">248</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">250</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">265</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">272</td>
#         <td width="50px"></td>
#     </tr>
#     <tr>
#         <td width="100px" style="text-align: left;">Thresholds</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">178</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">222</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">244,5</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">246,5</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">249</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">257,5</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">268,5</td>
#         <td colspan="2" width="75px" style="border: black solid 1px; text-align: center;">275,5</td>
#     </tr>
# </table>
# 
#  Example:

# In[8]:


def get_split_values(data, feature_name):
    # All values that are present for the requested feature.
    # They are sorted and do not contain any duplicates.
    col = sorted(list(set(data[feature_name].array)))

    thresholds = []

    # Calculate the mean values between all adjacent values for the feature
    for index in range(len(col) - 1):
        current = col[index]
        next = col[index + 1]
        deriv = (next - current) / 2
        thresholds.append(current + deriv)

    # Add threshold below the smallest value (178 in the example)
    thresholds.insert(0, col[0] - (thresholds[0] - col[0]))

    # Add threshold above the largest value (275.5 in the example)
    thresholds.append(col[-1] + (col[-1] - thresholds[-1]))

    return thresholds

# Example:
get_split_values(data, 'Temperatur')


# ### Find the best feature and threshold
# Given a dataset and a set of features, select the feature whose optimum threshold  
# results in the overall lowest gini-index as described above.
# 
# The calculated feature with its threshold is therefore the most optimum split criterion, to separate the dataset into two subsets,  
# within each of which the most similar values are present for the feature to predict.

# In[9]:


def find_optimal_split(data, features):
    # Find the most optimal threshold for a given feature
    # using the gini-index as metric.
    def get_optimal_feature_split(data, feature_name):
        splits = get_split_values(data, feature_name)

        # Map all possile thresholds to their gini-index
        gini_dict = { split: split_gini_index(data, feature_name, split) for split in splits }

        # Find the threshold with the lowest gini-index
        optimal_split = min(splits, key = lambda split: gini_dict[split])
        return [optimal_split, gini_dict[optimal_split]]

    # Get for each feature the most optimal split with its corresponding gini-index
    splits_with_gini_indices = { feature: get_optimal_feature_split(data, feature) for feature in features }

    best_feature = min(features, key = lambda feature: splits_with_gini_indices[feature][1])
    [split, gini] = splits_with_gini_indices[best_feature]

    return [best_feature, split]

# Example: Calculating the question for the root node
find_optimal_split(data, list(input_features))


# # Build the Decision-Tree
# Since we have already defined comprehensive utility-functions above,  
# the construction of the decision three is now a simple recursive procedure.
# 
# If it is possible to distinglish between the provided value-sets, then we look for the most optimum question.  
# The dataset has to be partitioned using the question and a sub-decision-tree should get build for each of those data-partitions recursivily.

# In[10]:


# Check whether it is possible to make any further distictions between
# the value-sets by asking questions on the features.
def is_data_distinguishable(data, features):
    if len(features) == 0:
        return False

    # Is there only one value-set remaining / are all value-sets equal?
    if len(data.drop_duplicates(features)) == 1:
        return False

    # Do all remaining value-sets predict the same value?
    if len(data[predict_feature].unique()) == 1:
        return False

    return True

def build_tree(data, features):
    if is_data_distinguishable(data, features):
        # Get the best feature to ask a question on and corresponding threshold
        [best_feature, threshold] = find_optimal_split(data, features)

        # Divide the dataset based at the identified optimal threshold into two subdatasets
        below = data[data[best_feature] <= threshold]
        above = data[data[best_feature] > threshold]

        remaining_features = list(filter(lambda f: f != best_feature, features))
        return Decision(best_feature, threshold,
                        build_tree(below, remaining_features),
                        build_tree(above, remaining_features))
    else:
        return Prediction(list(data[predict_feature]))


# In[11]:


tree = build_tree(data, input_features)
render_tree(tree)


# In[12]:


print(f"The execution of this notebook completed at: {datetime.datetime.now()}")

