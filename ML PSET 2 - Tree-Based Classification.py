#!/usr/bin/env python
# coding: utf-8

# In[142]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from six import StringIO
from sklearn.tree import export_graphviz
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_text
from IPython.display import Image  
import pydotplus


# ## CLASSIFICATION TREES

# ### QUESTION: 1

# In[143]:


def data_clean ():
    path = '/Users/ivnagpal/Desktop/MACS 33002(1)-Machine Learning/ps2-ivnagpal/data'
    os.chdir(path)
    df = pd.read_csv (r'anes_pilot_2016.csv')
    df['ftobama'].mask(df['ftobama'] > 100, np.nan, inplace=True)
    df['fttrump'].mask(df['fttrump'] > 100, np.nan, inplace=True)
    dummy = pd.get_dummies (df['pid3']).rename(columns ={1:'democrat'})
    dummy_dem = dummy['democrat']
    df = pd.concat ([df[['ftobama','fttrump']],dummy_dem],axis = 1)
    df['democrat'] = df['democrat'].astype('float')
    df = df.dropna()
    return df

df = data_clean() #Dimensions: 1195 x 3
df.head()


# In[144]:


def feeling_hist ():
    plt.hist([df['ftobama'],df['fttrump']], bins= 50, range=[0,100], label=['Obama', 'Trump'], 
             color = ['#175bbb','#8d1430'],alpha = .66, histtype= 'stepfilled')
    plt.legend(loc='upper right')
    plt.xlabel('Rating on Feeling Themometer')
    plt.ylabel('Count')
    plt.title('Histogram of Trump and Obama Feeling Themometer')
    plt.xlim(0,100)
    plt.show()

feeling_hist ()


# In[145]:


def demo_hist ():
    vcount = df['democrat'].value_counts()
    vcount = vcount.rename(index={0: 'Not Democrat',1:'Democrat'}).to_frame ()
    vcount.reset_index(level=0, inplace=True)
    plt.bar (vcount['index'],vcount['democrat'],color=['#8d1430', '#175bbb'])
    plt.xlabel('Party Id')
    plt.ylabel('Count')
    plt.title('Number of Democrats and Non Democrats in Dataset')
    plt.show()

demo_hist()


# ### QUESTION: 2

# In[146]:


# Splitting the data frame into features and labels
#Freatures
x = df[['ftobama', 'fttrump']]
#Labels
y = df[['democrat']]

#Test Training Split (75-25 per instructions)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state= 4)

print ('Number of training samples:', len (x_train), 'Observations')
print ('Number of test samples:', len (x_test), 'Observations')


# ### QUESTION: 3

# In[147]:


# Create Decision Tree
tree = DecisionTreeClassifier (random_state= 4, min_impurity_decrease = 0.005)
# Train Decision Tree Classifer
tree = tree.fit(x_train,y_train)


# ### QUESTION: 4

# In[148]:


#Summary of fitted decision tree
print ('Total Number of Nodes:', tree.tree_.node_count)
print ('Total Number of Leaves:', tree.get_n_leaves())
print ('Maximal Depth of Tree',tree.tree_.max_depth)
print ('Training Data Accuracy Score:', tree.score(x_train, y_train))
print ('Basic Tree Structure:\n',export_text(tree,feature_names= ['ftobama','fttrump'],show_weights = True))


# The simple decision tree contains 9 nodes, 5 which are terminal. The root node splits the training dataset based on whether the Obama Feeling Themometer (ftobama) has a value greater than 62.50. The maximum depth of the tree is 3, where we first get a subset of the data with a ftobama > 62.5. From that population subset, we obtain a population subset where ftobama <= 86.5. Finally, from that subset of the population, we obtain a population subset where the Trump Feeling Themometer (ftttrup <= 59.0). To determine the training error rate, we obtain the accuracy score, which is approximately 0.818. This accuracy score roughly correlates to 0.1819 Training Error. While we don't want too low ofa training error rate to avoid overfitting, 0.1819, at first glance, appears too low.

# In[149]:


#Gini Weighted Impurity Value

def leaf_impurities ():
    imp = tree.tree_.impurity
    indices = (2,3,6,7,8)
    imp = [imp[i] for i in indices]
    imp = [round(i, 4) for i in imp]
    return imp

leaf_impurity = leaf_impurities ()

def gini_weightavg(impurities): 
    w = [np.divide(381,896),np.divide(128,896),np.divide(146,896),np.divide(24,896),np.divide(217,896)]
    w_gini = np.average(impurities, weights = w)
    return w_gini

print ('Gini Leaf Impurities:',leaf_impurity)
print ('Weighted Avearge of Gini Leaf Impurities:', round (gini_weightavg (leaf_impurity),4))





# We calculate leaf note puritiy using gini Values. The higher the gini value, the greater the probability of misclassificaton, and thus less pure. Given that each leaf node does not have an equal sample size, we add weights to get the gini average. The weighted average of the the leaf ginis is approximately 0.2617.  

# ### QUESTION: 5

# In[150]:


dt_feature_names = list(x_train.columns)
dot_data = StringIO()
export_graphviz(tree , out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names=dt_feature_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Trump Obama Feeling Themometer Decision Tree')
Image(graph.create_png())


# The plot of the decision tree provides us with significant insight into the order in which subpopulations were split, their respective attributes, and gini values before and after passing a node. By calculating change in gini values before and after passing the node, we can gauge the effectiveness of the splitting. We note that the fttrump values were only considered for individuals with ftobama value which were greater than 62.5 and less than or equal to 86.5. We also note that the gini values for the leaf nodes range from 0.104 to 0.479. The subpopulation with ftobama less than or equal to 34.5 had the greatest purity, with 360 out of 381 individuals not being a democrat. Conversely, the subpopulation with ftobama values greater than 62.5 and less than or equal to 86.5 and fttrump values less than or equal to 59 was the most "polluted" with 88 out of 146 being a democrat. 

# ### QUESTION: 6

# In[151]:


#Predict using Test Data
y_pred = tree.predict(x_test)


# In[152]:


#Evaluation the Tree's performance on the Test Data.

#Accuracy Score
print("Accuracy:", round (metrics.accuracy_score(y_test, y_pred),4))

#Confusion Matrix
print('\nConfusion Matrix:\n',confusion_matrix(y_test, y_pred))


# We tested the decision tree model on 25% of the data set (299 observations) and found that it correctly predicted whether an individual would be a democrat or not a democrat approximately 75.92% of the time. Based on the confusion matrix, we calculate a precision rate of 0.754, sensitivity rate of .8364, and a specificity rate of 0.664. The relatively low specificity rate might be a cause for concern. While we might want to hyper-tune the model’s parameters to prevent false “democrat” classification, we don’t want to compromise the relatively high sensitivity rate.

# ## CLASSIFICATION TREES

# ### QUESTION: 7

# In[153]:


bag = BaggingClassifier(DecisionTreeClassifier(),n_estimators = 1000, oob_score= True,random_state = 4)
bag.fit(x_train,y_train)
decision_function = bag.oob_decision_function_
print ('\nTraining Data Accuracy Score:', bag.score(x_train, y_train))
print ("Decision Function Array:\n", decision_function)


# Our bagging tree classifier takes bootstrapped samples of the test data to generate 1000 decision trees. It then determines whether the observation is democrat based on the most popular outcome when running the attributes through all randomly generated decision tree. We obtain a Training Data Accuracy score of 0.93415 which is 16.5% higher than the accuracy score for the simple decision tree model. The fact that the Training Data Accuracy score is relatively close to 1 suggests that the model might be overfitted. Printed above is brief prieve of the decision function array. For each observation, when the value in the first column is larger than the second, the bagging tree classifies the observation as “Not a Democrat.” Conversely, when the observation in the second column is larger than the first, the bagging tree classifies the observation as a "Democrat.”

# ### QUESTION: 8

# In[155]:


#Predict using test data
y_bagp = bag.predict(x_test)
y_prob = bag.predict_proba(x_test)

#Printing the first 10 predicitions of the test data and probabilities underlying the decision.
print ('Test Prediction Values:\n', y_bagp[0:10])
print ('\nTest Probability Values:\n', y_prob[0:10] )


# In[156]:


#Accuracy Score
print("Accuracy:", round (metrics.accuracy_score(y_test, y_bagp),4))

#Confusion Matrix
print('\nConfusion Matrix:\n',confusion_matrix(y_test, y_bagp))


# We again tested the bagging tree on 25% of the dataset and found that it accurately predicted whether an indivdual is a democrat 74.25% of the time, which is approximately 2.2% less accurate than the simple decision tree. Based on the confusion matrix, we calculate a precision rate of .7704, sensitivity rate of 0.8011, and a specificity rate of 0.6581. In contrast to simple decision tree model,the bagging tree's precision rate increased by 1.022%, sensitivity rate decreased by 4.22%, and specificity rate decreased by 0.89%.

# ### QUESTION: 9

# In[159]:


importance = np.mean([tree.feature_importances_ for tree in bag.estimators_], axis=0)
df_imp = pd.DataFrame(data=importance,index=['ftobama', 'fttrump'], columns=['Importance'])
df_imp = df_imp.reset_index()
plt.bar (df_imp['index'],df_imp['Importance'],color=['#175bbb','#8d1430'])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Importance of Bagging Tree Features')
plt.show()


# In[131]:


print ('FTObama Mean Decrease Impurity:',round (importance[0],4))
print ('FTTrump Mean Decrease Impurity:', round (importance[1],4))


# We gauge feature's relative importance by looking at how much each feature decreases gini impuritiy. The Bagging Model randomly creates a 1000 trees, each with internal nodes which further split the population based on some criteria to increase sample purity. To figure out what feature was more important, we looked at how much each feature reduced impurity for each of the 1000 trees and took the average of the reduction. We find that ftobama had a mean decrease impurity of 0.7024 and fttrump had a mean decrease impurity of 0.2976. This suggests that the Obama Feeling Themometer does almost a 2.36X better job decreasing impurities in our Bagging Tree Model, and is thus a more important feature for classification.

# ### Question 10

# We utilized both a simple decision tree and bagging tree classifier to guage whether we can predict whether someone is a democrat based on their feelings towards Obama and Trump. Our dataset, which contained 1195 observations was split 75/25 for training and testing. Surprisingly, the bagging tree model was 2.2% less accurate than simple decision tree, making an accurate prediction in the test data 74.25% of the time. One possible reason for the bagging tree's poor performance was its overfitting. The bagging tree obtained a training score of 0.93415, which suggests a poor ability to generalize. Another shocking finding was that the the bagging tree had a lower sensitivity and specificity rate than the simple decision tree, while having a higher precision rate. We again hypothesize that this could be attributable to model overfitting. One way to prevent overfitting would be to add resitrictions to the tree's growth so it does not perfectly fit the data. For example, we can add a restriction which will prevent the splitting of the node if the reduction in gini is below a certain threshold. 

# ### Tree Pruning

# In[160]:


#Train Simple Decision Tree Model with CV of 10
tree_cv_scores = cross_val_score(tree, x_train, y_train, cv=10)

print('Simple Decision Tree cv_scores mean:{}'.format(np.mean(tree_cv_scores))) 
print('Simple Decision Tree standard deviation:{}'.format(np.std(tree_cv_scores ))) 


# By pruning the simple decision tree using cross validation, we increase the accuracy by approximately 9.42% to 79.22% accuracy
