import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def df_clean():   
    path = '/Users/ivnagpal/Desktop/Enron Final Project/MACS31300_Final_Project'
    os.chdir(path)
    df = pd.read_csv (r'enron.csv')
    df = df.drop(columns=['email_address','position'])
    df = df.fillna(df.mean())
    return df
    
df = df_clean()


# Splitting the data frame into features and labels

#Freatures
x = df.drop (columns = ['insider','POI'])

#Labels
y = df[['POI']]

#Test Training Split (80-20 per instructions)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state= 4)
print ('Number of training samples:', len (x_train), 'Observations')
print ('Number of test samples:', len (x_test), 'Observations')

#Predict using Test Data
bag = BaggingClassifier(DecisionTreeClassifier(),n_estimators = 10, oob_score= True,random_state = 4)
bag.fit(x_train,y_train)
decision_function = bag.oob_decision_function_
print ('\nTraining Data Accuracy Score:', bag.score(x_train, y_train))
print ("Decision Function Array:\n", decision_function)

#Predict using test data
y_bagp = bag.predict(x_test)
y_prob = bag.predict_proba(x_test)

#Printing the first 10 predicitions of the test data and probabilities underlying the decision.
print ('Test Prediction Values:\n', y_bagp[0:10])
print ('\nTest Probability Values:\n', y_prob[0:10] )

#Accuracy Score
print("Accuracy:", round (metrics.accuracy_score(y_test, y_bagp),4))

#Confusion Matrix
print('\nConfusion Matrix:\n',confusion_matrix(y_test, y_bagp))

importance = np.mean([tree.feature_importances_ for tree in bag.estimators_], axis=0)
df_imp = pd.DataFrame(data=importance,index= list (x_train.columns), columns=['Importance'])
df_imp = df_imp.reset_index()
plt.bar (df_imp['index'],df_imp['Importance'])
plt.xticks(rotation = 90, fontsize=8)
plt.xlabel('FEATURE',fontsize = 12)
plt.ylabel('IMPORTANCE',fontsize = 12)
plt.title('Importance of Bagging Tree Features')
plt.show()

bag_cv_scores = cross_val_score(bag, x_train, y_train, cv=10)

print('Simple Decision Tree cv_scores mean:{}'.format(np.mean(bag_cv_scores))) 