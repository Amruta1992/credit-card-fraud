#!/usr/bin/env python
# coding: utf-8

# #             Project    -    CREDIT CARD FRAUD DETECTION
# 

# ## Problem Statement :
# The problem at hand involves developing a machine learning model that can accurately detect fraudulent credit card transactions from a given dataset. Fraudulent transactions can lead to significant financial losses and a negative impact on customers' trust in the financial system. Therefore, building an effective fraud detection system is crucial to identifying and preventing such incidents in a timely manner.

# ## Objective:
# The primary objective of this project is to create a credit card fraud detection model using machine learning techniques. The model should be able to accurately classify transactions as either legitimate (non-fraudulent) or fraudulent based on the features provided in the dataset.

# ## Step 1: Import Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy import stats
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings("ignore")


# ## Step 2: Load Data

# In[2]:


df_card_data = pd.read_csv("C:\\Users\\Sneha & Sankalp\\Desktop\\IMARTICUS PGA12\\Aditional project\\creditcard.csv")
df_card_data.head()


# ## Step 3: Explore Data

# In[3]:


# data information

df_card_data.info()


# In[4]:


df_card_data.shape


# In[5]:


df_card_data.describe()


# In[6]:


# check for missing values
missing_values = df_card_data.isnull().sum()
print(missing_values)


# #### there is no missing values in the dataset
# 

# In[7]:


summary = df_card_data.drop(['Class'], axis=1).describe()
print(summary)


# In[8]:


class_counts = df_card_data['Class'].value_counts()
print(class_counts)


# In[9]:


# Plot the class counts
plt.bar(class_counts.index, class_counts.values, color=['blue', 'red'])
plt.xticks(class_counts.index, ['Non-Fraudulent', 'Fraudulent'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Imbalance')
plt.show()


# In[10]:


# making Hour feature from Time feature
df_card_data["Hour"] = df_card_data["Time"].apply(lambda x: np.ceil(float(x)/3600) % 24)
df_card_data["Hour"] = df_card_data["Hour"].astype("int")


# In[11]:


# Distribution of Fraud and Normal Transactions

px.pie(df_card_data, names="Class", title="Distribution of Fraud and Normal Transactions:(Normal:0 | Fraud:1)", 
       color_discrete_sequence=['#1E90FF', '#FF4500'],
       template="plotly_dark", width=400, height=300)


# In[12]:


# correlation heatmap:
plt.figure(figsize = (8,8))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = df_card_data.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()


# In[13]:


# time vs fraud

# Transaction count with non frauds over time(Hour)
count_0 = df_card_data[df_card_data['Class']==0].groupby('Hour').count()["Class"]

# Transaction count with frauds over time(Hour)
count_1 = df_card_data[df_card_data['Class']==1].groupby('Hour').count()["Class"]

# Concatenate
counts_df_card_data = pd.concat([count_0, count_1], axis=1, keys=["Class 0", "Class 1"]).fillna(0)
df_counts = counts_df_card_data.reset_index()


# In[14]:


# Create subplots for Transaction Count with Non-Frauds and Frauds over time (Hour)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Plot Transaction Count with Non-Frauds over time
axes[0].plot(df_counts['Hour'], df_counts['Class 0'], label='Non-Fraudulent', color='blue')
axes[0].set_xlabel('Hour of the Day')
axes[0].set_ylabel('Transaction Count')
axes[0].set_title('Transaction Count with Non-Frauds over Time (Hour)')
axes[0].legend()

# Plot Transaction Count with Frauds over time
axes[1].plot(df_counts['Hour'], df_counts['Class 1'], label='Fraudulent', color='red')
axes[1].set_xlabel('Hour of the Day')
axes[1].set_ylabel('Transaction Count')
axes[1].set_title('Transaction Count with Frauds over Time (Hour)')
axes[1].legend()

plt.tight_layout()
plt.show()


# ## Step 4: Data Preprocessing

# In[15]:


# Separate features (X) and target (y)
X = df_card_data.drop('Class', axis=1)
y = df_card_data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## Step 5: Build ,Train And Evaluation Of Models

# In[17]:


# model building

knn=KNeighborsClassifier()
svc=SVC()
nb=GaussianNB()
dtc=DecisionTreeClassifier()
rfc=RandomForestClassifier()
lr=LogisticRegression()


models = [knn, svc, nb, dtc, rfc,lr]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(type(model).__name__, "Model Test Accuracy Score is: ", accuracy_score(y_test, y_pred))
    print(type(model).__name__, "Model Test F1 Score is: ", f1_score(y_test, y_pred))
    print(type(model).__name__,"Mean_absolute_error: ",mean_absolute_error(y_test, y_pred))
    print(type(model).__name__,"Mean_squared_error: ",mean_squared_error(y_test, y_pred))
    print(type(model).__name__,"Root_mean_squared_error: ",np.sqrt(mean_squared_error(y_test, y_pred)))
    print(type(model).__name__,"R2_score: ",r2_score(y_test, y_pred))
    print(type(model).__name__,"Classification_report: \n",classification_report(y_test, y_pred))
    fig=px.imshow(confusion_matrix(y_test, y_pred),color_continuous_scale="Viridis",title=type(model).__name__,
                  width=400,height=400,labels=dict(x="Predicted", y="Actual", color="Counts"),template="plotly_dark")
    fig.show()


# In[22]:


# comparing all model accuracy:   

models = [knn, svc, nb, dtc, rfc,lr]

fig=px.bar(x=[type(model).__name__ for model in models], y=[accuracy_score(y_test, model.predict(X_test)) for model in models],
         color=[accuracy_score(y_test, model.predict(X_test)) for model in models], color_continuous_scale="Viridis",
            title="Model Comparison", labels=dict(x="Model", y="Accuracy"), template="plotly_dark", width=800,
            height=600,text=[accuracy_score(y_test, model.predict(X_test)) for model in models])
            #fig.update_layout(yaxis_range=[0.97,0.99]) 
fig.show()


# ## FEATURE IMPORTANCE
# 

# In[24]:


# there is no feature importance attribute in SVC and Logistic regression

models = [svc,dtc, rfc,lr]

target = 'Class'
predictors = ['Time',"Hour", 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',      'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',      'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',      'Amount']


# In[27]:


for model in models:
    if model == svc or model == lr:
        continue
    print(type(model).__name__, "Model Feature Importance: \n")
    tmp = pd.DataFrame({'Feature': predictors, 'Feature importance': model.feature_importances_})
    tmp = tmp.sort_values(by='Feature importance',ascending=False)
    plt.figure(figsize = (7,4))
    plt.title('Features importance of '+type(model).__name__,fontsize=14)
    s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
    s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.show()


# ## PCA

# In[28]:


from sklearn.decomposition import PCA


# In[29]:


pca=PCA(n_components=2)
X_reduced=pca.fit_transform(X)


# In[30]:


#visualization of reduced data:
sns.set_style("darkgrid")
pca_data=pd.DataFrame(X_reduced,columns=["p1","p2"])
pca_data["target"]=y
sns.scatterplot(x="p1",y="p2",hue="target",data=pca_data)
plt.title("PCA: 2 Component")


# ## AUC-ROC CURVE

# In[31]:


from sklearn.metrics import roc_curve, auc


# In[34]:


models = [dtc, rfc,lr]

for model in models:
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=type(model).__name__)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve "+type(model).__name__+"Model")
    plt.legend()
    plt.show()
    print(type(model).__name__, "Model AUC Score is: ", auc(fpr, tpr))


# ##  Step 6 - CONCLUSION

# After investigating the data, checking for data unbalancing, visualizing the features and understanding the relationship between different features. Then investigated six predictive models. The data was split in 2 parts, a train set and a test set.
# 
# Then evaluated the performance of the models using the AUC-ROC curve and other metrics. The best results were obtained with LogisticRegression Model,with an AUC score of 0.97 and a accuracy of 0.9991.

# ## Model Deployment

# In[38]:


import numpy as np

# Prepare the new input features
new_sample = np.array([[0,-0.5, 0.8, -1.2, 2.5, -0.3, 0.1, -0.6, 0.7, -0.2, 0.3, -0.9, -1.5,
                        0.2, -0.4, -0.8, 0.3, -1.2, -0.5, 0.6, 0.8, -0.3, 0.1, 0.2, 0.4,
                        -0.1, 0.5, -0.4, 0.2, 150.0, 15]])


# In[40]:


# Scale the new input features using the same StandardScaler
new_sample_scaled = scaler.transform(new_sample)

# Make a prediction using the trained Logistic Regression model
prediction = lr.predict(new_sample_scaled)

if prediction[0] == 0:
    print("Non-Fraudulent Transaction")
else:
    print("Fraudulent Transaction")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




