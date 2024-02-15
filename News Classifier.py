#!/usr/bin/env python
# coding: utf-8

# In[1]:


## News Classifier
from bs4 import BeautifulSoup
import requests


# In[2]:


from datetime import date
today=date.today()

d=today.strftime("%m-%d-%y")
print("date =",d)


# In[3]:


bbc_world="https://www.bbc.com/news/world-middle-east-68274757".format(d)
response=requests.get(bbc_world)
soup=BeautifulSoup(response.content,'html.parser')


# In[4]:


bbc_world


# In[5]:


soup


# In[6]:


## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))


# In[7]:


## Getting the article
bbc_world=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_world +=news.text.strip()


# In[8]:


## Getting World news
bbc_world_1="https://www.bbc.com/news/world-australia-68271324".format(d)
response=requests.get(bbc_world_1)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_world_1=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_world_1 +=news.text.strip()


# In[9]:


## Getting World news
bbc_world_2="https://www.bbc.com/news/world-africa-68283799".format(d)
response=requests.get(bbc_world_2)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_world_2=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_world_2 +=news.text.strip()


# In[10]:


## Getting World news
bbc_world_3="https://www.bbc.com/news/world-australia-67364463".format(d)
response=requests.get(bbc_world_3)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_world_3=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_world_3 +=news.text.strip()


# In[11]:


## Getting Climate News
bbc_climate="https://www.bbc.com/news/science-environment-68254027".format(d)
response=requests.get(bbc_climate)
soup=BeautifulSoup(response.content,'html.parser')
bbc_climate


# In[12]:


## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))


# In[13]:


## Getting the article
bbc_climate=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_climate +=news.text.strip()


# In[14]:


## Getting Climate News
bbc_climate_1="https://www.bbc.com/weather/features/68129796".format(d)
response=requests.get(bbc_climate_1)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_climate_1=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_climate_1 +=news.text.strip()


# In[15]:


## Getting Climate News
bbc_climate_2="https://www.bbc.com/news/science-environment-65754296".format(d)
response=requests.get(bbc_climate_2)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_climate_2=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_climate_2 +=news.text.strip()


# In[16]:


## Getting Climate News
bbc_climate_3="https://www.bbc.com/news/world-australia-68281756".format(d)
response=requests.get(bbc_climate_3)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_climate_3=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_climate_3 +=news.text.strip()


# In[17]:


## Getting Politics News
bbc_politics="https://www.bbc.com/news/world-us-canada-68277167".format(d)
response=requests.get(bbc_politics)
soup=BeautifulSoup(response.content,'html.parser')
bbc_politics


# In[18]:


## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))


# In[19]:


## Getting the article
bbc_politics=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_politics +=news.text.strip()


# In[20]:


## Getting Politics News
bbc_politics_1="https://www.bbc.com/news/world-asia-68271462".format(d)
response=requests.get(bbc_politics_1)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_politics_1=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_politics_1 +=news.text.strip()


# In[21]:


## Getting Politics News
bbc_politics_2="https://www.bbc.com/news/world-us-canada-68280588".format(d)
response=requests.get(bbc_politics_2)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page

for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_politics_2=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_politics_2 +=news.text.strip()


# In[22]:


## Getting Politics News
bbc_politics_3="https://www.bbc.com/news/world-asia-china-68213161".format(d)
response=requests.get(bbc_politics_3)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_politics_3=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_politics_3 +=news.text.strip()


# In[23]:


bbc_games="https://www.bbc.com/news/world-asia-china-68236902".format(d)
response=requests.get(bbc_games)
soup=BeautifulSoup(response.content,'html.parser')
bbc_games


# In[24]:


## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))


# In[25]:


## Getting the article
bbc_games=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_games +=news.text.strip()


# In[26]:


## Getting Sports News
bbc_games_1="https://www.bbc.com/sport/tennis/68287273".format(d)
response=requests.get(bbc_games_1)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_games_1=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_games_1 +=news.text.strip()


# In[27]:


## Getting Sports News
bbc_games_2="https://www.bbc.com/sport/formula1/68270056".format(d)
response=requests.get(bbc_games_2)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_games_2=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_games_2 +=news.text.strip()


# In[28]:


## Getting Sports News
bbc_games_3="https://www.bbc.com/sport/africa/68282325".format(d)
response=requests.get(bbc_games_3)
soup=BeautifulSoup(response.content,'html.parser')
## Getting headings of that particular page
for headings in soup.findAll("h2"):
    print("Headline : {}".format(headings.text))
## Getting the article
bbc_games_3=""
for news in soup.findAll('article',{'class':"ssrcss-pv1rh6-ArticleWrapper e1nh2i2l5"}):
    bbc_games_3 +=news.text.strip()


# In[29]:


## Converting to dataframe
import pandas as pd

bbc_world = bbc_world
bbc_world_1 = bbc_world_1
bbc_world_2 = bbc_world_2
bbc_world_3 = bbc_world_3
bbc_climate = bbc_climate
bbc_climate_1 = bbc_climate_1
bbc_climate_2 = bbc_climate_2
bbc_climate_3 = bbc_climate_3
bbc_games = bbc_games
bbc_games_1 = bbc_games_1
bbc_games_2 = bbc_games_2
bbc_games_3 = bbc_games_3
bbc_politics = bbc_politics
bbc_politics_1 = bbc_politics_1
bbc_politics_2 = bbc_politics_2
bbc_politics_3 = bbc_politics_3

# Create dictionary
data = {
    'News_headline': ['bbc_world','bbc_world_1','bbc_world_2','bbc_world_3', 'bbc_climate','bbc_climate_1','bbc_climate_2','bbc_climate_3',
                      'bbc_games','bbc_games_1','bbc_games_2','bbc_games_3', 'bbc_politics','bbc_politics_1','bbc_politics_2','bbc_politics_3'],
    'Text': [bbc_world,bbc_world_1,bbc_world_2,bbc_world_3, bbc_climate,bbc_climate_1,bbc_climate_2,bbc_climate_3
             , bbc_games,bbc_games_1,bbc_games_2,bbc_games_3, bbc_politics,bbc_politics_1,bbc_politics_2,bbc_politics_3]
}

# Convert to DataFrame
df = pd.DataFrame(data)

df


# In[2]:


df


# In[3]:


# Randomly shuffle the rows of the DataFrame
df = df.sample(frac=1).reset_index(drop=True)
df


# In[4]:


## Data Pre-processing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Function to clean and preprocess the text
def clean_and_preprocess(text):
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")

    # Remove non-alphanumeric characters and extra whitespaces
    clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Tokenization and removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(clean_text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text## Applying the cleaning technique
df['Cleaned_Text'] = df['Text'].apply(clean_and_preprocess)


# In[5]:


df


# In[6]:


df_1 = df.drop(columns=['Text']).copy()


# In[7]:


import re

# Remove 'bbc', '-' and any trailing digits
df_1['News_headline'] = df_1['News_headline'].apply(lambda x: re.sub('bbc|-|\d|_', '', x))
df_1


# In[8]:


from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
df_1['News_headline_label']=label.fit_transform(df_1['News_headline'])


# In[9]:


df_1


# In[10]:


# Splitting Dataset into Training and Testing
Input=df_1['Cleaned_Text']
Output=df_1['News_headline_label']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(Input,Output,test_size=0.4,random_state=0)


# In[11]:


#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
tfidf_x_train=tfidf.fit_transform(x_train.values)
tfidf_x_test=tfidf.transform(x_test.values)


# In[12]:


# Model Bulding
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,confusion_matrix
MB=MultinomialNB()
MB.fit(tfidf_x_train,y_train)
y_pred=MB.predict(tfidf_x_test)
accuracy_score=accuracy_score(y_test,y_pred)
confusion_matrix=confusion_matrix(y_test,y_pred)
classification_report=classification_report(y_test,y_pred)
print(accuracy_score)
print(confusion_matrix)
print(classification_report)
print()


# In[13]:


## Model Building
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score

Models = {'Logistic Regression': (LogisticRegression(solver='lbfgs'), {'C': [0.1,0.008,0.0045], 'penalty': ['l2']}),
         'Decision Tree': (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [4,2]}),
         'SVM':(svm.SVC(),{'C': [0.0012,0.005,0.09], 'kernel': ['linear', 'rbf'], 'gamma': [0.01, 0.2,0.5,0.008]}),
         'Random Forest':(RandomForestClassifier(),{'n_estimators': [10,500,400,20,70], 'max_depth': [3,2]}),
         'k-Nearest Neighbors': (KNeighborsClassifier(), {'n_neighbors': [1,2,3,4], 'weights': ['distance']}),
         'Gradient Boosting':(GradientBoostingClassifier(),{'n_estimators': [500,200],'learning_rate': [0.7,0.04], 'max_depth': [1,2,4]}),
         'XG Boost':(XGBClassifier(),{'n_estimators': [500,300],'learning_rate': [0.78,0.45],'gamma': [0.09,0.8],'reg_alpha': [0.78,0.23],'reg_lambda': [0.56,0.43]}),
         'Multinomial Naive Bayes': (MultinomialNB(), {'alpha': [0.1, 0.5, 1.0, 2.0]})}

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_array

# Determine the minimum number of samples in each class
min_samples_per_class = min(y_train.value_counts())

# Determine the maximum number of splits that can be performed
max_splits = min(4, min_samples_per_class)

# Define StratifiedKFold with the maximum possible splits
skf = StratifiedKFold(n_splits=max_splits)

# Use skf in GridSearchCV
for model_name, (model, param_grid) in Models.items():
    if model_name != 'Naive Bayes':  # Exclude Naive Bayes model
        grid = GridSearchCV(model, param_grid, cv=skf)
        grid.fit(tfidf_x_train, y_train)
        print(f"Best Parameters for {model_name}: {grid.best_params_}")
        print(f"Best Accuracy for {model_name}: {grid.best_score_}\n")
    else:
        # Convert sparse data to dense for Naive Bayes
        dense_x_train = check_array(tfidf_x_train, accept_sparse='csr').toarray()
        grid = GridSearchCV(model, param_grid, cv=skf)
        grid.fit(dense_x_train, y_train)
        print(f"Best Parameters for {model_name}: {grid.best_params_}")
        print(f"Best Accuracy for {model_name}: {grid.best_score_}\n")


# In[23]:


from sklearn.metrics import confusion_matrix, classification_report

knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(tfidf_x_train, y_train)
y_pred = knn.predict(tfidf_x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[24]:


from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

# Reduce the dimensionality of the TF-IDF matrix using PCA with arpack solver
pca = PCA(n_components=2, svd_solver='arpack')
tfidf_x_train_pca = pca.fit_transform(tfidf_x_train)

# Define the step size in the mesh
h = 0.02

# Create a mesh grid to plot the decision boundaries
x_min, x_max = tfidf_x_train_pca[:, 0].min() - 1, tfidf_x_train_pca[:, 0].max() + 1
y_min, y_max = tfidf_x_train_pca[:, 1].min() - 1, tfidf_x_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Fit the classifier to the reduced-dimensional data
knn.fit(tfidf_x_train_pca, y_train)

# Make predictions on the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])

# Plot the decision boundary
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot the training points with labels
scatter = plt.scatter(tfidf_x_train_pca[:, 0], tfidf_x_train_pca[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)

# Annotate each point with its label
for i, label in enumerate(y_train):
    plt.text(tfidf_x_train_pca[i, 0], tfidf_x_train_pca[i, 1], str(label), fontsize=8, ha='center', va='center')

# Set plot limits and labels
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("KNN Decision Boundary (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

# Create a legend for the labels
plt.legend(handles=scatter.legend_elements()[0], labels=['0', '1', '2', '3'], title='Labels')

plt.show()


# In[32]:


# Importing pickle to save the model
import pickle

# File paths
file_clean = "C:\\Users\\DELL\\Downloads\\News_Classifier\\clean.pkl"
file_knn = "C:\\Users\\DELL\\Downloads\\News_Classifier\\knn_Model_S.pkl"
file_vec = "C:\\Users\\DELL\\Downloads\\News_Classifier\\vec.pkl"
file_pca = "C:\\Users\\DELL\\Downloads\\News_Classifier\\pca.pkl"

# Save preprocess step
with open(file_clean, 'wb') as f_clean:
    pickle.dump(clean_and_preprocess, f_clean)

# Load proprocess step
with open(file_clean, 'rb') as f_clean:
    clean_loaded = pickle.load(f_clean)

# Save KNN model
with open(file_knn, 'wb') as f_knn:
    pickle.dump(knn, f_knn)

# Load KNN model
with open(file_knn, 'rb') as f_knn:
    knn_loaded = pickle.load(f_knn)

# Save TF-IDF vectorizer
with open(file_vec, 'wb') as f_vec:
    pickle.dump(tfidf, f_vec)

# Load TF-IDF vectorizer
with open(file_vec, 'rb') as f_vec:
    tfidf_loaded = pickle.load(f_vec)

# Save PCA model
with open(file_pca, 'wb') as f_pca:
    pickle.dump(pca, f_pca)

# Load PCA model
with open(file_pca, 'rb') as f_pca:
    pca_loaded = pickle.load(f_pca)


# In[28]:


df_1.to_excel('News_Classifier.xlsx')


# In[34]:


df.to_excel('News_Classifier_org.xlsx')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




