
### 3.2 Data Preparation


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import pickle



C=1.0
df = pd.read_csv('Telco-Customer-Churn.csv')
df.head()


df.columns = df.columns.str.lower().str.replace(' ', '_')


df.totalcharges = pd.to_numeric(df.totalcharges,errors='coerce')


str_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in str_columns:
    df[col] = df[col].str.lower().str.replace(' ','_')


df.isnull().sum()
df.totalcharges = df.totalcharges.fillna(0)
df.isnull().sum()





df.churn = (df.churn == 'yes').astype(int)


# ### 3.3 Setting up the validation framework



df_fulltrain, df_test = train_test_split(df, test_size = 0.2, random_state =1)
df_train, df_val = train_test_split(df_fulltrain, test_size = 0.25, random_state =1)
df_fulltrain.shape, df_test.shape, df_val.shape
df.nunique()
numerical = ['tenure', 'monthlycharges', 'totalcharges']
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']
fields = categorical + numerical
df_fulltrain.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.churn
y_val = df_val.churn
y_test = df_test.churn
del df_train['churn']
del df_val['churn']
del df_test['churn']


# ### 3.4 Training your model




dv = DictVectorizer(sparse = False)
dict_train = df_train[fields].to_dict(orient='records')
dv.fit(dict_train)
X_train = dv.transform(dict_train)

model = LogisticRegression(max_iter=10000, C=1)
model.fit(X_train, y_train)


w = model.coef_[0]


w0 = model.intercept_[0]


dict_val = df_val[fields].to_dict(orient='records')
X_val = dv.transform(dict_val)
val_predict = (model.predict_proba(X_val)[:,1].round(2)) > 0.5
(val_predict == y_val).mean()


# ### 3.5 Retrain and Use model


dict_fulltrain = df_fulltrain[fields].to_dict(orient='records')
X_fulltrain = dv.transform(dict_fulltrain)
y_fulltrain = df_fulltrain.churn.values
model = LogisticRegression(max_iter=10000, C=C)
model.fit(np.concatenate([X_train,X_val]), np.concatenate([y_train,y_val]))
w = model.coef_[0]
w0 = model.intercept_[0]


dict_test = df_test[fields].to_dict(orient='records')
X_test = dv.transform(dict_test)
predict = (model.predict_proba(X_test)[:,1].round(2)) >= 0.5


# ### 4.1 Confusion Table


tp = ((val_predict == y_val) & (val_predict == 1)).sum()


fp = ((val_predict != y_val) & (val_predict == 1)).sum()


tn = ((val_predict == y_val) & (val_predict == 0)).sum()


fn = ((val_predict != y_val) & (val_predict == 0)).sum()

cm = np.array([
    [tn, fp],
    [fn,tp]])
cm


(cm/cm.sum()).round(2)


# ### 4.2 Precison and Recall


precision = tp/(tp+fp)


recall = tp/(tp+fn)


# ### 4.3 ROC Curves




def scores_roc(y, y_predict):
    treasholds = np.linspace(0,1, 101)
    scores = []
    for t in treasholds:
        predict = y_predict >= t
        tp = ((predict == y) & (predict == 1)).sum()
        fp = ((predict != y) & (predict == 1)).sum()
        tn = ((predict == y) & (predict == 0)).sum()
        fn = ((predict != y) & (predict == 0)).sum()
        tpr = tp / (tp+fn)
        fpr = fp / (fp + tn)
        scores.append((t, tn, fp, fn, tp, tpr,fpr))
    return scores
   
    


val_predict = (model.predict_proba(X_val)[:,1].round(2))
scores = scores_roc(y_val, val_predict)
df_scores = pd.DataFrame(scores, columns = ['t','tn','fp','fn','tp','tpr','fpr'])


plt.plot(df_scores.fpr, df_scores.tpr)
plt.title('ROC Curve for our Model')


np.random.seed(1)
y_rand = np.random.uniform(0,1, len(y_val))
scores_rand = scores_roc(y_val, y_rand)
df_scores_rand = pd.DataFrame(scores_rand,columns = ['t','tn','fp','fn','tp','tpr','fpr'])


plt.plot(df_scores_rand.fpr, df_scores_rand.tpr)
plt.title('ROC Curve for a random model')


npos = (y_val == 1).sum()
nneg = (y_val == 0).sum()


y_ideal = np.repeat([0,1], [nneg,npos])


y_ideal_pred = np.linspace(0,1, len(y_val))
scores_ideal = scores_roc(y_ideal, y_ideal_pred)
df_scores_ideal = pd.DataFrame(scores_ideal, columns = ['t','tn','fp','fn','tp','tpr','fpr'])



plt.plot(df_scores_ideal.fpr, df_scores_ideal.tpr)
plt.title('ROC Curve for a the ideal model')


# ### 4.4 Putting Everythin together


plt.figure(figsize =(6,6))
plt.plot(df_scores.fpr, df_scores.tpr, label='model')
plt.plot(df_scores_rand.fpr, df_scores_rand.tpr, label='random')
plt.plot(df_scores_ideal.fpr, df_scores_ideal.tpr, label='ideal')
plt.title('ROC Curve showing comparison between Our Model, A Random Model, The Ideal Model')
plt.legend()



# ### 4.5 calculation ROC curves using scikit learn



def scores_roc(y, y_predict):
    fpr, tpr, treasholds = roc_curve(y, y_predict)
    scores = []
    for t in treasholds:
        predict = y_predict >= t
        tp = ((predict == y) & (predict == 1)).sum()
        fp = ((predict != y) & (predict == 1)).sum()
        tn = ((predict == y) & (predict == 0)).sum()
        fn = ((predict != y) & (predict == 0)).sum()
        scores.append((t, tn, fp, fn, tp))
    scores = np.column_stack([scores, tpr, fpr])
    return scores
val_predict = (model.predict_proba(X_val)[:,1].round(2))
scores = scores_roc(y_val, val_predict)
df_scores = pd.DataFrame(scores, columns = ['t','tn','fp','fn','tp','tpr','fpr'])




y_rand = np.random.uniform(0,1, len(y_val))
scores_rand = scores_roc(y_val, y_rand)
df_scores_rand = pd.DataFrame(scores_rand,columns = ['t','tn','fp','fn','tp','tpr','fpr'])


y_ideal = np.repeat([0,1], [nneg,npos])
y_ideal_pred = np.linspace(0,1, len(y_val))
scores_ideal = scores_roc(y_ideal, y_ideal_pred)
df_scores_ideal = pd.DataFrame(scores_ideal, columns = ['t','tn','fp','fn','tp','tpr','fpr'])



plt.figure(figsize =(6,6))
plt.plot(df_scores.fpr, df_scores.tpr, label='model')
plt.plot(df_scores_rand.fpr, df_scores_rand.tpr, label='random')
plt.plot(df_scores_ideal.fpr, df_scores_ideal.tpr, label='ideal')
plt.title('ROC Curve showing comparison between Our Model, A Random Model, The Ideal Model')
plt.legend()


# ### 4.6 Area under the ROC curve



area = auc(df_scores.fpr, df_scores.tpr)
area_rand = auc(df_scores_rand.fpr, df_scores_rand.tpr)
area_ideal = auc(df_scores_ideal.fpr, df_scores_ideal.tpr)


# ### 4.7 Area under the ROC curve method 2



area = roc_auc_score(y_val, val_predict)
area_rand = roc_auc_score(y_val, y_rand)
area_ideal = roc_auc_score(y_ideal, y_ideal_pred)


# ## Week 5 Saving and Loading our Model





filename = f"model_C={C}.bin"


with open(filename, 'wb') as file:
    pickle.dump((dv, model), file)



# ### Loading our Model


filename = f"model_C={C}.bin"







