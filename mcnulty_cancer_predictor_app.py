import flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.metrics import *
from sklearn.cross_validation import *
# %matplotlib inline
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data1 = pd.read_csv('cleveland.csv', header = None)### 282 samples 75 columns
data_cleveland = pd.DataFrame(data1)


data2 = pd.read_csv('hungarian.csv', header = None) ###294 samples 75 columns
data_hungary = pd.DataFrame(data2)


data3= pd.read_csv('switzerland.csv', header = None)###123 samples 75 columns
data_switzerland=pd.DataFrame(data3)


data4= pd.read_csv('long-beach-va.csv', header = None)###200 samples 75 columns
data_longbeach=pd.DataFrame(data4)

#combine all dataframes  ###899 rows
df= data_cleveland.append(data_hungary).append(data_switzerland).append(data_longbeach)

df.columns=['id','ccf','age','sex','painloc','painexer','relrest', 'pncaden',
           'cp', 'trestbps', 'htn', 'chol','smoke', 'cigs', 
            'years', 'fbs', 'dm', 'famhist','restecg','ekgmo', 'ekgday', 'ekgyr', 
            'dig', 'prop', 'nitr', 'pro', 'diuretic','proto','thaldur','thaltime',
            'met', 'thalach','thalrest','tpeakbps','tpeakbpd','dummy','trestbpd',
            'exang','xhypo','oldpeak','slope','rldv5','rldv5e','ca','restckm',
            'exerckm','restef','restwm','exeref','exerwm','thal','thalsev',
            'thalpul','earlobe','cmo','cday','cyr','num','lmt','ladprox','laddist',
            'diag','cxmain','ramus','om1','om2','rcaprox','rcadist','lvx1','lvx2',
            'lvx3','lvx4','lvf','cathef','junk','name']
df=df.replace(to_replace=-9, value=np.nan)
df=df.reset_index(drop=True)

df_response=df['num']###response label column back up

df=df.drop('num', axis=1)####dropping the response variable from the dataframe

df_response=df_response.apply(lambda x: 1 if x >1 else x)

###remove useless columns from df and dummify and add new columns for those categories
remove_col_list= ['id','ccf','ekgmo','ekgday','ekgyr','thaltime','dummy','restckm',
                'exerckm','cmo','cday','cyr',
                'junk','name','lvx1','lvx2','lvx3','lvx4','lvf','cathef']

dummify_col_list=['pncaden','cp','restecg','proto','slope','ca','restwm','thal']


df=df.drop(remove_col_list,axis=1)

for i in range(len(dummify_col_list)):
    df_dummies = pd.get_dummies(df[dummify_col_list[i]],dummy_na=True)
    dum_prefix= dummify_col_list[i]
    dum_columns_list= list(df_dummies.columns)
    dum_new_columns_list=[str(dum_prefix)+'_'+str(z) for z in dum_columns_list]
    df_dummies.columns=dum_new_columns_list
    df=df.drop([dum_prefix],axis=1)##important to drop the col being dummified
    df = pd.concat([df, df_dummies], axis=1)

# %matplotlib inline
count_null=[]
for i in range(len(df.columns)):###only 75 columns here
    count_null.append(df[df.columns[i]].isnull().sum())
    
count_null=[(i)*100/len(df) for i in count_null] 

# if more than 95 % data is not recorded it is most probably not important
index_nan= [i for i, j in enumerate(count_null) if j >=95]

df=df.drop(df.columns[index_nan], axis=1)

###now remove dummy columns which have no corresponding real columns such as pncaden_nan etc
df=df.drop('pncaden_nan',axis=1)

simple_model_columns=['age','sex','cigs']
drop_column_simple= [item for item in df.columns if item not in simple_model_columns]
# print drop_column_simple
df=df.drop(drop_column_simple,axis=1)

df.describe()
##age, sex, smoke, cigs, years(as smoker), famhist
from sklearn.preprocessing import Imputer
y=df_response###only response
X=df###only features

imp = Imputer(missing_values='NaN', strategy='median', axis=0)##impute along columns
imp_X= imp.fit_transform(X)##impute whole data frame and use it to split

X=imp_X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler_list=['age', 'trestbps','chol','cigs','years','prop','thaldur','met','thalach','thalrest',
             'tpeakbps','tpeakbpd','trestbpd','oldpeak','rldv5','rldv5e','thalsev','lmt','ladprox']
# getting scaler column indices from latest df
scaler_list_num=[i for i,item in enumerate(list(df.columns)) if item in scaler_list]


X_train=pd.DataFrame.from_records(X_train)
X_test=pd.DataFrame.from_records(X_test)

X_train_contns=X_train.iloc[:,scaler_list_num]

X_test_contns=X_test.iloc[:,scaler_list_num]

std_scale = preprocessing.StandardScaler().fit(X_train_contns)# do we use all or only training 

X_train_contns_std = std_scale.transform(X_train_contns)
X_test_contns_std = std_scale.transform(X_test_contns)

df_X_train_contns_std = pd.DataFrame.from_records(X_train_contns_std)
df_X_test_contns_std = pd.DataFrame.from_records(X_test_contns_std)


categorical_columns_list= [item for item in list(X_train.columns) if item not in scaler_list_num]


df_X_train_categorical=X_train[categorical_columns_list]
df_X_test_categorical=X_test[categorical_columns_list]

X_train_std = pd.concat([df_X_train_contns_std, df_X_train_categorical], axis=1)
X_test_std = pd.concat([df_X_test_contns_std, df_X_test_categorical], axis=1)


# define a method for retrieving roc parameters

models = [LogisticRegression(), SVC(probability = True), GaussianNB(), DecisionTreeClassifier(max_depth = 4), RandomForestClassifier(), KNeighborsClassifier(n_neighbors = 9)]

#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on breast cancer survival,
# Build a LogisticRegression predictor on it

PREDICTOR = models[0].fit(X_train, y_train)

print PREDICTOR

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("mcnulty_awesome.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data["example"])
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
# does not run app.run(host='0.0.0.0', port=80) without sudo so removed arguments
##it goes to default port local host
app.run()