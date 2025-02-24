import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedKFold, cross_val_score
from sklearn.impute import KNNImputer, SimpleImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier

train_data = pd.read_csv('train_set.csv')
test_data = pd.read_csv('test_set.csv')
df_submission = pd.read_csv("sample_solution.csv")

X = train_data.drop(columns=['target'])
y = train_data['target']

print(X.shape)
print(test_data.shape)

categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(len(categorical_columns))
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
df_test_encoded = pd.get_dummies(test_data, columns=categorical_columns, drop_first=True)

print(X.shape)
print(df_test_encoded.shape)

numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X[numerical_columns])
df_test_encoded = imputer.transform(df_test_encoded[numerical_columns])

ss = StandardScaler()
X = ss.fit_transform(X)
df_test_encoded = ss.transform(df_test_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=400, random_state=42)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)

test_probabilities = model.predict_proba(df_test_encoded)[:, 1]

df_submission['Y'] = test_probabilities
df_submission.to_csv('submission.csv', index=False)

