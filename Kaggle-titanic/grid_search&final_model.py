import warnings
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC


def load_data():
    data_test = pd.read_csv('new_test.csv', index_col= 0)
    data_train = pd.read_csv('new_train.csv', index_col= 0)
    return data_train , data_test

# def fare_scale(df):
#     scaler = StandardScaler()
#     fare_scale = scaler.fit(df['Fare'])
#     df['fare_scale'] = scaler.fit_transform(df['Fare'], fare_scale)
#     df.drop(['Fare'], axis=1, inplace= True)
#     return df


if __name__ == '__main__':
    data_train, data_test = load_data()
    # fare_scale(data_train)
    # fare_scale(data_test)
    pd.set_option('display.width', 5000)
    np.set_printoptions(suppress=True)
    x_train = np.array(data_train.values[:,1:])
    y_train = np.array(data_train.values[:,0])
    x_test = np.array(data_test.values[:,1:])

    # model = RandomForestClassifier(max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
    # param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10],
    #                               "min_samples_split" : [2, 4, 10, 12, 16], "n_estimators": [50, 100, 150, 200, 400, 700, 1000]}
    # clf = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=20, n_jobs=-1)
    # 
    # clf.fit(x_train, y_train)
    # 
    # print(clf.best_score_)
    # print(clf.best_params_)
    #
    rf = RandomForestClassifier(criterion='gini',
                                 n_estimators=700,
                                 min_samples_split=16,
                                 min_samples_leaf=1,
                                 max_features='auto',
                                 oob_score=True,
                                 random_state=1,
                                 n_jobs=-1)
    rf.fit(x_train, y_train)
    Y_pred = rf.predict(x_test)
    submission = pd.DataFrame({
        "PassengerId": data_test["PassengerId"],

        "Survived": Y_pred.astype(int)
    })
    submission.to_csv('submission.csv', index=False)
