import time
import sys
import pandas as pd
from project import my_model
sys.path.insert(0, '../..')
from my_evaluation import my_evaluation
from sklearn.model_selection import train_test_split

def test(data):
    
    data_split = train_test_split(data, test_size=.2,train_size=.8)
    y_test = data_split[1]["Class"]
    X_test = data_split[1].drop(["Class"], axis=1)
    y_train = data_split[0]["Class"]
    X_train = data_split[0].drop(["Class"], axis=1)

    #just taking first 20% of document as is
    # y = data["Class"]
    # X = data.drop(['Class'], axis=1)
    # split_point = int(0.8 * len(y))
    # X_train = X.iloc[:split_point]
    # X_test = X.iloc[split_point:]
    # y_train = y.iloc[:split_point]
    # y_test = y.iloc[split_point:]

    clf = my_model()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    eval = my_evaluation(predictions, y_test)
    from sklearn import metrics
    print(metrics.classification_report(y_test.values, predictions))

    f1 = eval.f1(average = "micro")
    return f1


if __name__ == "__main__":
    start = time.time()
    # Load data
    data = pd.read_csv("data/FR-Dataset.csv")
    # Replace missing values with empty strings
    data = data.fillna("")
    f1 = test(data)
    print("Micro Average: %f" % f1)
    runtime = (time.time() - start) / 60.0
    print(runtime)
