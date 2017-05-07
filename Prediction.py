import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

data_frame = pd.read_excel('/Users/ndahiwad/Desktop/Employee_data.xlsx')

print(data_frame.head(15))


def data_preprocessing(data_frame):

    # Initialize a dict for text to integer mapping
    text_to_digit = {}

    columns = data_frame.columns.values
    for column in columns:

        def raw_data_to_int(value):
            return text_to_digit[value]

        # if column values are neither int nor float, convert all the unique
        # text data into integers starting from 0
        if data_frame[column].dtype != np.int64\
                and data_frame[column].dtype != np.float64:
            column_values = data_frame[column].values.tolist()
            unique_elements = set(column_values)
            x = 0
            for each_element in unique_elements:
                text_to_digit[each_element] = x
                x += 1

        data_frame[column] = [raw_data_to_int(x) for x in
                              data_frame[column].values]

    return data_frame

data_frame.drop(['Employee_ID'], 1, inplace=True)
# data_frame.drop(['Performance'], 1, inplace=True)
# data_frame.drop(['Org_change'], 1, inplace=True)
'''We can drop either 'Performance' or 'Org_change' to see the effect of
   either (individually) on the Employees being 'Laid-Off'. But our current
   implementation is predicting on the basis of both.
'''


data_preprocessing(data_frame)

# print(data_frame.head(15))

X = np.array(data_frame.drop(['Laid-Off'], 1).astype(float))
y = np.array(data_frame['Laid-Off'])
X = preprocessing.scale(X)
# print(X)


clf = KMeans(n_clusters=2)
# Two Clusters chosen to classify whether an Employee was laid-off or not
clf.fit(X)


def predict(x):
    correct = 0
    for i in range(len(x)):
        predict_me = np.array(x[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        # print(predict_me)
        prediction = clf.predict(predict_me)
        # print(prediction)
        if prediction[0] == y[i]:
            correct += 1
    return correct

correct_prediction = predict(X)
print(correct_prediction/len(X))
