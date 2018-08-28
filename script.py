# Load Requirements
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Explore the data

## 1
breast_cancer_data = load_breast_cancer()

## 2
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

"""
Each number within printed for the first datapoint represents a "feature" or
predictor variable. For example, 1.799e+01 (17.99) represents the "mean radius".
"""


## 3
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

"""
The first datapoint is malignant.
malignant: 0
benign: 1
"""

# Splitting the data into Training and Validation Sets

## 4
### already run at top of script
### from sklearn.model_selection import train_test_split

## 5
### specifying train_size=0.8 is the same as specifying test_size=0.2
train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size=0.2,
    random_state=100
)

## 6
### specifying train_size=0.8 is the same as specifying test_size=0.2
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size=0.2,
    random_state=100
)

## 7
print("training data length:", len(training_data))
print("training labels length:", len(training_labels))

# Running the classifier

## 8
### already run at top of script
### from sklearn.neighbors import KNeighborsClassifier

## 9
classifier = KNeighborsClassifier(n_neighbors=3)

## 10
classifier.fit(training_data, training_labels)

## 11
print(classifier.score(validation_data, validation_labels))

## 12
best_k = None
best_score = 0
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    if classifier.score(validation_data, validation_labels) > best_score:
        best_score = classifier.score(validation_data, validation_labels)
        best_k = k
    print('k = {}:'.format(k), classifier.score(validation_data, validation_labels))

print('best k:', best_k)
print('best score:', best_score)

# Graphing the results

## 13
### already run at top of script
### import matplotlib.pyplot as plt

## 14
k_list = list(range(1, 101))

## 15
accuracies = []
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

## 16
plt.plot(k_list, accuracies)
plt.show()

## 17
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.plot(k_list, accuracies)
plt.show()

## 18
"""
Thanks for the challenge.
"""
