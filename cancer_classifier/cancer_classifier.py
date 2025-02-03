from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd

# Load data
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

# split data
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

# ensure training data and labels are same length
print(len(training_data)) #455
print(len(training_labels)) #455

# find best k for KNN classifier 
k_list = []
accuracies = []
for i in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(training_data, training_labels)
    k_list.append(i)
    accuracies.append(classifier.score(validation_data, validation_labels))

print(accuracies.index(max(accuracies))) # 22

print(accuracies[22]) # 0.9649122807017544
print(k_list[22]) # 23

# model with best k
classifier = KNeighborsClassifier(n_neighbors=23)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels)) # 0.9649122807017544

# confusion matrix
confusion_matrix = pd.crosstab(validation_labels, classifier.predict(validation_data), rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)

# plot confusion matrix
plt.matshow(confusion_matrix)
plt.title('Confusion matrix')
plt.colorbar()
plt.text(0, 0, confusion_matrix[0][0], ha='center', va='center', color='black')
plt.text(0, 1, confusion_matrix[0][1], ha='center', va='center', color='white')
plt.text(1, 0, confusion_matrix[1][0], ha='center', va='center', color='white')
plt.text(1, 1, confusion_matrix[1][1], ha='center', va='center', color='black')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# save plot
plt.savefig('cancer_classifier/confusion_matrix.png')

# calculate accuracy, precision, recall
true_positives = confusion_matrix[1][1]
true_negatives = confusion_matrix[0][0]
false_positives = confusion_matrix[0][1]
false_negatives = confusion_matrix[1][0]

accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1 = 2 * (precision * recall) / (precision + recall)

print(accuracy) # 0.9649122807017544
print(precision) # 0.9538461538461539
print(recall) # 0.9841269841269841
print(f1) # 0.96875




