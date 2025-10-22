import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

data_dict = pickle.load(open('./data.pickle', 'rb'))

DATA_DIR = './data'


labels = sorted(os.listdir(DATA_DIR))  


label_mapping = {label: idx for idx, label in enumerate(labels)}
reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

filtered_data = []
filtered_labels = []
for data, label in zip(data_dict['data'], data_dict['labels']):
    if label in label_mapping:  
        filtered_data.append(data)
        filtered_labels.append(label_mapping[label])

# Convert data and labels into numpy arrays
data = np.asarray(filtered_data)
labels = np.asarray(filtered_labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(random_state=84)
model.fit(x_train, y_train)

# Make predictions on the test set
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a pickle file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_mapping': reverse_label_mapping}, f)
