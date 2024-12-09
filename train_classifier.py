import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure consistent label mapping
label_mapping = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8,
    "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16,
    "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25
}

# Map string labels to integers
reverse_label_mapping = {v: k for k, v in label_mapping.items()}  # For reverse mapping later

filtered_data = []
filtered_labels = []
for data, label in zip(data_dict['data'], data_dict['labels']):
    if label in label_mapping:  # Ovo osigurava da su svi podaci pravilno mapirani
        filtered_data.append(data)
        filtered_labels.append(label_mapping[label])

# Fix inconsistency in data length by padding with zeros
max_length = max(len(item) for item in filtered_data)
data_fixed = []
for item in filtered_data:
    if len(item) < max_length:
        item = item + [0] * (max_length - len(item))  # Popravite dužinu podataka
    data_fixed.append(item)


# Convert data and labels into numpy arrays
data = np.asarray(data_fixed)
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
