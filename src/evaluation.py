import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

x_train = np.load(r"data/processed/x_train.npy")
y_train = np.load(r"data/processed/y_train.npy")
x_test = np.load(r"data/processed/x_test.npy")
y_test = np.load(r"data/processed/y_test.npy")

model = tf.keras.models.load_model(r"model/model.keras")

train_score = model.evaluate(x_train, y_train)
test_score = model.evaluate(x_test, y_test)

print(f"Train Accuracy: {train_score[1]}")
print(f"Test Accuracy: {test_score[1]}")

result = model.predict(x_test)
result = np.argmax(result, axis=1)

y_test_classes = np.argmax(y_test, axis=1)
class_report = classification_report(y_test_classes, result, target_names=["No Tumor", "Tumor"])
print(class_report)