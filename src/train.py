import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

# Set tracking URI and experiment so it's easier to find in the MLflow UI
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Tumor-Classification")

# Enable MLflow autologging without epoch checkpoints
mlflow.tensorflow.autolog(checkpoint=False)

x_train = np.load(r"data/processed/x_train.npy")
x_test = np.load(r"data/processed/x_test.npy")
y_train = np.load(r"data/processed/y_train.npy")
y_test = np.load(r"data/processed/y_test.npy")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(2,2), input_shape=(128, 128, 3), padding = "same"),
    tf.keras.layers.Conv2D(32, kernel_size=(2,2), activation='relu', padding = "same"),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, kernel_size=(2,2), activation='relu', padding = "same"),
    tf.keras.layers.Conv2D(64, kernel_size=(2,2), activation='relu', padding = "same"),

    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

model.summary()

with mlflow.start_run():
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[callback])

model.save(r"model/model.keras")