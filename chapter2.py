import tensorflow as tf
from tensorflow.keras import Sequential, datasets
from tensorflow.keras.layers import Dense, Flatten

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            print("\n정확도 95%에 도달하여 훈련을 멈춥니다!")
            self.model.stop_training = True

(train_images, train_labels), (test_images, test_labels)  = datasets.fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50, callbacks=[myCallback()])
model.evaluate(test_images, test_labels)


classification = model.predict(test_images)
print(classification[0])
print(test_labels[0])



