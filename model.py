from tensorflow.keras.layers import Input, Flatten, Dense

model=tf.keras.models.Sequential(
    [Input((28,28)),
     Flatten(),
     Dense(100, activation='relu'),
     Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

loss, acc=model.evaluate(x_test, y_test)
predictions=model.predict(x_test)
print("\nLoss=%.4f, Accuracy=%.4f" % (loss*100, acc*100))
