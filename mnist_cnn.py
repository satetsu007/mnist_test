from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from load_data import load_data

# 各種パラメータ設定
num_pic = 10
batch_size = 32
epochs = 10

print("Load Data.")

X_train, X_test, y_train, y_test = load_data(num_pic, mode="cnn")

n_in = X_train.shape[-1]
n_out = y_train.shape[-1]
input_shape = (num_pic, 28, 28)

print("Define Model.")

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_out, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

print("Train Phase.")

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,)

print("Test Phase.\n")
test = model.evaluate(X_test, y_test, batch_size=32)

print("\n\nTest Loss: %f\nTest Acc: %f" % (test[0], test[1]))
