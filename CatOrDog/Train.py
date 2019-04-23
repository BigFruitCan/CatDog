from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
import Load_Data

model = Sequential()

# 第一个卷积层
model.add(Conv2D(32, kernel_size=(3, 3),input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# 第二个卷积层
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# 第三个卷积层
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
# model.add(MaxPool2D())
# model.add(Dropout(0.5))

model.add(Flatten())
# 全连接层
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
epochs = 40
model.fit(Load_Data.train_data, Load_Data.train_label,
         epochs=epochs, batch_size=100,
         validation_data=(Load_Data.test_data, Load_Data.test_label))

# # 绘制曲线
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, epochs), model.history["loss"], label="train_loss")
# plt.plot(np.arange(0, epochs), model.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, epochs), model.history["acc"], label="train_acc")
# plt.plot(np.arange(0, epochs), model.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on sar classifier")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper right")

model.save('output')
