import numpy as np
import cv2

size = (128, 128)


def load():
    images = []
    labels = []
    for i in range(0, 1000):
        img_path = "Data/cat/cat." + str(i) + ".jpg"
        x = cv2.imread(img_path)
        y = cv2.resize(x, dsize=size)
        images.append(y)
        labels.append(0)

    for i in range(1000, 2000):
        img_path = "Data/dog/dog." + str(i-1000) + ".jpg"
        x = cv2.imread(img_path)
        y = cv2.resize(x, dsize=size)
        images.append(y)
        labels.append(1)

    data = np.array(images)
    label = np.array(labels)
    return data / 255.0, label


# 打乱数据
data, label = load()
permutation = np.random.permutation(label.shape[0])
all_data = data[permutation]
all_label = label[permutation]

# 划分测试集和数据集

train_data = all_data[0:1600]
train_label = all_label[0:1600]

test_data = all_data[1600:]
test_label = all_label[1600:]

# print(train_data.shape)
# print(train_label.shape)
# print(test_data.shape)
# print(test_label.shape)

# img_path = "Data/cat/cat.0.jpg"
# x = cv2.imread(img_path)
# y = cv2.resize(x, dsize=(128, 128))
# y = train_data[5]
# cv2.imshow("a", y)
# print(y)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
