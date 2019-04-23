from keras.models import load_model
import random
import cv2
import numpy as np

size = (128, 128)
model = load_model('output')

def getpicture():
    i = random.randint(0, 2000)
    type = random.randint(0, 1)
    a = []
    if type == 0:
        path = "Data/cat/cat." + str(i) + ".jpg"
        x = cv2.imread(path)
        y = cv2.resize(x, dsize=size)
        a.append(y)
        a = np.array(a)
        return a / 255.0
    else:
        path = "Data/dog/dog." + str(i) + ".jpg"
        x = cv2.imread(path)
        y = cv2.resize(x, dsize=size)
        a.append(y)
        a = np.array(a)
        return a / 255.0


x = getpicture()
res = model.predict(x)
cv2.imshow("a", x[0])
print(res)
if res[0][0] < 0.5:
    print("猫")
else:
    print("狗")
cv2.waitKey(0)
cv2.destroyAllWindows()
