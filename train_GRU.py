# --- Import thư viện ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from sklearn.metrics import accuracy_score

# --- Đọc dữ liệu ảnh ---
data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Duyệt qua 43 thư mục con (ứng với 43 loại biển báo)
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Chuyển đổi sang array
data = np.array(data)
labels = np.array(labels)

# --- Chia tập Train và Test ---
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# One-hot encoding cho nhãn
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# --- Chuyển đổi dữ liệu đầu vào cho GRU ---
# Ảnh (30,30,3) -> reshape thành (30, 90)
X_train = X_train.reshape(X_train.shape[0], 30, 90)
X_test = X_test.reshape(X_test.shape[0], 30, 90)

# --- Xây dựng mô hình GRU ---
model = Sequential()

# Lớp GRU đầu tiên
model.add(GRU(128, input_shape=(30, 90), activation='tanh', return_sequences=True))
model.add(Dropout(0.2))

# Lớp GRU thứ hai
model.add(GRU(64, activation='tanh'))
model.add(Dropout(0.2))

# Lớp Dense ẩn
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

# Lớp output
model.add(Dense(43, activation='softmax'))

# Compile mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Huấn luyện mô hình ---
epochs = 30
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# --- Lưu mô hình ---
model.save("my_model_GRU.h5")

# --- Vẽ đồ thị Accuracy ---
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# --- Vẽ đồ thị Loss ---
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# --- Đọc file Test.csv để kiểm tra mô hình ---
y_test = pd.read_csv('Test.csv')
labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)
X_test = X_test.reshape(X_test.shape[0], 30, 90)  # Reshape cho đúng input GRU

# Dự đoán
predictions = model.predict(X_test)
pred = np.argmax(predictions, axis=1)

# Tính Accuracy
print("Accuracy on Test.csv:", accuracy_score(labels, pred))

# --- Lưu mô hình thêm một file nữa ---
model.save('traffic_classifier_GRU.h5')
