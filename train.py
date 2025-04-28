# Import các thư viện cần thiết
import numpy as np                  # Thư viện xử lý mảng số học
import pandas as pd                 # Thư viện xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt     # Thư viện vẽ đồ thị
import cv2                          # Thư viện xử lý ảnh OpenCV
import tensorflow as tf             # Thư viện TensorFlow cho học sâu
from PIL import Image               # Thư viện PIL để xử lý ảnh
import os                           # Thư viện thao tác với hệ thống tệp (file)
from sklearn.model_selection import train_test_split  # Hàm chia tập dữ liệu
from keras.utils import to_categorical # Hàm chuyển nhãn thành dạng one-hot
from keras.models import Sequential, load_model  # Mô hình Sequential, hàm load model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # Các lớp trong Keras

# Khởi tạo danh sách rỗng để lưu dữ liệu và nhãn
data = []
labels = []

# Tổng số lớp (số loại biển báo giao thông)
classes = 43

# Lấy đường dẫn hiện tại (thư mục chứa file code)
cur_path = os.getcwd()

# Lặp qua tất cả 43 lớp (tương ứng với 43 folder từ 0 đến 42)
for i in range(classes):
    path = os.path.join(cur_path,'train',str(i))  # Tạo đường dẫn đến folder của lớp i
    images = os.listdir(path)                     # Lấy danh sách tất cả các ảnh trong thư mục đó

    for a in images:                              # Lặp qua từng ảnh
        try:
            image = Image.open(path + '\\'+ a)     # Mở ảnh
            image = image.resize((30,30))          # Resize ảnh về kích thước 30x30
            image = np.array(image)                # Chuyển ảnh thành mảng numpy
            # sim = Image.fromarray(image)         # (comment) - dòng này không dùng
            data.append(image)                     # Thêm ảnh vào danh sách data
            labels.append(i)                       # Thêm nhãn (i) vào danh sách labels
        except:
            print("Error loading image")           # Nếu lỗi (ảnh hỏng) thì in thông báo

# Chuyển danh sách thành mảng numpy để huấn luyện dễ hơn
data = np.array(data)
labels = np.array(labels)

# In ra kích thước của tập dữ liệu và nhãn
print(data.shape, labels.shape)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra (test 20%)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# In ra kích thước tập train và test
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Chuyển nhãn từ dạng số sang dạng one-hot encoding (43 lớp)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Xây dựng mô hình CNN theo dạng Sequential (tuyến tính lớp này đến lớp kia)
model = Sequential()

# Thêm lớp Convolutional: 32 filter, kích thước kernel 5x5, activation relu, input_shape dựa trên kích thước ảnh
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))

# Thêm lớp Convolutional tiếp theo: 32 filter, kernel 5x5
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))

# Thêm lớp Max Pooling: giảm kích thước không gian đặc trưng
model.add(MaxPool2D(pool_size=(2, 2)))

# Thêm Dropout: giúp tránh overfitting bằng cách tắt ngẫu nhiên 25% số neuron
model.add(Dropout(rate=0.25))

# Thêm lớp Convolutional: 64 filter, kernel 3x3
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# Thêm lớp Convolutional tiếp theo: 64 filter, kernel 3x3
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))

# Thêm lớp Max Pooling
model.add(MaxPool2D(pool_size=(2, 2)))

# Thêm Dropout 25% sau pooling
model.add(Dropout(rate=0.25))

# Làm phẳng dữ liệu từ 2D thành 1D để đưa vào Dense layer
model.add(Flatten())

# Thêm lớp Fully Connected (Dense) với 256 neuron và hàm relu
model.add(Dense(256, activation='relu'))

# Thêm Dropout 50% để tránh overfitting
model.add(Dropout(rate=0.5))

# Thêm lớp output: 43 neuron tương ứng với 43 lớp, activation softmax để phân loại
model.add(Dense(43, activation='softmax'))

# Compile mô hình: chọn hàm loss là categorical_crossentropy, tối ưu hóa bằng Adam, đánh giá bằng accuracy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Đặt số epoch để train
epochs = 15

# Huấn luyện mô hình
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

# Lưu mô hình đã huấn luyện vào file 'my_model.h5'
model.save("my_model.h5")

# --- Gộp đồ thị Accuracy và Loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))   # 1 dòng 2 cột

# Đồ thị Accuracy
ax1.plot(history.history['accuracy'], label='training accuracy')
ax1.plot(history.history['val_accuracy'], label='validation accuracy')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.set_ylim([0, 1])    # <--- Cố định trục Oy từ 0 đến 1
ax1.legend()

# Đồ thị Loss
ax2.plot(history.history['loss'], label='training loss')
ax2.plot(history.history['val_loss'], label='validation loss')
ax2.set_title('Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.set_ylim([0, 1])    # <--- Cố định trục Oy từ 0 đến 1
ax2.legend()

plt.tight_layout()
plt.show()

# Import thư viện tính toán độ chính xác
from sklearn.metrics import accuracy_score

# Đọc file Test.csv chứa dữ liệu test thực tế
y_test = pd.read_csv('Test.csv')

# Lấy nhãn thật từ cột ClassId
labels = y_test["ClassId"].values

# Lấy đường dẫn file ảnh từ cột Path
imgs = y_test["Path"].values

# Tạo danh sách lưu dữ liệu ảnh test
data=[]

# Duyệt qua từng đường dẫn ảnh
for img in imgs:
    image = Image.open(img)            # Mở ảnh
    image = image.resize((30,30))       # Resize ảnh về 30x30
    data.append(np.array(image))        # Chuyển ảnh thành array rồi thêm vào list

# Chuyển danh sách ảnh test thành numpy array
X_test=np.array(data)

# Dự đoán kết quả bằng mô hình
predictions = model.predict(X_test)

# Lấy index của giá trị xác suất cao nhất cho mỗi dự đoán (tức là label dự đoán)
pred = np.argmax(predictions, axis=1)

# Import lại thư viện tính accuracy (thừa, nhưng không ảnh hưởng)
from sklearn.metrics import accuracy_score

# Tính và in độ chính xác so sánh nhãn thật và nhãn dự đoán
print(accuracy_score(labels, pred))

# Lưu lại mô hình đã huấn luyện thành file 'traffic_classifier.h5'
model.save('traffic_classifier.h5')
