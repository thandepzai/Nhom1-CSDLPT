import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

import pandas as pd
from preprocess import extract_features, to_csv

#  Độ tương đồng cosine là một độ đo được sử dụng để đo độ giống nhau giữa hai vector, thường được sử dụng trong các bài toán liên quan đến xử lý ngôn ngữ tự nhiên (NLP), khai phá dữ liệu, và máy học
#  Hàm trả về giá trị độ tương đồng cosine giữa hai đặc trưng ft1 và ft2.
def cosine_similarity(ft1, ft2):
    return - (ft1 * ft2).sum() / (np.linalg.norm(ft1) * np.linalg.norm(ft2))

# # Khoảng cách Euclidean là một phép đo khoảng cách giữa hai điểm trong không gian nhiều chiều (trong trường hợp này, không gian nhiều chiều được tạo bởi các đặc trưng của các mẫu dữ liệu)
# # Hàm trả về giá trị khoảng cách Euclidean giữa hai đặc trưng ft1 và ft2.
# def euclidean_distance(ft1, ft2):
#     return np.linalg.norm(ft1 - ft2)

# # Khoảng cách Manhattan là một phép đo khoảng cách giữa hai điểm trong không gian nhiều chiều, được tính bằng cách lấy tổng giá trị tuyệt đối của sự khác biệt giữa các phần tử của hai vector
# # Hàm trả về giá trị khoảng cách Manhattan giữa hai vector p_vec và q_vec.
# def manhattan_distance(p_vec, q_vec):
#     return np.sum(np.fabs(p_vec - q_vec))


to_csv('precomputed')

train_folder = 'data/train'
test_folder = 'data/test'
test_images_path = '1.jpg'
df = pd.read_csv('data.csv')
# Kích thước ảnh
shape = (256, 256) 

# Đọc ảnh test từ đường dẫn 'data/test/chinhsida.jpg' bằng hàm cv2.imread(), lưu vào biến test_img.
test_img = cv2.imread(os.path.join(train_folder, test_images_path))
# Gọi hàm extract_features() để trích xuất đặc trưng từ ảnh test. Đầu vào của hàm gồm ảnh test (test_img) và kích thước mong muốn của ảnh sau khi resize (shape). 
test_color_mean, test_color_std, test_hog = extract_features(test_img, shape)
print(test_color_mean, test_hog)

# Code này sử dụng thư viện matplotlib để vẽ các hình ảnh trên một khung hình (figure). Đầu tiên, chúng ta tạo một đối tượng hình (figure) với kích thước 10 x 7 (inch)
fig = plt.figure(figsize=(10, 7))
rows = 6
columns = 2
#  tạo một subplot đầu tiên cho hình ảnh trên khung hình với vị trí là (1, 1). Các subplot khác sẽ được tạo tương tự với vị trí tăng dần theo chiều từ trái sang phải và từ trên xuống dưới, ví dụ (1, 2), (2, 1), (2, 2), vv.
# Kiểu như lúc in ra thì nó in ra ảnh 1 chạy file testaddsublot.py là hiểu
fig.add_subplot(rows, columns, 1)

dsts = []
images = []

#for idx, row in df.iterrows() sẽ lặp qua tất cả các dòng trong DataFrame df và cho phép truy cập đến chỉ số của dòng (biến idx) và giá trị của dòng (biến row) dưới dạng một Series.
for idx, row in df.iterrows():
    color_mean = np.load(row[2])
    color_std = np.load(row[3])
    hog_mean = np.load(row[4])
    # Tính toán khoảng cách giữa ảnh kiểm tra và mỗi ảnh trong df bằng cách tính cosine similarity giữa các tính năng mà chúng ta đã trích xuất từ ảnh (màu trung bình, độ lệch chuẩn của màu, và histogram của đặc trưng hướng cạnh).
    color_mean_dst = cosine_similarity(color_mean, test_color_mean)
    color_std_dst = cosine_similarity(color_std, test_color_std)
    hog_dst = cosine_similarity(hog_mean, test_hog)
    # Tổng hợp khoảng cách bằng cách cộng các giá trị cosine similarity vừa tính ở trên.
    # Thêm giá trị khoảng cách tính được vào danh sách dsts.
    dsts.append(color_mean_dst + hog_dst + color_std_dst)
    # Thêm tên của ảnh tương ứng được lưu trong cột 1 của df vào danh sách images.
    images.append(row[1])

# Chuyển list dsts và images sang kiểu numpy array bằng hàm np.array.
dsts = np.array(dsts)
images = np.array(images)
print(dsts, images)

#  Sắp xếp lại images theo thứ tự của dsts, lấy 10 ảnh đầu tiên và gán vào biến top_ten.
top_ten = images[dsts.argsort()][:10]

# Biểu đồ
# chuyển đổi định dạng màu của ảnh từ BGR sang RGB.
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
# thay đổi kích thước của ảnh test_img thành kích thước shape (256x256) bằng phương pháp resize.
test_img = cv2.resize(test_img, (shape[1], shape[0]))
# hiển thị ảnh test_img.
plt.imshow(test_img)
#  tắt trục x và y của đồ thị.
plt.axis('off')
#  đặt tiêu đề của đồ thị là tên của ảnh test_images_path.
plt.title(test_images_path)

#  Lặp qua danh sách top_ten với i là vị trí thứ tự của hình ảnh trong danh sách và image là tên của hình ảnh đó
for i, image in enumerate(top_ten):
    # Tạo subplot mới trong hình ảnh với số hàng là rows, số cột là columns và vị trí là i + 3. Vị trí được tính toán bằng cách thêm 3 để tránh ghi đè lên hình ảnh ban đầu và dòng tiêu đề của nó.
    fig.add_subplot(rows, columns, i + 3)
    # Đọc hình ảnh từ thư mục train_folder với tên là image.
    img = cv2.imread(os.path.join(train_folder, image))
    # Chuyển đổi không gian màu của hình ảnh từ BGR sang RGB để có thể hiển thị đúng trên matplotlib.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hay đổi kích thước của hình ảnh thành (shape[1], shape[0]), tức là (256, 256).
    img = cv2.resize(img, (shape[1], shape[0]))
    #  Hiển thị hình ảnh trên subplot hiện tại.
    plt.imshow(img)
    # Ẩn các trục của subplot để tạo ra hình ảnh đẹp hơn.
    plt.axis('off')
    #  Đặt tiêu đề cho hình ảnh hiện tại là tên của nó.
    plt.title(image, fontsize=12)

print(top_ten)

plt.show()
