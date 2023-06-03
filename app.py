from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

import pandas as pd
from preprocess import extract_features, to_csv


def cosine_similarity(ft1, ft2):
    return - (ft1 * ft2).sum() / (np.linalg.norm(ft1) * np.linalg.norm(ft2))


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


to_csv('precomputed')


@app.route('/upload', methods=['POST'])
def upload():
  train_folder = 'data/train'
  df = pd.read_csv('data.csv')
  # Kích thước ảnh
  shape = (256, 256)

  # Đọc ảnh từ đối tượng file trực tiếp
  uploaded_file = request.files['image']
  image_data = uploaded_file.read()

  # Chuyển đổi dữ liệu ảnh thành mảng numpy
  nparr = np.frombuffer(image_data, np.uint8)

  # Đọc ảnh từ mảng numpy bằng cv2.imdecode
  test_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  # Gọi hàm extract_features() để trích xuất đặc trưng từ ảnh test. Đầu vào của hàm gồm ảnh test (test_img) và kích thước mong muốn của ảnh sau khi resize (shape).
  test_color_mean, test_color_std, test_hog = extract_features(
      test_img, shape)
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

  # for idx, row in df.iterrows() sẽ lặp qua tất cả các dòng trong DataFrame df và cho phép truy cập đến chỉ số của dòng (biến idx) và giá trị của dòng (biến row) dưới dạng một Series.
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

  print(top_ten)
  ten_anh = ', '.join(str(element) for element in top_ten)
  return ten_anh
  # plt.show()

if __name__ == "__main__":
    app.run()
