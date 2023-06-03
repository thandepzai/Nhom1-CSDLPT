# importing required libraries
import cv2
import numpy as np
import os
import pandas as pd

# Hàm _hog tính toán đặc trưng HOG (Histogram of Oriented Gradients) từ một hình ảnh và kích thước mong muốn của hình ảnh đầu ra. 
# Đầu vào của hàm là một hình ảnh và kích thước mong muốn của hình ảnh đầu ra, hàm trả về đặc trưng HOG của hình ảnh. HOG là một 
# phương pháp rút trích đặc trưng từ hình ảnh bằng cách tính toán các histogram hướng của các gradient của hình ảnh. Các histogram 
# được tính toán trên các khối của hình ảnh và được chuẩn hóa để đảm bảo tính không đổi với phép xoay, phép co giãn và chiều sâu của hình ảnh. 
def _hog(image, shape):
    #  Đây là kích thước của mỗi khối tính histogram, trong đó các khối histogram sẽ được xếp chồng lên nhau để tạo ra một feature vector đầy đủ.
    block_size = 16
    #  Kích thước của mỗi ô tính gradient và tính toán histogram trong khối. Mỗi ô có kích thước cell_size x cell_size.
    cell_size = 8
    # Kiểm tra xem kích thước ảnh có chia hết cho cell_size hay không. Nếu không, sẽ có thông báo lỗi.
    assert (image.shape[0] % cell_size == 0 and image.shape[1] % cell_size == 0), "Size not supported"
    # Số lượng bin của histogram. Trong trường hợp này, mỗi ô trong khối sẽ có histogram 9 bin.
    nbins = 9
    # Ma trận lọc sobel theo trục x.
    dx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    #  Ma trận lọc sobel theo trục y là ma trận chuyển vị của dx.
    dy = dx.T
    # tinh -1: độ sâu
    gx = cv2.filter2D(image, -1, dx)
    gy = cv2.filter2D(image, -1, dy)

    #
    gs = np.sqrt(np.square(gx) + np.square(gy))
    phis = np.arctan(gy / (gx + 1e-6))
    phis[gx == 0] = np.pi / 2

    argmax_g = gs.argmax(axis=-1)

    # lấy ra g, phi mà tại đó g max
    g = np.take_along_axis(gs, argmax_g[..., None], axis=1)[..., 0]
    phi = np.take_along_axis(phis, argmax_g[..., None], axis=1)[..., 0]
    histogram = np.zeros((g.shape[0] // cell_size, g.shape[1] // cell_size, nbins))
    for i in range(0, g.shape[0] - cell_size + 1, cell_size):
        for j in range(0, g.shape[1] - cell_size + 1, cell_size):
            g_in_square = g[i:i + cell_size, j:j + cell_size]
            phi_in_square = phi[i:i + cell_size, j:j + cell_size]

            bins = np.zeros(9)

            for u in range(0, g_in_square.shape[0]):
                for v in range(0, g_in_square.shape[1]):
                    g_pixel = g_in_square[u, v]
                    phi_pixel = phi_in_square[u, v] * 180 / np.pi
                    bin_index = int(phi_pixel // 20)
                    a = bin_index * 20
                    b = (bin_index + 1) * 20

                    value_1 = (phi_pixel - a) / 20 * g_pixel
                    value_2 = (b - phi_pixel) / 20 * g_pixel

                    bins[bin_index] += value_2
                    bins[(bin_index + 1) % 9] += value_1

            histogram[int(i / cell_size), int(j / cell_size), :] = bins

    t = block_size // cell_size
    hist = []
    for i in range(0, histogram.shape[0] - t + 1):
        for j in range(0, histogram.shape[1] - t + 1):
            block = histogram[i:i + t, j:j + t, :]
            block = block.flatten()
            block /= np.linalg.norm(block) + 1e-6
            hist.append(block)

    hist = np.array(hist)

    return hist.flatten()


# Hàm _extract_object là một hàm để trích xuất đối tượng từ một ảnh.

# Đầu tiên, hàm chuyển ảnh đầu vào thành ảnh xám bằng cách sử dụng hàm cv2.cvtColor của thư viện OpenCV.

# Sau đó, hàm thực hiện việc ngưỡng ảnh để tách vật thể ra khỏi nền bằng cách sử dụng hàm cv2.threshold.
# Khi thực hiện ngưỡng ảnh này, hàm sử dụng ngưỡng 60 và phương pháp Otsu để tìm ngưỡng phù hợp.

# Tiếp theo, hàm thực hiện đảo ngược màu của ảnh bằng cách sử dụng hàm cv2.bitwise_not.

# Sau đó, hàm cắt chỉ vật thể ra khỏi ảnh ban đầu bằng cách tìm vị trí của vật thể trên ảnh.
# Đầu tiên, hàm tính tổng các giá trị trên trục x và y của ảnh đã được ngưỡng. 
# Sau đó, hàm tìm chỉ số đầu tiên và cuối cùng mà tổng đó không bằng 0, đó là chỉ số của vật thể trên các trục x và y.
# Cuối cùng, hàm trả về ảnh chỉ chứa vật thể đã được cắt từ ảnh gốc.

# Để trả về ảnh chứa vật thể đã cắt, hàm sử dụng cú pháp lát cắt trên numpy để chọn các phần tử của ảnh gốc tương ứng với vật thể đã tìm được.
def _extract_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_cpy = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    image_cpy = cv2.bitwise_not(image_cpy)

    # tìm vị trí đối tượng trong ảnh
    sums = image_cpy.sum(axis=0)
    t = np.where(sums != 0)
    x1, x2 = t[0][0], t[0][-1]
    sums = image_cpy.sum(axis=1)
    t = np.where(sums != 0)
    y1, y2 = t[0][0], t[0][-1]

    return image[y1:y2 + 1, x1:x2 + 1]


# Hàm _get_color_mean nhận đầu vào là ảnh object được trích xuất từ ảnh gốc và trả về giá trị trung bình màu của ảnh object đó dưới dạng một numpy array.

# Cụ thể:
# Dòng đầu tiên chuyển đổi không gian màu của ảnh object từ BGR sang LAB để thuận tiện cho việc tính toán các giá trị L*, a*, b*.
# Dòng thứ hai tính giá trị trung bình của các pixel theo chiều rộng, chiều cao của ảnh object. Ta chỉ quan tâm đến 3 kênh màu L*, a*, b* 
# nên lấy phần tử đầu tiên trong vector giá trị trung bình này.
# Dòng thứ ba reshape kết quả về một numpy array có shape là (3,1) để tiện cho việc tính toán ở các hàm tiếp theo. 
# Cuối cùng, hàm trả về giá trị trung bình màu của ảnh object dưới dạng một numpy array có shape là (3,1).
def _get_color_mean(image):
    # Lấy trung bình 3 từng kênh L*, a*, b*
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    obj_color_mean = np.array(lab.mean(axis=(0, 1))[:3])
    obj_color_mean = obj_color_mean.reshape((-1, 1))

    return obj_color_mean

# Hàm _get_color_std(image) được dùng để tính toán độ lệch chuẩn của màu sắc của một đối tượng trong ảnh.

# Giải thích từng dòng code:

# Dòng 1: Chuyển đổi không gian màu của ảnh từ BGR sang LAB để tính toán độ lệch chuẩn của các kênh L*, a*, b*.
# Dòng 2: Tính toán độ lệch chuẩn của màu sắc trên 3 kênh L*, a*, b* của ảnh. Hàm std được sử dụng để tính toán độ lệch chuẩn.
# Dòng 3: Định dạng kết quả trả về thành một mảng 2 chiều với 3 hàng và 1 cột.
# Dòng 5: Trả về giá trị độ lệch chuẩn của màu sắc của đối tượng trong ảnh.
def _get_color_std(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    obj_color_std = np.array(lab.std(axis=(0, 1))[:3])
    obj_color_std = obj_color_std.reshape((-1, 1))

    return obj_color_std


# Hàm _pad_resize(image, shape) có chức năng chuyển đổi kích thước của ảnh đầu vào (image) sao cho kích thước của nó bằng với shape. 
# Trong quá trình thay đổi kích thước, hàm sẽ giữ nguyên tỷ lệ kích thước của ảnh.

# Cụ thể, hàm sẽ thực hiện các bước như sau:

# Sử dụng hàm cv2.resize() để thay đổi kích thước của ảnh đầu vào (image) sao cho kích thước của nó bằng với shape.
# Trả về ảnh đã được thay đổi kích thước.
# Các đối số đầu vào của hàm:

# image: ảnh đầu vào cần thay đổi kích thước.
# shape: kích thước mới của ảnh (tuple) với định dạng (height, width).
# Các đối số đầu ra của hàm:

# ảnh đã được thay đổi kích thước sao cho kích thước của nó bằng với shape.
def _pad_resize(image, shape):
    image = cv2.resize(image, (shape[1], shape[0]))
    return image

# Hàm extract_features được sử dụng để trích xuất đặc trưng từ ảnh đầu vào. 
# Hàm này có các đầu vào là image (ảnh cần trích xuất đặc trưng), shape (kích thước của ảnh đầu ra sau khi được xử lý), và name (tên đối tượng trong ảnh).

# Đầu tiên, hàm kiểm tra xem ảnh đầu vào có phải là ảnh RGB hay không (3 kênh màu) bằng cách kiểm tra số kênh màu của ảnh. Nếu không phải ảnh RGB, hàm sẽ báo lỗi.

# Sau đó, hàm sử dụng hàm _extract_object để tách đối tượng cần nhận diện khỏi ảnh gốc.

# Tiếp theo, hàm sử dụng hàm _get_color_mean và _get_color_std để tính toán giá trị trung bình và độ lệch chuẩn của các kênh màu trong đối tượng.

# Sau khi tính toán xong giá trị trung bình và độ lệch chuẩn của các kênh màu, 
# hàm sử dụng hàm _pad_resize để chuyển đối tượng đã được tách ra về kích thước và độ phân giải cố định và được đệm 1 pixel màu đen quanh nó.

# Cuối cùng, hàm sử dụng hàm _hog để tính toán histogram của đối tượng đã được xử lý. Các giá trị tính toán được trả về dưới dạng một vector đặc trưng của đối tượng.
def extract_features(image, shape, name=""):
    assert image.shape[-1] == 3, "Expected 3 channels, got %d" % image.shape[-1]
    image = _extract_object(image)

    obj_color_mean = _get_color_mean(image)
    obj_color_std = _get_color_std(image)
    # Resize về cùng một cỡ và đệm 1 vòng pixel 0 bên ngoài,
    image = _pad_resize(image, shape)
    # Chồng hog và màu thành 1 vector

    feature = obj_color_mean, obj_color_std, _hog(image, shape)

    return feature




# Hàm to_csv có chức năng trích xuất các đặc trưng từ các hình ảnh trong thư mục "data/train", 
# lưu trữ chúng và tạo một tệp CSV để lưu trữ các đường dẫn đến các tệp được lưu trữ.

# Giải thích từng dòng code:

# train_folder = "data/train": Đường dẫn đến thư mục chứa các hình ảnh huấn luyện.
# shape = (256, 256): Kích thước hình ảnh đầu ra sau khi được resize.
# data = []: Tạo một danh sách rỗng để lưu trữ thông tin về các đặc trưng của hình ảnh.
# for name in os.listdir(train_folder):: Duyệt qua tất cả các tệp trong thư mục huấn luyện.
# img_path = os.path.join(train_folder, name): Xây dựng đường dẫn đầy đủ đến tệp hình ảnh.
# img = cv2.imread(img_path): Đọc hình ảnh bằng OpenCV.
# color_mean, color_std, hog = extract_features(img, shape, img_path): Trích xuất đặc trưng từ hình ảnh bằng hàm extract_features.
# color_mean_output_path = os.path.join(path, 'color_mean_' + name.split('.')[0] + '.npy'): Tạo đường dẫn đến tệp lưu trữ giá trị trung bình màu.
# color_std_output_path = os.path.join(path, 'color_out_' + name.split('.')[0] + '.npy'): Tạo đường dẫn đến tệp lưu trữ độ lệch chuẩn màu.
# hog_output_path = os.path.join(path, 'hog_' + name.split('.')[0] + '.npy'): Tạo đường dẫn đến tệp lưu trữ đặc trưng hog.
# np.save(color_mean_output_path, color_mean): Lưu trữ giá trị trung bình màu vào tệp.
# np.save(color_std_output_path, color_std): Lưu trữ độ lệch chuẩn màu vào tệp.
# np.save(hog_output_path, hog): Lưu trữ đặc trưng hog vào tệp.
# data.append([name, color_mean_output_path, color_std_output_path, hog_output_path]): Thêm thông tin về các đặc trưng của hình ảnh vào danh sách data.
# df = pd.DataFrame(data, columns=['name', 'color_mean', 'color_std', 'hog']): Tạo DataFrame từ danh sách data và gán tên cho các cột.
# df.to_csv('data.csv'): Lưu trữ DataFrame thành tệp CSV.
def to_csv(path):
    train_folder = "data/train"
    shape = (256, 256)
    data = []

    for name in os.listdir(train_folder):
        img_path = os.path.join(train_folder, name)
        img = cv2.imread(img_path)
        color_mean, color_std, hog = extract_features(img, shape, img_path)
        color_mean_output_path = os.path.join(path, 'color_mean_' + name.split('.')[0] + '.npy')
        color_std_output_path = os.path.join(path, 'color_out_' + name.split('.')[0] + '.npy')
        hog_output_path = os.path.join(path, 'hog_' + name.split('.')[0] + '.npy')

        np.save(color_mean_output_path, color_mean)
        np.save(color_std_output_path, color_std)
        np.save(hog_output_path, hog)

        data.append([name, color_mean_output_path, color_std_output_path, hog_output_path])

    df = pd.DataFrame(data, columns=['name', 'color_mean', 'color_std', 'hog'])
    df.to_csv('data.csv')
