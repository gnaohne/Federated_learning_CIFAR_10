# Federated_learning_CIFAR_10

## Danh sách công việc cần làm

## Đọc dữ liệu và train trước với mô hình Hồi quy
- [x] Đọc dữ liệu và transform thành tensor
- [x] Train mô hình Hồi quy với dữ liệu không embeded.
- [] Sử dụng CNN để embeded dữ liệu
- [] Train mô hình Hồi quy với dữ liệu embeded.

### Chuẩn bị data, mô hình cho các Client
- [] Tách bộ dữ liệu, class thành các phần nhỏ cho các Client.
    - [] Thiết kế thành các file data.

### Train từng mô hình với dữ liệu thực tế của mỗi Client
- [] Client nhận mô hình được broadcast từ Server.
- [] Client đọc dữ liệu của mình.
- [] Train mô hình với dữ liệu của mình.
- [] Gửi bộ tham số mô hình đã train qua cho Server.

### Server tổng hợp dữ liệu và cập nhật mô hình
- [] Init mô hình ban đầu 
    - [] Lựa chọn các tham số phù hợp
- [] Gửi tên mô hình và các tham số ban đầu cho Client.
- [] Server nhận các bộ tham số từ Client.
- [] Tính toán trung bình các tham số và cập nhật mô hình.

## Đánh giá mô hình
### Đánh giá mô hình trên tập dữ liệu test
- [] Đánh giá mô hình trên tập dữ liệu test.
### Chương trình Server dự đoán với dữ liệu mới
- [] Server nhận dữ liệu mới.
- [] Dự đoán với mô hình đã train.
- [] Trả kết quả dự đoán cho end-user.
