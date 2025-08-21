# Khai Thác NSL-KDD Dataset Bằng Linear Regression

Dự án này thực hiện khai thác bộ dữ liệu NSL-KDD (Network Security Laboratory - Knowledge Discovery and Data Mining) sử dụng Linear Regression để dự đoán các loại tấn công mạng.

## 🎯 Mục Tiêu

- Áp dụng Linear Regression cho bài toán Network Intrusion Detection
- So sánh hiệu suất của các phương pháp regularization (OLS, Ridge, Lasso)
- Đánh giá tác động của categorical features
- Thử nghiệm các phương pháp encoding target variable

## 📊 Dataset

**NSL-KDD** là phiên bản cải tiến của KDD Cup 1999 dataset:
- **Training set**: 25,192 samples
- **Test set**: 22,544 samples
- **Features**: 41 features + class + difficulty
- **Classes**: 40 loại tấn công khác nhau
- **Categories**: Normal, DoS, Probe, R2L, U2R

## 🚀 Cách Sử Dụng

### Quick Start
```bash
# Chạy toàn bộ demo
python3 run_demo.py

# Hoặc chỉ chạy phân tích cơ bản
python3 run_demo.py --basic

# Hoặc chỉ chạy phân tích cải tiến
python3 run_demo.py --improved
```

### Manual Run
```bash
# 1. Tải dữ liệu
python3 src/download_data.py

# 2. Chạy phân tích cơ bản (numerical features only)
python3 notebooks/basic_linear_regression.py

# 3. Chạy phân tích cải tiến (+ categorical features)
python3 notebooks/improved_linear_regression.py
```

## 📋 Requirements

```bash
pip3 install pandas numpy matplotlib
```

## 📁 Cấu Trúc Dự Án

```
ttn/
├── README.md                           # Hướng dẫn sử dụng
├── requirements.txt                    # Thư viện cần thiết
├── run_demo.py                        # Script demo chính
├── src/
│   └── download_data.py               # Tải dữ liệu NSL-KDD
├── notebooks/
│   ├── basic_linear_regression.py     # Phân tích cơ bản
│   └── improved_linear_regression.py  # Phân tích cải tiến
├── data/                              # Dữ liệu NSL-KDD
│   ├── KDDTrain+_20Percent.txt
│   ├── KDDTest+.txt
│   └── column_names.txt
└── results/
    └── nsl_kdd_linear_regression_report.md  # Báo cáo chi tiết
```

## 🔬 Phương Pháp

### 1. Tiền Xử Lý Dữ Liệu
- **One-Hot Encoding** cho categorical features (protocol_type, service, flag)
- **StandardScaler** cho numerical features
- **Label Encoding** và **Severity Encoding** cho target variable

### 2. Mô Hình Linear Regression
- **Ordinary Least Squares (OLS)**
- **Ridge Regression** với các giá trị α khác nhau
- **Implementation từ đầu** sử dụng Normal Equation

### 3. Đánh Giá
- **R² Score** (Coefficient of Determination)
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Visualization** với matplotlib

## 📈 Kết Quả

### Phiên Bản Cơ Bản (Numerical Features Only)
- **Features**: 38 numerical features
- **Test R²**: 0.1017
- **Test MSE**: 46.0521

### Phiên Bản Cải Tiến (+ Categorical Features)
- **Features**: 120 features (38 numerical + 81 categorical + 1 bias)
- **Test R²**: 0.1768
- **Test MSE**: 42.2014
- **Cải thiện**: +73.9%

### Mô Hình Tốt Nhất
- **Ridge Regression** với α = 10.0
- **Simple Label Encoding** cho target
- **Test R² Score**: 0.1768

## 📊 Visualization

Dự án tạo ra các biểu đồ:
- **Actual vs Predicted** scatter plots
- **Residuals** analysis
- **Feature Importance** (top 20 features)
- **Model Comparison** bar charts
- **Class Distribution** histograms

## 💡 Nhận Xét

### Điểm Mạnh
✅ Regularization hiệu quả giải quyết overfitting  
✅ Categorical features cải thiện đáng kể hiệu suất  
✅ Framework dễ mở rộng và tùy chỉnh  
✅ Implementation từ đầu giúp hiểu sâu thuật toán  

### Điểm Yếu
⚠️ R² Score thấp (0.1768) cho thấy linear model không phù hợp  
⚠️ NSL-KDD có thể cần non-linear approaches  
⚠️ Severity encoding không hiệu quả như mong đợi  

### Khuyến Nghị Cải Thiện
- **Classification** thay vì regression
- **Feature Engineering** (polynomial, interactions)
- **Advanced Models** (Random Forest, SVM, Neural Networks)
- **Ensemble Methods**

## 🔍 Chi Tiết Kỹ Thuật

### Linear Regression Implementation
```python
# Normal Equation: w = (X^T X + αI)^-1 X^T y
XtX = np.dot(X.T, X)
I = np.eye(XtX.shape[0])
XtX_reg = XtX + alpha * I
weights = np.dot(np.linalg.inv(XtX_reg), np.dot(X.T, y))
```

### One-Hot Encoding
```python
# Combine train và test để đảm bảo consistent encoding
combined = pd.concat([train_categorical, test_categorical])
encoded = pd.get_dummies(combined, prefix=categorical_features)
```

### Regularization
- **Ridge (L2)**: Thêm α||w||² vào loss function
- **Lasso (L1)**: Thêm α||w||₁ vào loss function (trong phiên bản đầy đủ)

## 📚 Tài Liệu Tham Khảo

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [KDD Cup 1999](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- Elements of Statistical Learning
- Pattern Recognition and Machine Learning

## 👨‍💻 Tác Giả

Developed by **Augment Agent** - AI Assistant for Code Analysis and Development

## 📄 License

This project is for educational purposes. NSL-KDD dataset follows its original license terms.

---

**Lưu ý**: Dự án này chứng minh rằng Linear Regression có thể áp dụng cho NSL-KDD dataset nhưng không phải là phương pháp tối ưu. Để có hiệu suất tốt hơn, nên xem xét các phương pháp classification hoặc advanced machine learning models.
