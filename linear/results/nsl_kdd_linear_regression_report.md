# Báo Cáo Khai Thác NSL-KDD Dataset Bằng Linear Regression

## Tổng Quan Dự Án

Dự án này thực hiện khai thác bộ dữ liệu NSL-KDD (Network Security Laboratory - Knowledge Discovery and Data Mining) sử dụng Linear Regression để dự đoán các loại tấn công mạng.

## Thông Tin Dataset

- **Dataset**: NSL-KDD (phiên bản cải tiến của KDD Cup 1999)
- **Training set**: 25,192 samples
- **Test set**: 22,544 samples  
- **Số features gốc**: 43 (bao gồm 41 features + class + difficulty)
- **Số classes**: 40 loại tấn công khác nhau
- **Categorical features**: 3 (protocol_type, service, flag)
- **Numerical features**: 38

## Phương Pháp Thực Hiện

### 1. Tiền Xử Lý Dữ Liệu

#### Xử Lý Categorical Features
- **One-Hot Encoding** cho 3 categorical features:
  - `protocol_type`: 3 values (icmp, tcp, udp)
  - `service`: 66 values (HTTP, FTP, SSH, etc.)
  - `flag`: 11 values (SF, S0, REJ, etc.)
- **Tổng số features sau encoding**: 81 categorical features

#### Chuẩn Hóa Dữ Liệu
- Sử dụng **StandardScaler** (Z-score normalization)
- Áp dụng cho cả numerical và categorical features
- **Tổng số features cuối cùng**: 120 (38 numerical + 81 categorical + 1 bias)

#### Xử Lý Target Variable
Thử nghiệm 2 phương pháp encoding:

1. **Simple Encoding**: Gán số thứ tự cho mỗi class (0-39)
2. **Severity Encoding**: Gán điểm theo mức độ nghiêm trọng:
   - Normal: 0
   - DoS attacks: 1
   - Probe attacks: 2  
   - R2L attacks: 3
   - U2R attacks: 4

### 2. Mô Hình Linear Regression

#### Các Phương Pháp Thử Nghiệm
1. **Ordinary Least Squares (OLS)**: Không regularization
2. **Ridge Regression**: L2 regularization với α = 0.1, 1.0, 10.0
3. **Lasso Regression**: L1 regularization (trong phiên bản đầy đủ)

#### Implementation
- Sử dụng **Normal Equation**: w = (X^T X + αI)^-1 X^T y
- Fallback sang **Pseudo-inverse** khi ma trận không khả nghịch
- Thêm **bias term** cho intercept

## Kết Quả Thực Nghiệm

### Phiên Bản Cơ Bản (Chỉ Numerical Features)
- **Features sử dụng**: 38 numerical features
- **Mô hình tốt nhất**: Linear Regression
- **Test R² Score**: 0.1017
- **Test MSE**: 46.0521
- **Test MAE**: 4.1016

### Phiên Bản Cải Tiến (Bao Gồm Categorical Features)
- **Features sử dụng**: 120 features (38 numerical + 81 categorical + 1 bias)
- **Mô hình tốt nhất**: Ridge Regression (α=10.0) với Simple Encoding
- **Test R² Score**: 0.1768
- **Test MSE**: 42.2014
- **Cải thiện**: +73.9% so với phiên bản cơ bản

### So Sánh Các Phương Pháp Encoding

| Encoding Method | Best Model | Test R² | Test MSE |
|----------------|------------|---------|----------|
| Simple Encoding | Ridge (α=10.0) | 0.1768 | 42.2014 |
| Severity Encoding | Ridge (α=10.0) | -0.1143 | 1.1962 |

### So Sánh Regularization

| Model | Simple Encoding R² | Severity Encoding R² |
|-------|-------------------|---------------------|
| OLS | -14,432,148,305,272,612,864 | -23,612,894,260,276,707,328 |
| Ridge (α=0.1) | 0.1758 | -0.1158 |
| Ridge (α=1.0) | 0.1759 | -0.1156 |
| Ridge (α=10.0) | **0.1768** | -0.1143 |

## Phân Tích Kết Quả

### Điểm Mạnh
1. **Regularization hiệu quả**: Ridge regression giải quyết được vấn đề overfitting của OLS
2. **Categorical features quan trọng**: Cải thiện 73.9% hiệu suất khi thêm categorical features
3. **Ổn định**: Mô hình không bị diverge với regularization phù hợp
4. **Khả năng mở rộng**: Framework có thể dễ dàng thêm features mới

### Điểm Yếu
1. **R² Score thấp**: 0.1768 cho thấy mô hình chỉ giải thích được ~17.7% variance
2. **Linear assumption**: NSL-KDD có thể có quan hệ phi tuyến phức tạp
3. **Severity encoding kém hiệu quả**: Có thể do mapping không phù hợp
4. **Overfitting với OLS**: Ma trận không khả nghịch do số features lớn

### Nguyên Nhân R² Score Thấp
1. **Bản chất bài toán**: Network intrusion detection có thể cần non-linear models
2. **Feature engineering**: Cần thêm feature interactions, polynomial features
3. **Target encoding**: Simple label encoding có thể không phù hợp cho regression
4. **Outliers**: Dataset có thể chứa nhiều outliers ảnh hưởng đến linear model

## Khuyến Nghị Cải Thiện

### 1. Feature Engineering
- Thêm **polynomial features** và **feature interactions**
- **Feature selection** để loại bỏ features không quan trọng
- **Principal Component Analysis (PCA)** để giảm chiều
- **Log transformation** cho skewed features

### 2. Advanced Models
- **Polynomial Regression** cho non-linear relationships
- **Support Vector Regression (SVR)** với RBF kernel
- **Random Forest Regression** cho feature interactions
- **Neural Networks** cho complex patterns

### 3. Alternative Approaches
- **Multi-class Classification** thay vì regression
- **Binary Classification** (normal vs attack)
- **Hierarchical Classification** (attack type → specific attack)
- **Ensemble Methods** kết hợp multiple models

### 4. Data Preprocessing
- **Outlier detection và removal**
- **Feature scaling methods** khác (MinMax, Robust)
- **Handling imbalanced classes** với SMOTE
- **Cross-validation** cho model selection

## Kết Luận

Dự án đã thành công thực hiện khai thác NSL-KDD dataset bằng Linear Regression với những kết quả chính:

1. ✅ **Hoàn thành mục tiêu**: Áp dụng Linear Regression cho NSL-KDD
2. ✅ **Cải thiện hiệu suất**: +73.9% khi thêm categorical features  
3. ✅ **Regularization hiệu quả**: Ridge regression giải quyết overfitting
4. ⚠️ **Hiệu suất hạn chế**: R² = 0.1768 cho thấy cần approaches khác

**Nhận xét cuối cùng**: Linear Regression có thể áp dụng cho NSL-KDD nhưng không phải là phương pháp tối ưu. Bài toán network intrusion detection có bản chất phức tạp, phi tuyến, phù hợp hơn với classification models hoặc advanced regression techniques.

## Files Tạo Ra

1. `src/download_data.py` - Script tải dữ liệu NSL-KDD
2. `notebooks/basic_linear_regression.py` - Phiên bản cơ bản (numerical only)
3. `notebooks/improved_linear_regression.py` - Phiên bản cải tiến (+ categorical)
4. `results/nsl_kdd_linear_regression_report.md` - Báo cáo này

## Tài Liệu Tham Khảo

- NSL-KDD Dataset: https://www.unb.ca/cic/datasets/nsl.html
- KDD Cup 1999: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- Linear Regression Theory: Elements of Statistical Learning
- Regularization Methods: Pattern Recognition and Machine Learning
