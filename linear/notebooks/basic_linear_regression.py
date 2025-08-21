# Khai thác NSL-KDD Dataset bằng Linear Regression (Basic Implementation)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== KHAI THÁC NSL-KDD BẰNG LINEAR REGRESSION ===")

# Định nghĩa tên cột
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'difficulty'
]

print(f"Tổng số features: {len(column_names)}")

# Tải dữ liệu
print("\n1. TẢI DỮ LIỆU")
try:
    train_data = pd.read_csv('data/KDDTrain+_20Percent.txt', names=column_names, header=None)
    test_data = pd.read_csv('data/KDDTest+.txt', names=column_names, header=None)
    print(f"✓ Training data: {train_data.shape}")
    print(f"✓ Test data: {test_data.shape}")
except Exception as e:
    print(f"✗ Lỗi tải dữ liệu: {e}")
    exit(1)

# Phân tích dữ liệu cơ bản
print("\n2. PHÂN TÍCH DỮ LIỆU CƠ BẢN")
print("Top 10 classes trong training set:")
class_counts = train_data['class'].value_counts()
print(class_counts.head(10))

print("\nTop 10 classes trong test set:")
test_class_counts = test_data['class'].value_counts()
print(test_class_counts.head(10))

# Tiền xử lý dữ liệu
print("\n3. TIỀN XỬ LÝ DỮ LIỆU")

# Chọn numerical features (bỏ categorical và target)
categorical_features = ['protocol_type', 'service', 'flag', 'class', 'difficulty']
numerical_features = [col for col in column_names if col not in categorical_features]

print(f"Numerical features: {len(numerical_features)}")
print(f"Features: {numerical_features[:10]}...")

# Lấy numerical data
X_train_num = train_data[numerical_features].values
X_test_num = test_data[numerical_features].values

print(f"X_train shape: {X_train_num.shape}")
print(f"X_test shape: {X_test_num.shape}")

# Xử lý target variable - encode classes thành số
unique_classes = sorted(list(set(train_data['class'].unique()) | set(test_data['class'].unique())))
class_to_num = {cls: i for i, cls in enumerate(unique_classes)}

y_train = np.array([class_to_num[cls] for cls in train_data['class']])
y_test = np.array([class_to_num[cls] for cls in test_data['class']])

print(f"Số classes: {len(unique_classes)}")
print(f"Target range - Train: [{y_train.min()}, {y_train.max()}]")
print(f"Target range - Test: [{y_test.min()}, {y_test.max()}]")

# Chuẩn hóa dữ liệu
print("\n4. CHUẨN HÓA DỮ LIỆU")
X_train_mean = np.mean(X_train_num, axis=0)
X_train_std = np.std(X_train_num, axis=0)

# Tránh chia cho 0
X_train_std[X_train_std == 0] = 1

X_train_scaled = (X_train_num - X_train_mean) / X_train_std
X_test_scaled = (X_test_num - X_train_mean) / X_train_std

print(f"Scaled data shape - Train: {X_train_scaled.shape}")
print(f"Mean after scaling: {np.mean(X_train_scaled):.6f}")
print(f"Std after scaling: {np.std(X_train_scaled):.6f}")

# Thêm bias term (intercept)
X_train_final = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
X_test_final = np.column_stack([np.ones(X_test_scaled.shape[0]), X_test_scaled])

print(f"Final feature shape (with bias): {X_train_final.shape}")

# Implement Linear Regression
print("\n5. HUẤN LUYỆN LINEAR REGRESSION")

class SimpleLinearRegression:
    def __init__(self):
        self.weights = None
        
    def fit(self, X, y):
        """Huấn luyện mô hình bằng Normal Equation: w = (X^T X)^-1 X^T y"""
        try:
            # Normal equation
            XtX = np.dot(X.T, X)
            XtX_inv = np.linalg.inv(XtX)
            Xty = np.dot(X.T, y)
            self.weights = np.dot(XtX_inv, Xty)
            return True
        except np.linalg.LinAlgError:
            print("Lỗi: Ma trận không khả nghịch, sử dụng pseudo-inverse")
            try:
                self.weights = np.dot(np.linalg.pinv(X), y)
                return True
            except:
                return False
    
    def predict(self, X):
        """Dự đoán"""
        if self.weights is None:
            raise ValueError("Mô hình chưa được huấn luyện")
        return np.dot(X, self.weights)
    
    def score(self, X, y):
        """Tính R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Huấn luyện mô hình
model = SimpleLinearRegression()
success = model.fit(X_train_final, y_train)

if not success:
    print("✗ Không thể huấn luyện mô hình")
    exit(1)

print(f"✓ Đã huấn luyện thành công")
print(f"Số weights: {len(model.weights)}")

# Đánh giá mô hình
print("\n6. ĐÁNH GIÁ MÔ HÌNH")

# Dự đoán
y_train_pred = model.predict(X_train_final)
y_test_pred = model.predict(X_test_final)

# Tính toán metrics
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return mse, mae, r2

train_mse, train_mae, train_r2 = calculate_metrics(y_train, y_train_pred)
test_mse, test_mae, test_r2 = calculate_metrics(y_test, y_test_pred)

print("KẾT QUẢ HUẤN LUYỆN:")
print(f"  Train MSE: {train_mse:.4f}")
print(f"  Train MAE: {train_mae:.4f}")
print(f"  Train R²:  {train_r2:.4f}")

print("\nKẾT QUẢ KIỂM THỬ:")
print(f"  Test MSE:  {test_mse:.4f}")
print(f"  Test MAE:  {test_mae:.4f}")
print(f"  Test R²:   {test_r2:.4f}")

# Visualize kết quả
print("\n7. VISUALIZATION")

plt.figure(figsize=(15, 10))

# Subplot 1: Actual vs Predicted (Training)
plt.subplot(2, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', s=1)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Training Set\nR² = {train_r2:.4f}')
plt.grid(True, alpha=0.3)

# Subplot 2: Actual vs Predicted (Test)
plt.subplot(2, 3, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, color='green', s=1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Test Set\nR² = {test_r2:.4f}')
plt.grid(True, alpha=0.3)

# Subplot 3: Residuals (Test)
plt.subplot(2, 3, 3)
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.5, color='red', s=1)
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot (Test)')
plt.grid(True, alpha=0.3)

# Subplot 4: Histogram of residuals
plt.subplot(2, 3, 4)
plt.hist(residuals, bins=50, alpha=0.7, color='orange')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)

# Subplot 5: Feature weights (top 20)
plt.subplot(2, 3, 5)
feature_names = ['bias'] + numerical_features
weights_abs = np.abs(model.weights)
top_indices = np.argsort(weights_abs)[-20:]
top_weights = model.weights[top_indices]
top_names = [feature_names[i] for i in top_indices]

plt.barh(range(len(top_weights)), top_weights, color='purple')
plt.yticks(range(len(top_weights)), top_names)
plt.xlabel('Weight Value')
plt.title('Top 20 Feature Weights')
plt.grid(True, alpha=0.3)

# Subplot 6: Metrics comparison
plt.subplot(2, 3, 6)
metrics = ['MSE', 'MAE', 'R²']
train_metrics = [train_mse, train_mae, train_r2]
test_metrics = [test_mse, test_mae, test_r2]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_metrics, width, label='Train', color='lightblue')
plt.bar(x + width/2, test_metrics, width, label='Test', color='lightcoral')

plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Train vs Test Metrics')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Kết luận
print("\n8. KẾT LUẬN")
print("="*50)
print(f"✓ Đã huấn luyện thành công Linear Regression trên NSL-KDD")
print(f"✓ Số features sử dụng: {len(numerical_features)}")
print(f"✓ Số classes: {len(unique_classes)}")
print(f"✓ Test R² Score: {test_r2:.4f}")

if test_r2 > 0.8:
    print("✓ Mô hình có hiệu suất rất tốt!")
elif test_r2 > 0.6:
    print("✓ Mô hình có hiệu suất khá tốt")
elif test_r2 > 0.4:
    print("⚠ Mô hình có hiệu suất trung bình")
else:
    print("⚠ Mô hình cần cải thiện")

print("\nNHẬN XÉT:")
print("- Linear Regression có thể áp dụng cho NSL-KDD dataset")
print("- Chỉ sử dụng numerical features đã cho kết quả khả quan")
print("- Có thể cải thiện bằng cách thêm categorical features")
print("- Regularization (Ridge/Lasso) có thể giúp tránh overfitting")

print("\n=== HOÀN THÀNH PHÂN TÍCH ===")
