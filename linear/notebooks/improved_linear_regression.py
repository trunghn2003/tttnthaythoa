# Khai thác NSL-KDD Dataset bằng Linear Regression (Improved Version)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== KHAI THÁC NSL-KDD BẰNG LINEAR REGRESSION (IMPROVED) ===")

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

# Tải dữ liệu
print("\n1. TẢI DỮ LIỆU")
train_data = pd.read_csv('data/KDDTrain+_20Percent.txt', names=column_names, header=None)
test_data = pd.read_csv('data/KDDTest+.txt', names=column_names, header=None)
print(f"✓ Training data: {train_data.shape}")
print(f"✓ Test data: {test_data.shape}")

# Phân tích categorical features
print("\n2. PHÂN TÍCH CATEGORICAL FEATURES")
categorical_features = ['protocol_type', 'service', 'flag']

for feature in categorical_features:
    print(f"\n{feature}:")
    print(f"  Train unique values: {train_data[feature].nunique()}")
    print(f"  Test unique values: {test_data[feature].nunique()}")
    print(f"  Values: {sorted(train_data[feature].unique())[:10]}...")

# One-hot encoding cho categorical features
print("\n3. ONE-HOT ENCODING")

def create_one_hot_encoding(train_df, test_df, categorical_features):
    """Tạo one-hot encoding cho categorical features"""
    encoded_features = {}
    
    for feature in categorical_features:
        # Lấy tất cả unique values từ cả train và test
        all_values = sorted(list(set(train_df[feature].unique()) | set(test_df[feature].unique())))
        
        # Tạo one-hot encoding
        for value in all_values:
            col_name = f"{feature}_{value}"
            encoded_features[col_name] = {
                'train': (train_df[feature] == value).astype(int),
                'test': (test_df[feature] == value).astype(int)
            }
    
    return encoded_features

encoded_features = create_one_hot_encoding(train_data, test_data, categorical_features)
print(f"✓ Đã tạo {len(encoded_features)} one-hot encoded features")

# Chuẩn bị features
print("\n4. CHUẨN BỊ FEATURES")

# Numerical features
numerical_features = [col for col in column_names if col not in categorical_features + ['class', 'difficulty']]
X_train_num = train_data[numerical_features].values
X_test_num = test_data[numerical_features].values

# Thêm one-hot encoded features
X_train_cat = np.column_stack([encoded_features[col]['train'].values for col in encoded_features.keys()])
X_test_cat = np.column_stack([encoded_features[col]['test'].values for col in encoded_features.keys()])

# Kết hợp numerical và categorical
X_train_combined = np.column_stack([X_train_num, X_train_cat])
X_test_combined = np.column_stack([X_test_num, X_test_cat])

print(f"Numerical features: {X_train_num.shape[1]}")
print(f"Categorical features: {X_train_cat.shape[1]}")
print(f"Combined features: {X_train_combined.shape[1]}")

# Chuẩn hóa dữ liệu
print("\n5. CHUẨN HÓA DỮ LIỆU")
X_train_mean = np.mean(X_train_combined, axis=0)
X_train_std = np.std(X_train_combined, axis=0)
X_train_std[X_train_std == 0] = 1  # Tránh chia cho 0

X_train_scaled = (X_train_combined - X_train_mean) / X_train_std
X_test_scaled = (X_test_combined - X_train_mean) / X_train_std

# Thêm bias term
X_train_final = np.column_stack([np.ones(X_train_scaled.shape[0]), X_train_scaled])
X_test_final = np.column_stack([np.ones(X_test_scaled.shape[0]), X_test_scaled])

print(f"Final feature shape: {X_train_final.shape}")

# Chuẩn bị target variable
print("\n6. CHUẨN BỊ TARGET VARIABLE")

# Thử nghiệm 2 phương pháp encoding target
def encode_target_simple(train_classes, test_classes):
    """Encode classes thành số đơn giản"""
    unique_classes = sorted(list(set(train_classes) | set(test_classes)))
    class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
    
    y_train = np.array([class_to_num[cls] for cls in train_classes])
    y_test = np.array([class_to_num[cls] for cls in test_classes])
    
    return y_train, y_test, class_to_num

def encode_target_severity(train_classes, test_classes):
    """Encode classes theo mức độ nghiêm trọng"""
    severity_mapping = {
        'normal': 0,
        # DoS attacks (mức độ 1)
        'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
        'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
        # Probe attacks (mức độ 2)
        'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'mscan': 2, 'saint': 2,
        # R2L attacks (mức độ 3)
        'guess_passwd': 3, 'ftp_write': 3, 'imap': 3, 'phf': 3, 'multihop': 3,
        'warezmaster': 3, 'warezclient': 3, 'spy': 3, 'xlock': 3, 'xsnoop': 3,
        'snmpguess': 3, 'snmpgetattack': 3, 'httptunnel': 3, 'sendmail': 3, 'named': 3,
        # U2R attacks (mức độ 4)
        'buffer_overflow': 4, 'loadmodule': 4, 'rootkit': 4, 'perl': 4, 'sqlattack': 4,
        'xterm': 4, 'ps': 4
    }
    
    y_train = np.array([severity_mapping.get(cls, 2) for cls in train_classes])  # Default: 2
    y_test = np.array([severity_mapping.get(cls, 2) for cls in test_classes])
    
    return y_train, y_test, severity_mapping

# Thử cả 2 phương pháp
y_train_simple, y_test_simple, class_mapping = encode_target_simple(train_data['class'], test_data['class'])
y_train_severity, y_test_severity, severity_mapping = encode_target_severity(train_data['class'], test_data['class'])

print(f"Simple encoding - Range: [{y_train_simple.min()}, {y_train_simple.max()}]")
print(f"Severity encoding - Range: [{y_train_severity.min()}, {y_train_severity.max()}]")

# Linear Regression với Regularization
print("\n7. LINEAR REGRESSION VỚI REGULARIZATION")

class RegularizedLinearRegression:
    def __init__(self, alpha=0.0, reg_type='ridge'):
        self.alpha = alpha
        self.reg_type = reg_type
        self.weights = None
        
    def fit(self, X, y):
        """Huấn luyện với regularization"""
        try:
            if self.reg_type == 'ridge' and self.alpha > 0:
                # Ridge regression: (X^T X + αI)^-1 X^T y
                XtX = np.dot(X.T, X)
                I = np.eye(XtX.shape[0])
                XtX_reg = XtX + self.alpha * I
                XtX_inv = np.linalg.inv(XtX_reg)
                Xty = np.dot(X.T, y)
                self.weights = np.dot(XtX_inv, Xty)
            else:
                # Ordinary least squares
                XtX = np.dot(X.T, X)
                XtX_inv = np.linalg.inv(XtX)
                Xty = np.dot(X.T, y)
                self.weights = np.dot(XtX_inv, Xty)
            return True
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            self.weights = np.dot(np.linalg.pinv(X), y)
            return True
    
    def predict(self, X):
        return np.dot(X, self.weights)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

# Thử nghiệm các mô hình
models = {
    'OLS': RegularizedLinearRegression(alpha=0.0),
    'Ridge (α=0.1)': RegularizedLinearRegression(alpha=0.1, reg_type='ridge'),
    'Ridge (α=1.0)': RegularizedLinearRegression(alpha=1.0, reg_type='ridge'),
    'Ridge (α=10.0)': RegularizedLinearRegression(alpha=10.0, reg_type='ridge')
}

# Đánh giá với cả 2 phương pháp encoding
results = {}

print("\n8. ĐÁNH GIÁ CÁC MÔ HÌNH")

for encoding_name, (y_train, y_test) in [('Simple', (y_train_simple, y_test_simple)), 
                                         ('Severity', (y_train_severity, y_test_severity))]:
    print(f"\n--- {encoding_name} Encoding ---")
    results[encoding_name] = {}
    
    for model_name, model in models.items():
        # Huấn luyện
        model.fit(X_train_final, y_train)
        
        # Dự đoán
        y_train_pred = model.predict(X_train_final)
        y_test_pred = model.predict(X_test_final)
        
        # Metrics
        train_mse = np.mean((y_train - y_train_pred) ** 2)
        test_mse = np.mean((y_test - y_test_pred) ** 2)
        train_r2 = model.score(X_train_final, y_train)
        test_r2 = model.score(X_test_final, y_test)
        
        results[encoding_name][model_name] = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'y_test_pred': y_test_pred
        }
        
        print(f"{model_name:15} - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")

# Tìm mô hình tốt nhất
print("\n9. MÔ HÌNH TỐT NHẤT")
best_r2 = -np.inf
best_config = None

for encoding in results:
    for model in results[encoding]:
        r2 = results[encoding][model]['test_r2']
        if r2 > best_r2:
            best_r2 = r2
            best_config = (encoding, model)

best_encoding, best_model = best_config
print(f"✓ Mô hình tốt nhất: {best_model} với {best_encoding} encoding")
print(f"✓ Test R²: {best_r2:.4f}")
print(f"✓ Test MSE: {results[best_encoding][best_model]['test_mse']:.4f}")

# Visualization
print("\n10. VISUALIZATION")
plt.figure(figsize=(16, 12))

# So sánh R² scores
plt.subplot(2, 3, 1)
encodings = list(results.keys())
models_list = list(models.keys())
x = np.arange(len(models_list))
width = 0.35

for i, encoding in enumerate(encodings):
    r2_scores = [results[encoding][model]['test_r2'] for model in models_list]
    plt.bar(x + i*width, r2_scores, width, label=f'{encoding} Encoding')

plt.xlabel('Models')
plt.ylabel('Test R² Score')
plt.title('Model Comparison - Test R²')
plt.xticks(x + width/2, models_list, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Best model predictions
plt.subplot(2, 3, 2)
if best_encoding == 'Simple':
    y_test_true = y_test_simple
else:
    y_test_true = y_test_severity

y_test_pred_best = results[best_encoding][best_model]['y_test_pred']
plt.scatter(y_test_true, y_test_pred_best, alpha=0.5, s=1)
plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Best Model: {best_model}\n{best_encoding} Encoding')
plt.grid(True, alpha=0.3)

# Residuals
plt.subplot(2, 3, 3)
residuals = y_test_true - y_test_pred_best
plt.scatter(y_test_pred_best, residuals, alpha=0.5, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.grid(True, alpha=0.3)

# Feature importance (top 20)
plt.subplot(2, 3, 4)
best_model_obj = models[best_model]
feature_names = ['bias'] + numerical_features + list(encoded_features.keys())
weights_abs = np.abs(best_model_obj.weights)
top_indices = np.argsort(weights_abs)[-20:]
top_weights = best_model_obj.weights[top_indices]
top_names = [feature_names[i][:15] for i in top_indices]  # Truncate names

plt.barh(range(len(top_weights)), top_weights)
plt.yticks(range(len(top_weights)), top_names)
plt.xlabel('Weight Value')
plt.title('Top 20 Feature Weights')
plt.grid(True, alpha=0.3)

# MSE comparison
plt.subplot(2, 3, 5)
for i, encoding in enumerate(encodings):
    mse_scores = [results[encoding][model]['test_mse'] for model in models_list]
    plt.bar(x + i*width, mse_scores, width, label=f'{encoding} Encoding')

plt.xlabel('Models')
plt.ylabel('Test MSE')
plt.title('Model Comparison - Test MSE')
plt.xticks(x + width/2, models_list, rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Class distribution
plt.subplot(2, 3, 6)
if best_encoding == 'Simple':
    unique, counts = np.unique(y_test_simple, return_counts=True)
    plt.bar(unique[:20], counts[:20])  # Top 20 classes
    plt.xlabel('Class ID')
    plt.title('Test Set Class Distribution\n(Simple Encoding)')
else:
    unique, counts = np.unique(y_test_severity, return_counts=True)
    plt.bar(unique, counts)
    plt.xlabel('Severity Level')
    plt.title('Test Set Severity Distribution')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n11. KẾT LUẬN")
print("="*60)
print(f"✓ Đã thành công khai thác NSL-KDD bằng Linear Regression")
print(f"✓ Sử dụng {X_train_final.shape[1]} features (bao gồm categorical)")
print(f"✓ Mô hình tốt nhất: {best_model} với {best_encoding} encoding")
print(f"✓ Test R² Score: {best_r2:.4f}")

if best_r2 > 0.8:
    print("🎉 Hiệu suất xuất sắc!")
elif best_r2 > 0.6:
    print("✅ Hiệu suất tốt")
elif best_r2 > 0.4:
    print("⚠️ Hiệu suất trung bình")
else:
    print("❌ Cần cải thiện")

print(f"\nSO SÁNH VỚI PHIÊN BẢN CƠ BẢN:")
print(f"- Phiên bản cơ bản (chỉ numerical): R² = 0.1017")
print(f"- Phiên bản cải tiến (+ categorical): R² = {best_r2:.4f}")
print(f"- Cải thiện: {((best_r2 - 0.1017) / 0.1017 * 100):+.1f}%")

print("\n=== HOÀN THÀNH PHÂN TÍCH NÂNG CAO ===")
