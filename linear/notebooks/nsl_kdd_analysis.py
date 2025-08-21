# %% [markdown]
# # Khai thác NSL-KDD Dataset bằng Linear Regression
# 
# Script này thực hiện phân tích và khai thác bộ dữ liệu NSL-KDD sử dụng Linear Regression để dự đoán các loại tấn công mạng.

# %% [markdown]
# ## 1. Import các thư viện cần thiết

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Thiết lập style cho plots
plt.style.use('default')
sns.set_palette("husl")

print("Đã import thành công các thư viện!")

# %% [markdown]
# ## 2. Tải và khám phá dữ liệu

# %%
# Định nghĩa tên cột cho NSL-KDD dataset
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
print(f"Features: {column_names[:10]}...")  # Hiển thị 10 features đầu

# %%
# Tải dữ liệu training (sử dụng 20% subset để xử lý nhanh hơn)
train_data = pd.read_csv('data/KDDTrain+_20Percent.txt', names=column_names, header=None)
test_data = pd.read_csv('data/KDDTest+.txt', names=column_names, header=None)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")
print("\nFirst 5 rows of training data:")
print(train_data.head())

# %%
# Thông tin cơ bản về dataset
print("=== THÔNG TIN CƠ BẢN VỀ DATASET ===")
print(f"Training set: {train_data.shape[0]} samples, {train_data.shape[1]} features")
print(f"Test set: {test_data.shape[0]} samples, {test_data.shape[1]} features")

print("\n=== KIỂU DỮ LIỆU ===")
print(train_data.dtypes.value_counts())

print("\n=== MISSING VALUES ===")
print(f"Training set missing values: {train_data.isnull().sum().sum()}")
print(f"Test set missing values: {test_data.isnull().sum().sum()}")

# %%
# Phân tích phân phối các lớp (classes)
print("=== PHÂN PHỐI CÁC LỚP TRONG TRAINING SET ===")
class_counts = train_data['class'].value_counts()
print(class_counts)

print("\n=== PHÂN PHỐI CÁC LỚP TRONG TEST SET ===")
test_class_counts = test_data['class'].value_counts()
print(test_class_counts)

# Visualize class distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Training set
class_counts.plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_title('Phân phối các lớp - Training Set')
ax1.set_xlabel('Loại tấn công')
ax1.set_ylabel('Số lượng')
ax1.tick_params(axis='x', rotation=45)

# Test set
test_class_counts.plot(kind='bar', ax=ax2, color='lightcoral')
ax2.set_title('Phân phối các lớp - Test Set')
ax2.set_xlabel('Loại tấn công')
ax2.set_ylabel('Số lượng')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Tiền xử lý dữ liệu cho Linear Regression

# %%
# Tách categorical và numerical features
categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = [col for col in column_names if col not in categorical_features + ['class', 'difficulty']]

print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
print(f"Numerical features ({len(numerical_features)}): {numerical_features[:10]}...")  # Hiển thị 10 đầu

# Kiểm tra unique values trong categorical features
for feature in categorical_features:
    print(f"\n{feature}: {train_data[feature].nunique()} unique values")
    print(f"Values: {train_data[feature].unique()[:10]}...")  # Hiển thị 10 đầu

# %%
# Chuyển đổi target variable cho regression
def prepare_target_for_regression(data, method='label_encoding'):
    """
    Chuẩn bị target variable cho Linear Regression
    method: 'label_encoding' hoặc 'attack_severity'
    """
    if method == 'label_encoding':
        # Encode tất cả classes thành số
        le = LabelEncoder()
        target = le.fit_transform(data['class'])
        return target, le
    
    elif method == 'attack_severity':
        # Gán điểm severity cho từng loại tấn công
        severity_mapping = {
            'normal': 0,
            # DoS attacks
            'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
            'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
            # Probe attacks  
            'satan': 2, 'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'mscan': 2, 'saint': 2,
            # R2L attacks
            'guess_passwd': 3, 'ftp_write': 3, 'imap': 3, 'phf': 3, 'multihop': 3,
            'warezmaster': 3, 'warezclient': 3, 'spy': 3, 'xlock': 3, 'xsnoop': 3,
            'snmpguess': 3, 'snmpgetattack': 3, 'httptunnel': 3, 'sendmail': 3, 'named': 3,
            # U2R attacks
            'buffer_overflow': 4, 'loadmodule': 4, 'rootkit': 4, 'perl': 4, 'sqlattack': 4,
            'xterm': 4, 'ps': 4
        }
        
        target = data['class'].map(severity_mapping)
        # Xử lý các class không có trong mapping
        target = target.fillna(2)  # Gán mức độ trung bình cho unknown attacks
        return target.values, severity_mapping

# Sử dụng label encoding
y_train, label_encoder = prepare_target_for_regression(train_data, 'label_encoding')
y_test, _ = prepare_target_for_regression(test_data, 'label_encoding')

print(f"Target shape - Train: {y_train.shape}, Test: {y_test.shape}")
print(f"Target range - Train: [{y_train.min()}, {y_train.max()}], Test: [{y_test.min()}, {y_test.max()}]")
print(f"\nClass mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")

# %%
# Xử lý categorical features bằng One-Hot Encoding
def preprocess_features(train_df, test_df, categorical_features, numerical_features):
    """
    Tiền xử lý features cho Linear Regression
    """
    # Copy data
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # One-hot encoding cho categorical features
    train_categorical = train_processed[categorical_features]
    test_categorical = test_processed[categorical_features]
    
    # Combine train và test để đảm bảo consistent encoding
    combined_categorical = pd.concat([train_categorical, test_categorical], ignore_index=True)
    
    # One-hot encoding
    encoded_categorical = pd.get_dummies(combined_categorical, prefix=categorical_features)
    
    # Split back
    train_encoded = encoded_categorical[:len(train_processed)]
    test_encoded = encoded_categorical[len(train_processed):]
    
    # Numerical features
    train_numerical = train_processed[numerical_features]
    test_numerical = test_processed[numerical_features]
    
    # Combine features
    X_train = pd.concat([train_numerical, train_encoded], axis=1)
    X_test = pd.concat([test_numerical, test_encoded], axis=1)
    
    return X_train, X_test

# Preprocess features
X_train, X_test = preprocess_features(train_data, test_data, categorical_features, numerical_features)

print(f"Processed features shape - Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Feature names: {list(X_train.columns)[:10]}...")  # Hiển thị 10 features đầu

# %%
# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled features shape - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
print(f"Feature statistics after scaling:")
print(f"Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")

# %% [markdown]
# ## 4. Xây dựng và huấn luyện các mô hình Linear Regression

# %%
# Khởi tạo các mô hình
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0)
}

# Dictionary để lưu kết quả
results = {}

print("=== HUẤN LUYỆN CÁC MÔ HÌNH ===")
for name, model in models.items():
    print(f"\nHuấn luyện {name}...")

    # Huấn luyện mô hình
    model.fit(X_train_scaled, y_train)

    # Dự đoán
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Tính toán metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Lưu kết quả
    results[name] = {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

# %% [markdown]
# ## 5. Đánh giá và so sánh các mô hình

# %%
# Tạo bảng so sánh kết quả
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train MSE': [results[model]['train_mse'] for model in results.keys()],
    'Test MSE': [results[model]['test_mse'] for model in results.keys()],
    'Train R²': [results[model]['train_r2'] for model in results.keys()],
    'Test R²': [results[model]['test_r2'] for model in results.keys()],
    'Train MAE': [results[model]['train_mae'] for model in results.keys()],
    'Test MAE': [results[model]['test_mae'] for model in results.keys()]
})

print("=== BẢNG SO SÁNH KẾT QUẢ CÁC MÔ HÌNH ===")
print(comparison_df.round(4))

# %%
# Visualize kết quả so sánh
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# MSE comparison
axes[0, 0].bar(comparison_df['Model'], comparison_df['Test MSE'], color=['skyblue', 'lightgreen', 'lightcoral'])
axes[0, 0].set_title('Test MSE Comparison')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].tick_params(axis='x', rotation=45)

# R² comparison
axes[0, 1].bar(comparison_df['Model'], comparison_df['Test R²'], color=['skyblue', 'lightgreen', 'lightcoral'])
axes[0, 1].set_title('Test R² Comparison')
axes[0, 1].set_ylabel('R²')
axes[0, 1].tick_params(axis='x', rotation=45)

# MAE comparison
axes[1, 0].bar(comparison_df['Model'], comparison_df['Test MAE'], color=['skyblue', 'lightgreen', 'lightcoral'])
axes[1, 0].set_title('Test MAE Comparison')
axes[1, 0].set_ylabel('MAE')
axes[1, 0].tick_params(axis='x', rotation=45)

# Train vs Test R² comparison
x = np.arange(len(comparison_df))
width = 0.35
axes[1, 1].bar(x - width/2, comparison_df['Train R²'], width, label='Train R²', color='lightblue')
axes[1, 1].bar(x + width/2, comparison_df['Test R²'], width, label='Test R²', color='orange')
axes[1, 1].set_title('Train vs Test R² Comparison')
axes[1, 1].set_ylabel('R²')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(comparison_df['Model'], rotation=45)
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# %%
# Chọn mô hình tốt nhất dựa trên Test R²
best_model_name = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
best_model = results[best_model_name]['model']

print(f"=== MÔ HÌNH TỐT NHẤT: {best_model_name} ===")
print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"Test MSE: {results[best_model_name]['test_mse']:.4f}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.4f}")

# %% [markdown]
# ## 6. Phân tích chi tiết mô hình tốt nhất

# %%
# Scatter plot: Actual vs Predicted
best_y_test_pred = results[best_model_name]['y_test_pred']

plt.figure(figsize=(12, 5))

# Test set predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, best_y_test_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'{best_model_name} - Test Set\nActual vs Predicted')
plt.grid(True, alpha=0.3)

# Residuals plot
plt.subplot(1, 2, 2)
residuals = y_test - best_y_test_pred
plt.scatter(best_y_test_pred, residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'{best_model_name} - Residuals Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"=== PHÂN TÍCH CHI TIẾT {best_model_name} ===")
print(f"Residuals statistics:")
print(f"  Mean: {residuals.mean():.6f}")
print(f"  Std: {residuals.std():.4f}")
print(f"  Min: {residuals.min():.4f}")
print(f"  Max: {residuals.max():.4f}")

# %%
# Phân tích feature importance (cho Ridge và Lasso)
if best_model_name in ['Ridge Regression', 'Lasso Regression']:
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': best_model.coef_
    })
    feature_importance['abs_coefficient'] = np.abs(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

    print(f"\n=== TOP 20 FEATURES QUAN TRỌNG NHẤT ({best_model_name}) ===")
    print(feature_importance.head(20))

    # Visualize top features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['coefficient'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value')
    plt.title(f'Top 15 Feature Coefficients - {best_model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 7. Thử nghiệm với Attack Severity Mapping

# %%
# Thử nghiệm với attack severity mapping
print("=== THỬ NGHIỆM VỚI ATTACK SEVERITY MAPPING ===")

# Chuẩn bị target với severity mapping
y_train_severity, severity_mapping = prepare_target_for_regression(train_data, 'attack_severity')
y_test_severity, _ = prepare_target_for_regression(test_data, 'attack_severity')

print(f"Severity mapping: {severity_mapping}")
print(f"Target range - Train: [{y_train_severity.min()}, {y_train_severity.max()}], Test: [{y_test_severity.min()}, {y_test_severity.max()}]")

# Huấn luyện mô hình tốt nhất với severity target
severity_model = type(best_model)(**best_model.get_params())
severity_model.fit(X_train_scaled, y_train_severity)

# Dự đoán
y_test_severity_pred = severity_model.predict(X_test_scaled)

# Tính toán metrics
severity_mse = mean_squared_error(y_test_severity, y_test_severity_pred)
severity_r2 = r2_score(y_test_severity, y_test_severity_pred)
severity_mae = mean_absolute_error(y_test_severity, y_test_severity_pred)

print(f"\n=== KẾT QUẢ VỚI SEVERITY MAPPING ===")
print(f"Test MSE: {severity_mse:.4f}")
print(f"Test R²: {severity_r2:.4f}")
print(f"Test MAE: {severity_mae:.4f}")

# Visualize severity predictions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test_severity, y_test_severity_pred, alpha=0.5, color='purple')
plt.plot([y_test_severity.min(), y_test_severity.max()], [y_test_severity.min(), y_test_severity.max()], 'r--', lw=2)
plt.xlabel('Actual Severity')
plt.ylabel('Predicted Severity')
plt.title('Severity Mapping - Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
severity_residuals = y_test_severity - y_test_severity_pred
plt.scatter(y_test_severity_pred, severity_residuals, alpha=0.5, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Severity')
plt.ylabel('Residuals')
plt.title('Severity Mapping - Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Kết luận và Đánh giá

# %%
print("=== KẾT LUẬN VÀ ĐÁNH GIÁ ===")
print("\n1. HIỆU SUẤT CÁC MÔ HÌNH:")
for name in results.keys():
    print(f"   {name}:")
    print(f"     - Test R²: {results[name]['test_r2']:.4f}")
    print(f"     - Test MSE: {results[name]['test_mse']:.4f}")
    print(f"     - Test MAE: {results[name]['test_mae']:.4f}")

print(f"\n2. MÔ HÌNH TỐT NHẤT: {best_model_name}")
print(f"   - Đạt Test R² = {results[best_model_name]['test_r2']:.4f}")
print(f"   - Test MSE = {results[best_model_name]['test_mse']:.4f}")

print(f"\n3. SO SÁNH PHƯƠNG PHÁP TARGET:")
print(f"   - Label Encoding: R² = {results[best_model_name]['test_r2']:.4f}")
print(f"   - Severity Mapping: R² = {severity_r2:.4f}")

print("\n4. NHẬN XÉT:")
print("   - Linear Regression có thể áp dụng cho NSL-KDD dataset")
print("   - Regularization (Ridge/Lasso) có thể cải thiện hiệu suất")
print("   - Severity mapping có thể phù hợp hơn cho bài toán thực tế")
print("   - Cần xem xét các phương pháp khác như Classification để so sánh")

print("\n=== HOÀN THÀNH PHÂN TÍCH NSL-KDD VỚI LINEAR REGRESSION ===")

# %%
# Lưu kết quả
results_summary = {
    'best_model': best_model_name,
    'best_test_r2': results[best_model_name]['test_r2'],
    'best_test_mse': results[best_model_name]['test_mse'],
    'severity_r2': severity_r2,
    'severity_mse': severity_mse,
    'feature_count': X_train.shape[1],
    'train_samples': X_train.shape[0],
    'test_samples': X_test.shape[0]
}

print(f"\nKết quả đã được lưu: {results_summary}")
