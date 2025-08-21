# %% [markdown]
# # Khai thác NSL-KDD Dataset bằng Linear Regression (Simplified Version)

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Đã import thành công các thư viện!")

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

# %%
# Tải dữ liệu
print("Đang tải dữ liệu...")
train_data = pd.read_csv('data/KDDTrain+_20Percent.txt', names=column_names, header=None)
test_data = pd.read_csv('data/KDDTest+.txt', names=column_names, header=None)

print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# %%
# Thông tin cơ bản về dataset
print("=== THÔNG TIN CƠ BẢN VỀ DATASET ===")
print(f"Training set: {train_data.shape[0]} samples, {train_data.shape[1]} features")
print(f"Test set: {test_data.shape[0]} samples, {test_data.shape[1]} features")

print("\n=== PHÂN PHỐI CÁC LỚP TRONG TRAINING SET ===")
class_counts = train_data['class'].value_counts()
print(class_counts.head(10))

print("\n=== PHÂN PHỐI CÁC LỚP TRONG TEST SET ===")
test_class_counts = test_data['class'].value_counts()
print(test_class_counts.head(10))

# %%
# Tiền xử lý dữ liệu
print("=== TIỀN XỬ LÝ DỮ LIỆU ===")

# Tách categorical và numerical features
categorical_features = ['protocol_type', 'service', 'flag']
numerical_features = [col for col in column_names if col not in categorical_features + ['class', 'difficulty']]

print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")

# Chuyển đổi target variable
le = LabelEncoder()
y_train = le.fit_transform(train_data['class'])
y_test = le.transform(test_data['class'])

print(f"Target classes: {len(le.classes_)}")
print(f"Target range - Train: [{y_train.min()}, {y_train.max()}]")

# %%
# One-hot encoding cho categorical features
print("Đang thực hiện One-hot encoding...")

# Combine train và test để đảm bảo consistent encoding
train_categorical = train_data[categorical_features]
test_categorical = test_data[categorical_features]
combined_categorical = pd.concat([train_categorical, test_categorical], ignore_index=True)

# One-hot encoding
encoded_categorical = pd.get_dummies(combined_categorical, prefix=categorical_features)

# Split back
train_encoded = encoded_categorical[:len(train_data)]
test_encoded = encoded_categorical[len(train_data):]

# Numerical features
train_numerical = train_data[numerical_features]
test_numerical = test_data[numerical_features]

# Combine features
X_train = pd.concat([train_numerical, train_encoded], axis=1)
X_test = pd.concat([test_numerical, test_encoded], axis=1)

print(f"Final feature shape - Train: {X_train.shape}, Test: {X_test.shape}")

# %%
# Chuẩn hóa dữ liệu
print("Đang chuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled features shape - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# %%
# Huấn luyện các mô hình
print("=== HUẤN LUYỆN CÁC MÔ HÌNH ===")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0)
}

results = {}

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
        'y_test_pred': y_test_pred
    }
    
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

# %%
# So sánh kết quả
print("\n=== BẢNG SO SÁNH KẾT QUẢ ===")
comparison_data = []
for name in results.keys():
    comparison_data.append([
        name,
        results[name]['train_mse'],
        results[name]['test_mse'],
        results[name]['train_r2'],
        results[name]['test_r2'],
        results[name]['train_mae'],
        results[name]['test_mae']
    ])

comparison_df = pd.DataFrame(comparison_data, 
                           columns=['Model', 'Train MSE', 'Test MSE', 'Train R²', 'Test R²', 'Train MAE', 'Test MAE'])
print(comparison_df.round(4))

# %%
# Chọn mô hình tốt nhất
best_idx = comparison_df['Test R²'].idxmax()
best_model_name = comparison_df.loc[best_idx, 'Model']
best_model = results[best_model_name]['model']

print(f"\n=== MÔ HÌNH TỐT NHẤT: {best_model_name} ===")
print(f"Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"Test MSE: {results[best_model_name]['test_mse']:.4f}")
print(f"Test MAE: {results[best_model_name]['test_mae']:.4f}")

# %%
# Visualize kết quả
print("Đang tạo biểu đồ...")

# So sánh R² scores
plt.figure(figsize=(12, 8))

# Subplot 1: R² comparison
plt.subplot(2, 2, 1)
models_list = list(results.keys())
test_r2_scores = [results[model]['test_r2'] for model in models_list]
plt.bar(models_list, test_r2_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Test R² Score Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45)

# Subplot 2: MSE comparison
plt.subplot(2, 2, 2)
test_mse_scores = [results[model]['test_mse'] for model in models_list]
plt.bar(models_list, test_mse_scores, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Test MSE Comparison')
plt.ylabel('MSE')
plt.xticks(rotation=45)

# Subplot 3: Actual vs Predicted (best model)
plt.subplot(2, 2, 3)
best_y_test_pred = results[best_model_name]['y_test_pred']
plt.scatter(y_test, best_y_test_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'{best_model_name} - Actual vs Predicted')
plt.grid(True, alpha=0.3)

# Subplot 4: Residuals plot
plt.subplot(2, 2, 4)
residuals = y_test - best_y_test_pred
plt.scatter(best_y_test_pred, residuals, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'{best_model_name} - Residuals Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Kết luận
print("\n=== KẾT LUẬN ===")
print(f"1. Mô hình tốt nhất: {best_model_name}")
print(f"2. Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"3. Test MSE: {results[best_model_name]['test_mse']:.4f}")
print(f"4. Số features sau xử lý: {X_train.shape[1]}")
print(f"5. Số classes: {len(le.classes_)}")

print("\n=== NHẬN XÉT ===")
if results[best_model_name]['test_r2'] > 0.8:
    print("- Mô hình có hiệu suất rất tốt (R² > 0.8)")
elif results[best_model_name]['test_r2'] > 0.6:
    print("- Mô hình có hiệu suất khá tốt (R² > 0.6)")
else:
    print("- Mô hình có hiệu suất trung bình, cần cải thiện")

print("- Linear Regression có thể áp dụng cho bài toán NSL-KDD")
print("- Regularization có thể giúp cải thiện hiệu suất")
print("- Cần xem xét các phương pháp khác như Classification")

print("\n=== HOÀN THÀNH PHÂN TÍCH ===")
