"""
Script để tải xuống bộ dữ liệu NSL-KDD
"""
import os
import requests
import pandas as pd
from pathlib import Path

def download_file(url, filename):
    """Tải file từ URL"""
    print(f"Đang tải {filename}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Đã tải xong {filename}")

def download_nsl_kdd():
    """Tải bộ dữ liệu NSL-KDD"""
    # Tạo thư mục data nếu chưa có
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # URLs cho NSL-KDD dataset
    urls = {
        "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
        "KDDTrain+_20Percent.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B_20Percent.txt",
        "KDDTest-21.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest-21.txt"
    }
    
    # Tải từng file
    for filename, url in urls.items():
        filepath = data_dir / filename
        if not filepath.exists():
            try:
                download_file(url, filepath)
            except Exception as e:
                print(f"Lỗi khi tải {filename}: {e}")
        else:
            print(f"{filename} đã tồn tại, bỏ qua...")

def create_column_names():
    """Tạo danh sách tên cột cho NSL-KDD"""
    columns = [
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
    return columns

if __name__ == "__main__":
    print("Bắt đầu tải dữ liệu NSL-KDD...")
    download_nsl_kdd()
    
    # Lưu tên cột
    columns = create_column_names()
    with open("data/column_names.txt", "w") as f:
        for col in columns:
            f.write(f"{col}\n")
    
    print("Hoàn thành tải dữ liệu!")
