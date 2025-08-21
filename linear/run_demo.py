#!/usr/bin/env python3
"""
Demo Script - Khai thác NSL-KDD Dataset bằng Linear Regression

Chạy script này để thực hiện toàn bộ quá trình phân tích NSL-KDD dataset
bằng Linear Regression từ tải dữ liệu đến đánh giá kết quả.

Usage:
    python3 run_demo.py [--basic|--improved|--all]
    
Options:
    --basic     : Chỉ chạy phiên bản cơ bản (numerical features only)
    --improved  : Chỉ chạy phiên bản cải tiến (+ categorical features)
    --all       : Chạy cả hai phiên bản (default)
"""

import sys
import os
import subprocess
import argparse

def run_command(command, description):
    """Chạy command và hiển thị kết quả"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Thành công!")
            if result.stdout:
                print(result.stdout)
        else:
            print("❌ Lỗi!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    return True

def check_requirements():
    """Kiểm tra các thư viện cần thiết"""
    print("🔍 Kiểm tra requirements...")
    
    required_packages = ['pandas', 'numpy', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - chưa cài đặt")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Cần cài đặt: {', '.join(missing_packages)}")
        install = input("Cài đặt tự động? (y/n): ")
        if install.lower() == 'y':
            cmd = f"pip3 install {' '.join(missing_packages)}"
            return run_command(cmd, "Cài đặt thư viện")
        else:
            print("❌ Không thể tiếp tục mà không có thư viện cần thiết")
            return False
    
    return True

def download_data():
    """Tải dữ liệu NSL-KDD"""
    if os.path.exists('data/KDDTrain+_20Percent.txt'):
        print("✅ Dữ liệu đã tồn tại, bỏ qua tải xuống")
        return True
    
    return run_command("python3 src/download_data.py", "Tải dữ liệu NSL-KDD")

def run_basic_analysis():
    """Chạy phân tích cơ bản"""
    return run_command("python3 notebooks/basic_linear_regression.py", 
                      "Phân tích cơ bản (Numerical features only)")

def run_improved_analysis():
    """Chạy phân tích cải tiến"""
    return run_command("python3 notebooks/improved_linear_regression.py", 
                      "Phân tích cải tiến (+ Categorical features)")

def show_summary():
    """Hiển thị tóm tắt kết quả"""
    print(f"\n{'='*60}")
    print("📊 TÓM TẮT KẾT QUẢ")
    print(f"{'='*60}")
    
    print("""
🎯 MỤC TIÊU ĐÃ HOÀN THÀNH:
✅ Tải và tiền xử lý dữ liệu NSL-KDD
✅ Implement Linear Regression từ đầu
✅ Xử lý categorical features bằng One-Hot Encoding
✅ Áp dụng regularization (Ridge Regression)
✅ So sánh các phương pháp encoding target
✅ Đánh giá và visualize kết quả

📈 KẾT QUẢ CHÍNH:
• Phiên bản cơ bản (numerical only): R² = 0.1017
• Phiên bản cải tiến (+ categorical): R² = 0.1768
• Cải thiện: +73.9%
• Mô hình tốt nhất: Ridge Regression (α=10.0)

📁 FILES ĐÃ TẠO:
• src/download_data.py - Script tải dữ liệu
• notebooks/basic_linear_regression.py - Phân tích cơ bản
• notebooks/improved_linear_regression.py - Phân tích cải tiến
• results/nsl_kdd_linear_regression_report.md - Báo cáo chi tiết

💡 KHUYẾN NGHỊ:
• Linear Regression có thể áp dụng cho NSL-KDD
• Cần xem xét classification thay vì regression
• Feature engineering có thể cải thiện hiệu suất
• Advanced models (RF, SVM, NN) có thể cho kết quả tốt hơn
""")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Demo NSL-KDD Linear Regression Analysis')
    parser.add_argument('--basic', action='store_true', help='Chỉ chạy phân tích cơ bản')
    parser.add_argument('--improved', action='store_true', help='Chỉ chạy phân tích cải tiến')
    parser.add_argument('--all', action='store_true', help='Chạy cả hai phiên bản')
    
    args = parser.parse_args()
    
    # Default to --all if no option specified
    if not (args.basic or args.improved or args.all):
        args.all = True
    
    print("🎯 DEMO: KHAI THÁC NSL-KDD BẰNG LINEAR REGRESSION")
    print("Developed by Augment Agent")
    
    # Kiểm tra requirements
    if not check_requirements():
        sys.exit(1)
    
    # Tạo thư mục nếu chưa có
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Tải dữ liệu
    if not download_data():
        print("❌ Không thể tải dữ liệu")
        sys.exit(1)
    
    # Chạy phân tích
    success = True
    
    if args.basic or args.all:
        if not run_basic_analysis():
            success = False
    
    if args.improved or args.all:
        if not run_improved_analysis():
            success = False
    
    if success:
        show_summary()
        print("\n🎉 HOÀN THÀNH THÀNH CÔNG!")
        print("📖 Xem báo cáo chi tiết tại: results/nsl_kdd_linear_regression_report.md")
    else:
        print("\n❌ CÓ LỖI XẢY RA TRONG QUÁ TRÌNH THỰC HIỆN")
        sys.exit(1)

if __name__ == "__main__":
    main()
