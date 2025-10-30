# 🚀 LỆNH CHẠY SO SÁNH FedAvg vs FedLWS

## 📋 Cú pháp cơ bản

```bash
python compare_algorithms.py --dataset <dataset> --local_model <model> --T <rounds> --node_num <clients> --device <device>
```

---

## ⚡ Các lệnh thường dùng

### Test nhanh (10-15 phút)
```bash
python compare_algorithms.py --dataset cifar10 --local_model CNN --T 10 --node_num 5 --device cpu
```

### Cân bằng tốc độ/chất lượng (30-40 phút)
```bash
python compare_algorithms.py --dataset cifar10 --local_model CNN --T 20 --node_num 10 --device cpu
```

### Experiment đầy đủ (2-3 giờ)
```bash
python compare_algorithms.py --dataset cifar10 --local_model ResNet20 --T 50 --node_num 10 --device cpu
```

---

## 🎛️ Tham số chính

| Tham số | Mô tả | Giá trị | Ảnh hưởng tốc độ |
|---------|-------|---------|------------------|
| `--dataset` | Dataset | `cifar10`, `cifar100` | cifar10 nhanh hơn |
| `--local_model` | Model | `CNN`, `ResNet20` | CNN nhanh hơn ~2x |
| `--T` | Số rounds | 10, 20, 50, 100 | Tuyến tính |
| `--node_num` | Số clients | 5, 10, 20 | Tuyến tính |
| `--device` | Device | `cpu`, `0` | GPU nhanh hơn 5-10x |

**Lưu ý:** Fashion-MNIST không tương thích với CNN (cần 1 channel, CNN hiện tại dùng 3 channels)

---

## 📊 Bảng thời gian (CPU)

| Dataset | Model | Rounds | Clients | Thời gian |
|---------|-------|--------|---------|-----------|
| cifar10 | CNN | 10 | 5 | 10-15 phút |
| cifar10 | CNN | 20 | 10 | 30-40 phút |
| cifar10 | ResNet20 | 10 | 5 | 25-35 phút |
| cifar10 | ResNet20 | 20 | 10 | 1-1.5 giờ |
| cifar10 | ResNet20 | 50 | 10 | 2-3 giờ |

---

## 🎯 Khuyến nghị theo mục đích

### Demo nhanh (10-15 phút)
```bash
python compare_algorithms.py --dataset cifar10 --local_model CNN --T 10 --node_num 5 --device cpu
```

### Test thuật toán (30-40 phút)
```bash
python compare_algorithms.py --dataset cifar10 --local_model CNN --T 20 --node_num 10 --device cpu
```

### Paper results (với GPU)
```bash
python compare_algorithms.py --dataset cifar10 --local_model ResNet20 --T 100 --node_num 20 --device 0
```

---

## 💾 Kết quả

Tất cả lưu trong `comparison_results/`:
- `comparison_*.png` - Biểu đồ
- `comparison_*.csv` - Chi tiết
- `summary_*.csv` - Tóm tắt

---

## 🔧 Tùy chỉnh thêm

```bash
# Thay đổi learning rate
python compare_algorithms.py --lr 0.05

# Thay đổi batch size (lớn hơn = nhanh hơn)
python compare_algorithms.py --batchsize 256

# Non-IID mạnh hơn
python compare_algorithms.py --dirichlet_alpha 0.1

# Điều chỉnh FedLWS beta
python compare_algorithms.py --beta 0.05

# Lưu vào thư mục khác
python compare_algorithms.py --save_dir my_results
```

---

## ✅ Cài đặt (nếu chưa có)

```bash
pip install -r requirements.txt
```

Hoặc:
```bash
pip install torch torchvision numpy pandas matplotlib einops
```

