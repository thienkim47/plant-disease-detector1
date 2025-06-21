# Sử dụng image python chính thức
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy toàn bộ code vào container
COPY . /app

# Cài đặt các thư viện Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8080 (Cloud Run mặc định dùng 8080)
EXPOSE 8080

# Biến môi trường cho Flask
ENV PORT=8080
ENV FLASK_ENV=production

# Chạy app với gunicorn
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app
