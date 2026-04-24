====================================================================
      HƯỚNG DẪN CÀI ĐẶT VÀ KHỞI ĐỘNG HỆ THỐNG CHẨN ĐOÁN VIÊM PHỔI
====================================================================

Dự án bao gồm 2 phần chính: 
1. Backend (Python/FastAPI) - Chứa mô hình AI xử lý ảnh.
2. Frontend (Next.js/React) - Giao diện người dùng.

--------------------------------------------------------------------
YÊU CẦU HỆ THỐNG (PREREQUISITES)
--------------------------------------------------------------------
Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt:
1. Python 3.11 hoặc 3.12 (Lưu ý: Không dùng bản 3.13 hoặc 3.14 vì chưa hỗ trợ TensorFlow).
   - Tải tại: https://www.python.org/downloads/
   - QUAN TRỌNG: Nhớ tích chọn "Add Python to PATH" khi cài đặt.
2. Node.js (Bản LTS - Long Term Support).
   - Tải tại: https://nodejs.org/

--------------------------------------------------------------------
PHẦN 1: CÀI ĐẶT VÀ KHỞI ĐỘNG BACKEND (AI MODEL)
--------------------------------------------------------------------
1. Mở terminal (PowerShell hoặc VS Code) và di chuyển vào thư mục backend.
2. (Chỉ dành cho máy Windows lần đầu chạy môi trường ảo) Mở khóa quyền chạy script bằng lệnh:
   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
3. Tạo môi trường ảo (Virtual Environment):
   python -m venv venv
   (Nếu máy có nhiều bản Python, dùng: py -3.12 -m venv venv)
4. Kích hoạt môi trường ảo:
   .\venv\Scripts\activate
   (Thành công khi thấy chữ (venv) hiện ở đầu dòng lệnh).
5. Cài đặt các thư viện cần thiết:
   python -m pip install --upgrade pip
   pip install fastapi uvicorn tensorflow pillow python-multipart
6. Bổ sung file mô hình:
   Đảm bảo file mô hình AI (ví dụ: densenet121_pneumonia.h5) đã được đặt cùng thư mục với file main.py.
7. Khởi động server Backend:

.\venv\Scripts\activate
cd Desktop
cd backend
   uvicorn main:app --reload

=> Backend sẽ chạy tại địa chỉ: http://localhost:8000
Lưu ý: KHÔNG tắt cửa sổ terminal này trong suốt quá trình sử dụng web.

--------------------------------------------------------------------
PHẦN 2: CÀI ĐẶT VÀ KHỞI ĐỘNG FRONTEND (GIAO DIỆN WEB)
--------------------------------------------------------------------
1. Mở một cửa sổ terminal MỚI (giữ nguyên terminal của backend đang chạy) và di chuyển vào thư mục frontend.
2. Cài đặt các gói thư viện (chỉ cần chạy lần đầu hoặc khi mới tải code về):
   npm install
3. Khởi động giao diện web:
cd frontend
   npm run dev

=> Frontend sẽ chạy tại địa chỉ: http://localhost:3000

--------------------------------------------------------------------
HƯỚNG DẪN SỬ DỤNG
--------------------------------------------------------------------
1. Mở trình duyệt web (Chrome, Edge, Safari...) và truy cập: http://localhost:3000
2. Nhấn nút "Chọn tệp" để tải lên một bức ảnh X-quang phổi.
3. Nhấn "Chẩn đoán hình ảnh" và chờ hệ thống trả về kết quả (Bình thường hoặc Viêm phổi) cùng độ tin cậy.
