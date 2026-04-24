import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
import base64

app = FastAPI()

# Cấu hình CORS cho Next.js (Giữ nguyên)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tải mô hình đã huấn luyện (Bản không CLAHE, Dropout 0.4)
MODEL_PATH = 'best_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================
# 1. TỰ ĐỘNG KHOANH VÙNG LỒNG NGỰC (MASKING)
# ==========================================
def auto_lung_mask(img_batch):
    # Lấy ảnh từ batch và đưa về định dạng OpenCV 0-255
    img = (img_batch[0] * 255).astype('uint8')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Tách ngưỡng (Phổi thường là vùng tối nhất trong lồng ngực)
    _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    
    # Tạo vùng cắt an toàn (Loại bỏ vai và bụng dưới)
    spatial_mask = np.zeros((224, 224), dtype="uint8")
    cv2.rectangle(spatial_mask, (20, 20), (204, 190), 255, -1) 
    
    combined = cv2.bitwise_and(thresh, spatial_mask)
    
    # Tìm mảng khối lớn nhất (Chính là diện tích phổi)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros((224, 224), dtype="uint8")
    
    for c in contours:
        if cv2.contourArea(c) > 500:
            cv2.drawContours(final_mask, [c], -1, 255, -1)
            
    # Dự phòng: Nếu ảnh quá mờ, sử dụng Elip định vị vùng phổi trung tâm
    if np.sum(final_mask) == 0:
        cv2.ellipse(final_mask, (85, 115), (45, 80), 0, 0, 360, 255, -1)
        cv2.ellipse(final_mask, (139, 115), (45, 80), 0, 0, 360, 255, -1)
        
    mask_float = final_mask.astype("float32") / 255.0
    return cv2.GaussianBlur(mask_float, (21, 21), 0)

# ==========================================
# 2. TIỀN XỬ LÝ ẢNH ĐẦU VÀO (BỎ CLAHE)
# ==========================================
def preprocess_image(image_bytes, crop_ratio=0.08):
    original_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    width, height = original_pil.size
    
    # Center Cropping: Loại bỏ nhiễu chữ/viền đen ở rìa ảnh X-quang
    left = int(width * crop_ratio)
    top = int(height * crop_ratio)
    right = width - int(width * crop_ratio)
    bottom = height - int(height * crop_ratio)
    
    cropped_pil = original_pil.crop((left, top, right, bottom))
    original_size = (cropped_pil.width, cropped_pil.height)
    
    # Giữ bản OpenCV gốc để chồng Grad-CAM
    original_cv_img = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)
    
    # Resize về 224x224 cho DenseNet121
    img_resized = cropped_pil.resize((224, 224))
    img_array = np.array(img_resized).astype('float32')
    
    # CHUẨN HÓA (Rescale 1./255 - KHÔNG CÒN CLAHE Ở ĐÂY)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch, original_cv_img, original_size

# ==========================================
# 3. THUẬT TOÁN GRAD-CAM (TRỰC QUAN HÓA)
# ==========================================
def get_gradcam_base64(img_batch, original_cv_img, original_size, model):
    # Sử dụng lớp 'relu' cuối cùng của DenseNet121
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer('relu').output, model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_output, preds = grad_model(img_batch)
        class_channel = preds[0][0] 

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_output = last_conv_output[0]
    
    heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    # Áp dụng mặt nạ phổi để xóa nhiễu Grad-CAM vùng ngoài lồng ngực
    heatmap_res = cv2.resize(heatmap, (224, 224))
    mask = auto_lung_mask(img_batch)
    masked_heatmap = heatmap_res * mask
    
    # Phục hồi kích thước và tạo bản đồ nhiệt màu
    masked_heatmap_original = cv2.resize(masked_heatmap, original_size)
    masked_heatmap_original = np.uint8(255 * masked_heatmap_original)
    heatmap_color = cv2.applyColorMap(masked_heatmap_original, cv2.COLORMAP_JET)

    # Chồng ảnh màu lên ảnh X-quang gốc (Tỉ lệ 40% nhiệt - 60% gốc)
    superimposed_img = cv2.addWeighted(heatmap_color, 0.4, original_cv_img, 0.6, 0)
    _, buffer = cv2.imencode('.jpg', superimposed_img)
    
    return base64.b64encode(buffer).decode('utf-8')

# ==========================================
# 4. API ENDPOINT
# ==========================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # 1. Tiền xử lý (Đã bỏ CLAHE)
        processed_batch, original_cv_img, original_size = preprocess_image(image_bytes)
        
        # 2. Dự đoán xác suất
        prediction = model.predict(processed_batch)
        prob = float(prediction[0][0])
        
        # 3. Phân loại và tính độ tin cậy hiển thị
        if prob >= 0.6:
            label = "Viêm phổi (Pneumonia)"
            display_conf = prob
        else:
            label = "Bình thường (Normal)"
            display_conf = 1.0 - prob
            
        # 4. Tạo hình ảnh chẩn đoán vùng bệnh
        gradcam_img = get_gradcam_base64(processed_batch, original_cv_img, original_size, model)

        return {
            "success": True,
            "filename": file.filename,
            "prediction": label,
            "confidence": f"{display_conf:.2%}",
            "gradcam_image": gradcam_img,
            "raw_probability": prob # Dùng để debug nếu cần
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)