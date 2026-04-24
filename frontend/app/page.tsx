"use client";
import { useState, useEffect } from "react";

export default function PneumoniaDetector() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null); // State lưu link ảnh gốc
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      // Tạo đường dẫn tạm thời trên trình duyệt để hiển thị ảnh gốc ngay lập tức
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setResult(null); // Xóa kết quả cũ khi chọn ảnh mới
    }
  };

  // Dọn dẹp bộ nhớ trình duyệt khi component bị hủy hoặc người dùng chọn ảnh khác
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Lỗi khi gọi API:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto font-sans">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">
        Hệ Thống Chuẩn Đoán Viêm Phổi X-Quang
      </h1>
      
      <div className="mb-6 bg-white p-6 rounded-lg shadow-sm border">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange} 
          className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 mb-4" 
        />
        
        <button 
          onClick={handlePredict} 
          disabled={!file || loading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2 rounded transition-colors disabled:bg-gray-400"
        >
          {loading ? "Đang phân tích..." : "Chuẩn Đoán"}
        </button>
      </div>

      {result && !result.error && (
        <div className="mt-6 bg-white border p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-bold mb-4 border-b pb-2">Kết quả phân tích:</h2>
          
          <div className="mb-4 text-lg">
            <span className="text-gray-600">Nhãn chuẩn đoán: </span>
            <strong className={result.prediction.includes("Viêm") ? "text-red-600" : "text-green-600"}>
              {result.prediction}
            </strong>
          </div>
          
          <div className="mb-6 text-lg">
            <span className="text-gray-600">Độ tin cậy: </span>
            <span className="font-medium">{result.confidence}</span>
          </div>
          
          {/* Cấu trúc Grid chia 2 cột để so sánh ảnh */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            {/* Cột 1: Ảnh gốc */}
            <div>
              <h3 className="text-md font-semibold mb-3 text-gray-700 text-center">
                Ảnh X-quang Gốc
              </h3>
              {previewUrl && (
                <img 
                  src={previewUrl} 
                  alt="Original X-ray" 
                  className="w-full h-auto object-contain border-2 border-gray-200 rounded shadow-sm bg-black"
                />
              )}
            </div>

            {/* Cột 2: Ảnh Grad-CAM */}
            <div>
              <h3 className="text-md font-semibold mb-3 text-gray-700 text-center">
                Bản đồ nhiệt Grad-CAM
              </h3>
              <img 
                src={`data:image/jpeg;base64,${result.gradcam_image}`} 
                alt="Grad-CAM Heatmap" 
                className="w-full h-auto object-contain border-2 border-gray-200 rounded shadow-sm bg-black"
              />
            </div>
          </div>

        </div>
      )}

      {result?.error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 text-red-600 rounded">
          <strong>Đã xảy ra lỗi:</strong> {result.error}
        </div>
      )}
    </div>
  );
}