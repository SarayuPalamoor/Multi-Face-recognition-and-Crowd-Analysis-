
import React, { useState, useRef, useEffect } from 'react';
import { Html5Qrcode } from 'html5-qrcode';
import axios from 'axios';

function ScanAndCapture() {
    const [qrData, setQrData] = useState(null);
    const [isScanning, setIsScanning] = useState(false);
    const [capturedImages, setCapturedImages] = useState([]);
    const [isCapturing, setIsCapturing] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadSuccess, setUploadSuccess] = useState(false);
    const [trainingStatus, setTrainingStatus] = useState("");
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    let html5QrCode = null;

    useEffect(() => {
        return () => {
            if (html5QrCode) {
                html5QrCode.stop().catch(err => console.error("Error stopping scanner during cleanup:", err));
            }
        };
    }, []);

    const startScanner = async () => {
        if (html5QrCode && isScanning) {
            await stopScanner();
        }

        try {
            html5QrCode = new Html5Qrcode("reader");
            setIsScanning(true);

            await html5QrCode.start(
                { facingMode: "environment" },
                { fps: 10, qrbox: { width: 250, height: 250 } },
                (decodedText) => {
                    const qrDetails = parseQrData(decodedText);
                    setQrData(qrDetails);
                    stopScanner();
                },
                (errorMessage) => {
                    console.warn('QR Code scan error:', errorMessage);
                }
            );

            const videoElement = document.querySelector('video');
            if (videoElement) {
                videoRef.current = videoElement;
            }
        } catch (error) {
            console.error('Error starting scanner:', error);
        }
    };

    const stopScanner = async () => {
        if (html5QrCode && isScanning) {
            try {
                await html5QrCode.stop();
                setIsScanning(false);
                html5QrCode = null;
            } catch (err) {
                console.error("Error stopping scanner:", err);
            }
        }
    };

    const captureImages = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        if (!video || !video.srcObject) {
            console.error('Video stream is not ready or not found.');
            return;
        }

        let images = [];
        let currentIndex = 0;

        const captureImage = () => {
            if (currentIndex >= 10) {
                setCapturedImages(images);
                sendImages(images);
                setIsCapturing(false);
                return;
            }

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob((blob) => {
                if (blob && !images.some(image => image.size === blob.size)) {
                    images.push(blob);
                }
                currentIndex++;
                setTimeout(captureImage, 2000);
            }, 'image/jpeg');
        };

        setIsCapturing(true);
        captureImage();
    };

    const sendImages = async (images) => {
        setIsUploading(true);
        setTrainingStatus(""); // Reset training status

        const formData = new FormData();
        images.forEach((image, index) => formData.append("images", image, `capture${index}.jpg`));
        formData.append('qrData', JSON.stringify(qrData));

        try {
            const response = await axios.post('http://localhost:4000/api/uploadImages', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });
            console.log("Images uploaded successfully:", response.data);
            setUploadSuccess(true);
        } catch (error) {
            console.error("Error uploading images:", error);
            setUploadSuccess(false);
        } finally {
            setIsUploading(false);
        }
    };

    const handleTrainUser = async (rollNo) => {
        try {
            const response = await axios.post('http://localhost:4000/api/trainUser', { rollNo });
            console.log('User-specific model trained successfully:', response.data);
            setTrainingStatus("Trained Successfully");
        } catch (error) {
            console.error('Error training user-specific model:', error);
            setTrainingStatus("Training Failed");
        }
    };

    const parseQrData = (qrText) => {
        const [rollNo, name, fatherName, department, contact] = qrText.split(',');
        return {
            rollNo: rollNo.trim(),
            name: name.trim(),
            fatherName: fatherName.trim(),
            department: department.trim(),
            contact: contact.trim(),
        };
    };
    return (
        <div style={{ backgroundColor: "#1a202c", color: "#f7fafc", minHeight: "100vh", padding: "20px", fontFamily: "Arial, sans-serif" }}>
          <div style={{ textAlign: "center", marginBottom: "20px" }}>
            <button
              onClick={startScanner}
              disabled={isCapturing || isUploading}
              style={{
                backgroundColor: "#3182ce",
                color: "#fff",
                padding: "10px 20px",
                border: "none",
                borderRadius: "5px",
                fontSize: "16px",
                cursor: "pointer",
                boxShadow: "0 4px 6px rgba(0, 0, 0, 0.2)",
              }}
            >
              Start Scanner
            </button>
          </div>
          <div id="reader" style={{ width: "400px", height: "400px", margin: "20px auto", border: "2px solid #2d3748", borderRadius: "10px", backgroundColor: "#2d3748" }}></div>
      
          {/* Render QR Data Section only if qrData exists and is not empty */}
          {qrData && Object.keys(qrData).length > 0 && (
            <div style={{ backgroundColor: "#2d3748", padding: "20px", borderRadius: "10px", margin: "20px auto", maxWidth: "500px" }}>
              <h3 style={{ color: "#63b3ed", marginBottom: "10px" }}>QR Data:</h3>
              <pre style={{ backgroundColor: "#1a202c", padding: "10px", borderRadius: "5px", overflowX: "auto" }}>
                {JSON.stringify(qrData, null, 2)}
              </pre>
            </div>
          )}
      
          {/* Video and Canvas for Image Capture */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: "20px" }}>
            {/* <video
              ref={videoRef}
              autoPlay
              muted
              width="640"
              height="480"
              style={{
                borderRadius: "10px",
                border: "2px solid #63b3ed",
                backgroundColor: "#2d3748",
                boxShadow: "0 4px 8px rgba(0, 0, 0, 0.4)",
              }}
            ></video> */}
            <canvas ref={canvasRef} width="640" height="480" style={{ display: "none" }}></canvas>
          </div>
      
          <div style={{ textAlign: "center", marginBottom: "20px" }}>
            <button
              onClick={captureImages}
              disabled={isCapturing || isUploading || !qrData}
              style={{
                backgroundColor: "#2b6cb0",
                color: "#fff",
                padding: "10px 20px",
                border: "none",
                borderRadius: "5px",
                fontSize: "16px",
                cursor: "pointer",
                marginRight: "10px",
              }}
            >
              Capture 10 Images
            </button>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => sendImages(Array.from(e.target.files))}
              disabled={isCapturing || isUploading}
              style={{
                color: "#fff",
                backgroundColor: "#2d3748",
                padding: "10px",
                border: "1px solid #2b6cb0",
                borderRadius: "5px",
              }}
            />
          </div>
      
          {/* Train User Model Button only if QR Data is available */}
          {qrData && (
            <div style={{ textAlign: "center", marginBottom: "20px" }}>
              <button
                onClick={() => handleTrainUser(qrData.rollNo)}
                disabled={isCapturing || isUploading}
                style={{
                  backgroundColor: "#2c5282",
                  color: "#fff",
                  padding: "10px 20px",
                  border: "none",
                  borderRadius: "5px",
                  fontSize: "16px",
                  cursor: "pointer",
                }}
              >
                Train User Model
              </button>
            </div>
          )}
      
          {trainingStatus && <p style={{ textAlign: "center", color: "#63b3ed" }}>{trainingStatus}</p>}
      
          {/* Render Captured Images Section only if images exist */}
          {capturedImages.length > 0 && (
            <div style={{ textAlign: "center" }}>
              <h3 style={{ color: "#63b3ed", marginBottom: "10px" }}>Captured Images:</h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "10px", justifyContent: "center" }}>
                {capturedImages.map((image, index) => (
                  <img
                    key={index}
                    src={URL.createObjectURL(image)}
                    alt={`Captured ${index + 1}`}
                    style={{
                      width: "150px",
                      height: "120px",
                      borderRadius: "10px",
                      border: "2px solid #63b3ed",
                      boxShadow: "0 4px 8px rgba(0, 0, 0, 0.4)",
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      );
      
}

export default ScanAndCapture;
