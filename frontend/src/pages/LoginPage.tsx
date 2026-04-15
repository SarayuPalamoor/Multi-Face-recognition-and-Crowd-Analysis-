import React, { useState, useRef, useEffect } from 'react';
import { Html5Qrcode } from 'html5-qrcode';
import axios from 'axios';

interface QRData {
  rollNo: string;
  name: string;
  fatherName: string;
  department: string;
  contact: string;
}

const LoginPage = () => {
  const [qrData, setQrData] = useState<QRData | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [capturedImages, setCapturedImages] = useState<File[]>([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  let Html5QrCode: Html5Qrcode | null = null;

  useEffect(() => {
    return () => {
      if (Html5QrCode) {
        Html5QrCode.stop().catch(err => console.error("Error stopping scanner during cleanup:", err));
      }
    };
  }, []);

  const startScanner = async () => {
    if (Html5QrCode && isScanning) {
      await stopScanner();
    }

    try {
      Html5QrCode = new Html5Qrcode("reader");
      setIsScanning(true);

      await Html5QrCode.start(
        { facingMode: "environment" },
        { fps: 10, qrbox: { width: 250, height: 250 } },
        (decodedText: string) => {
          const qrDetails = parseQrData(decodedText);
          setQrData(qrDetails);
          stopScanner();
        },
        (errorMessage: string) => {
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
    if (Html5QrCode && isScanning) {
      try {
        await Html5QrCode.stop();
        setIsScanning(false);
        Html5QrCode = null;
      } catch (err) {
        console.error("Error stopping scanner:", err);
      }
    }
  };

  const captureImages = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');

    if (!video || !canvas || !ctx || !video.srcObject) {
      console.error('Video stream or canvas context is not ready or not found.');
      return;
    }

    let images: File[] = [];
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
          images.push(new File([blob], `capture${currentIndex}.jpg`, { type: 'image/jpeg' }));
        }
        currentIndex++;
        setTimeout(captureImage, 2000);
      }, 'image/jpeg');
    };

    setIsCapturing(true);
    captureImage();
  };
// Send images to backend
const sendImages = async (images: File[]) => {
  setIsUploading(true);
  setTrainingStatus(""); // Reset training status

  const formData = new FormData();
  images.forEach((image, index) => {
    formData.append("images", image, `capture${index}.jpg`);
  });
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

// Train user model
const handleTrainUser = async (rollNo: string) => {
  try {
    const response = await axios.post('http://localhost:4000/api/trainUser', { rollNo });
    console.log('User-specific model trained successfully:', response.data);
    setTrainingStatus("Trained Successfully");
  } catch (error) {
    console.error('Error training user-specific model:', error);
    setTrainingStatus("Training Failed");
  }
};


  const parseQrData = (qrText: string): QRData => {
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

      {qrData && Object.keys(qrData).length > 0 && (
        <div style={{ backgroundColor: "#2d3748", padding: "20px", borderRadius: "10px", margin: "20px auto", maxWidth: "500px" }}>
          <h3 style={{ color: "#63b3ed", marginBottom: "10px" }}>QR Data:</h3>
          <pre style={{ backgroundColor: "#1a202c", padding: "10px", borderRadius: "5px", overflowX: "auto" }}>
            {JSON.stringify(qrData, null, 2)}
          </pre>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "center", marginBottom: "20px" }}>
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
  onChange={(e) => {
    const files = e.target.files;
    if (files) {
      // Convert FileList to an array of files
      sendImages(Array.from(files));  // Pass the converted array
    } else {
      console.error("No files selected.");
    }
  }}
  disabled={isCapturing || isUploading}
/>

      </div>

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
};

export default LoginPage;




// import React, { useState, useEffect } from 'react';
// import { Html5Qrcode } from 'html5-qrcode';
// import axios from 'axios';

// const LoginPage = () => {
//   const [qrData, setQrData] = useState<string | null>(null);
//   const [isScanning, setIsScanning] = useState(false);
//   let Html5QrCode: Html5Qrcode | null = null;

//   useEffect(() => {
//     return () => {
//       if (Html5QrCode) {
//         Html5QrCode.stop().catch(err => console.error("Error stopping scanner during cleanup:", err));
//       }
//     };
//   }, []);

//   const startScanner = async () => {
//     try {
//       Html5QrCode = new Html5Qrcode("reader");
//       setIsScanning(true);

//       await Html5QrCode.start(
//         { facingMode: "environment" },
//         { fps: 10, qrbox: { width: 250, height: 250 } },
//         (decodedText: string) => {
//           setQrData(decodedText);
//           stopScanner();
//           sendToBackend(decodedText);
//         },
//         (errorMessage: string) => {
//           console.warn('QR Code scan error:', errorMessage);
//         }
//       );
//     } catch (error) {
//       console.error('Error starting scanner:', error);
//     }
//   };

//   const stopScanner = async () => {
//     if (Html5QrCode && isScanning) {
//       try {
//         await Html5QrCode.stop();
//         setIsScanning(false);
//         Html5QrCode = null;
//       } catch (err) {
//         console.error("Error stopping scanner:", err);
//       }
//     }
//   };

//   const sendToBackend = async (data: string) => {
//     try {
//       const parsedData = parseQrData(data);
//       const response = await axios.post('http://localhost:4000/api/saveDetails', parsedData);
//       console.log('Details saved successfully:', response.data);
//     } catch (error) {
//       console.error('Error sending data to backend:', error);
//     }
//   };

//   const parseQrData = (qrText: string) => {
//     const [rollNo, name, fatherName, department, contact] = qrText.split(',');
//     return {
//       rollNo: rollNo.trim(),
//       name: name.trim(),
//       fatherName: fatherName.trim(),
//       department: department.trim(),
//       contact: contact.trim(),
//     };
//   };

//   return (
//     <div style={{ backgroundColor: "#1a202c", color: "#f7fafc", minHeight: "100vh", padding: "20px", fontFamily: "Arial, sans-serif" }}>
//       <div style={{ textAlign: "center", marginBottom: "20px" }}>
//         <button
//           onClick={startScanner}
//           disabled={isScanning}
//           style={{
//             backgroundColor: "#3182ce",
//             color: "#fff",
//             padding: "10px 20px",
//             border: "none",
//             borderRadius: "5px",
//             fontSize: "16px",
//             cursor: "pointer",
//             boxShadow: "0 4px 6px rgba(0, 0, 0, 0.2)",
//           }}
//         >
//           Start Scanner
//         </button>
//       </div>
//       <div id="reader" style={{ width: "400px", height: "400px", margin: "20px auto", border: "2px solid #2d3748", borderRadius: "10px", backgroundColor: "#2d3748" }}></div>
//       {qrData && (
//         <div style={{ backgroundColor: "#2d3748", padding: "20px", borderRadius: "10px", margin: "20px auto", maxWidth: "500px" }}>
//           <h3 style={{ color: "#63b3ed", marginBottom: "10px" }}>QR Data:</h3>
//           <pre style={{ backgroundColor: "#1a202c", padding: "10px", borderRadius: "5px", overflowX: "auto" }}>
//             {qrData}
//           </pre>
//         </div>
//       )}
//     </div>
//   );
// };

// export default LoginPage;
