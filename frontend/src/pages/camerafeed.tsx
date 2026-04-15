import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const CameraFeed: React.FC = () => {
  const navigate = useNavigate();
  const [recognizedRollNos, setRecognizedRollNos] = useState<string[] | null>(null); // To hold roll numbers
  const [loading, setLoading] = useState<boolean>(false); // Loading state
  const [isCameraStarted, setIsCameraStarted] = useState<boolean>(false); // Camera status

  // Function to start the camera by calling the backend
  const startCamera = async () => {
    setLoading(true); // Show loading state
    try {
      await fetch('http://127.0.0.1:5006/start_camera');
      setIsCameraStarted(true);
    } catch (error) {
      console.error('Error starting camera:', error);
    } finally {
      setLoading(false);
    }
  };

  // Function to stop the camera and fetch recognized roll numbers
  const stopCamera = async () => {
    setLoading(true); // Show loading state
    try {
      const response = await fetch('http://127.0.0.1:5006/stop_camera');
      const data = await response.json();
      if (data.status === 'Camera Stopped') {
        setRecognizedRollNos(data.recognized_roll_nos); // Set recognized roll numbers
      }
    } catch (error) {
      console.error('Error stopping camera:', error);
    } finally {
      setLoading(false);
    }
  };

  // Cleanup function to stop the camera if component unmounts
  useEffect(() => {
    if (isCameraStarted) {
      startCamera();
    }
    return () => {
      if (isCameraStarted) {
        stopCamera();
      }
    };
  }, [isCameraStarted]);

  // Function to handle back button click (stop camera and navigate to home)
  const handleBackClick = () => {
    stopCamera(); // Stop the camera
    navigate('/'); // Redirect to home page
  };

  return (
    <div className="flex justify-center items-center h-screen bg-black">
      {/* Display Camera Feed */}
      <img
        src="http://127.0.0.1:5006/video_feed"
        alt="Live Camera Feed"
        className="max-w-full h-auto"
      />

      {/* Back Button */}
      <div className="absolute top-4 left-4 z-10">
        <button
          onClick={handleBackClick}
          className="bg-red-500 text-white px-4 py-2 rounded"
        >
          Back
        </button>
      </div>

      {/* Loading Spinner */}
      {loading && <div className="loading-spinner">Loading...</div>}

      {/* Display Recognized Roll Numbers */}
      {recognizedRollNos && recognizedRollNos.length > 0 && (
        <div className="absolute bottom-4 left-4 z-10 text-white">
          <h2>Recognized Roll Numbers:</h2>
          <ul>
            {recognizedRollNos.map((rollNo, index) => (
              <li key={index}>{rollNo}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default CameraFeed;

