import React, { useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { SingleFaceModal } from '../components/SingleFaceModal';
import { MultiFaceModal } from '../components/MultiFaceModal';

export const FaceRecognition = () => {
  const [isSingleFaceModalOpen, setSingleFaceModalOpen] = useState(false);
  const [isMultiFaceModalOpen, setMultiFaceModalOpen] = useState(false);
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <Link to="/" className="inline-flex items-center text-purple-400 hover:text-purple-300 mb-8">
        <ArrowLeft className="w-5 h-5 mr-2" />
        Back to Home
      </Link>

      <div className="max-w-7xl mx-auto space-y-16">
        <section className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="col-span-1 md:col-span-3">
            <h1 className="text-4xl font-bold text-white mb-4">Face Recognition System</h1>
            <p className="text-gray-300 text-lg">
              Advanced facial recognition technology powered by deep learning algorithms
            </p>
          </div>

          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <img
              src="https://www.deepseadev.com/wp-content/uploads/2024/03/cons-of-facial-recognition.jpg"
              alt="Single Face Recognition"
              className="w-full h-48 object-cover"
            />
            <div className="p-6">
              <h2 className="text-xl font-bold text-white mb-2">Single Face Recognition</h2>
              <p className="text-gray-400 mb-4">
                Our system uses state-of-the-art facial recognition technology to accurately identify
                and analyze facial features in real-time.
              </p>
              <Button onClick={() => setSingleFaceModalOpen(true)}>Try Now</Button>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <img
              src="https://www.jeremymathew.com/wp-content/uploads/2024/01/fcnn.jpg"
              alt="Multi-Face Recognition"
              className="w-full h-48 object-cover"
            />
            <div className="p-6">
              <h2 className="text-xl font-bold text-white mb-2">Multi-Face Recognition</h2>
              <p className="text-gray-400 mb-4">
                Multi-level face recognition that leverages hierarchical feature extraction, combining global
                and local facial attributes with deep learning embeddings.
              </p>
              {/* <Button onClick={() => setMultiFaceModalOpen(true)}>Analyze Multiple Faces</Button> */}
              <Button onClick={() => window.location.href = 'http://localhost:5000/'}>Upload Image</Button>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <img
              src="https://i0.wp.com/videoanalytics.in/wp-content/uploads/2024/03/Crowd-Analysis.png?fit=1200%2C300&ssl=1"
              alt="Crowd Analysis"
              className="w-full h-48 object-cover"
            />
            <div className="p-6">
              <h2 className="text-xl font-bold text-white mb-2">Crowd Analysis</h2>
              <p className="text-gray-400 mb-4">
                Crowd analysis that utilizes multi-level feature extraction to analyze individual and collective
                behaviors, integrating spatial, temporal, and density-based insights.
              </p>
              <Button onClick={() => window.location.href = 'http://localhost:5001/'}>
                Analyze Crowd
              </Button>
            </div>
          </div>
        </section>
      </div>

      <SingleFaceModal isOpen={isSingleFaceModalOpen} onClose={() => setSingleFaceModalOpen(false)} />
      <MultiFaceModal isOpen={isMultiFaceModalOpen} onClose={() => setMultiFaceModalOpen(false)} />
    </div>
  );
};