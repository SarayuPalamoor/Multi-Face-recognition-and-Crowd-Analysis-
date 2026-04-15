import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Modal } from './ui/Modal';
import { Button } from './ui/Button';

interface PricingModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const PricingModal: React.FC<PricingModalProps> = ({ isOpen, onClose }) => {
  const navigate = useNavigate();

  const handleFaceRecognition = () => {
    onClose();
    navigate('/face-recognition');
  };

  const handleLogin = () => {
    onClose();
    navigate('/login');
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="grid md:grid-cols-2 gap-6 p-6">
        <div className="pricing-card bg-gray-800 p-8 rounded-lg text-center">
          <h3 className="text-2xl font-bold text-white mb-4"></h3>
          <div className="text-4xl font-bold text-white mb-4">Login</div>
          <p className="text-gray-400 mb-6">
            Clicking the Login button allows you to securely register your facial data on our
            platform using advanced recognition technology. This ensures a personalized and
            hassle-free experience for future logins, providing both convenience and enhanced
            security tailored specifically for you.
            <br/>
            <br />
          </p>
          <Button onClick={handleLogin}>Login</Button>
        </div>
        
        <div className="pricing-card bg-purple-600 p-8 rounded-lg text-center">
          <h3 className="text-2xl font-bold text-white mb-4"></h3>
          <div className="text-4xl font-bold text-white mb-4">Face Recognition</div>
          <p className="text-white/80 mb-6">
            By clicking the Face Recognition button, you can seamlessly track and verify registered
            faces using cutting-edge facial recognition technology. This feature allows you to
            manage and monitor recognized individuals effectively, ensuring secure and accurate
            identification for various purposes, from access control to attendance tracking.
          </p>
          <Button onClick={handleFaceRecognition}>Face Recognition</Button>
        </div>
      </div>
    </Modal>
  );
};