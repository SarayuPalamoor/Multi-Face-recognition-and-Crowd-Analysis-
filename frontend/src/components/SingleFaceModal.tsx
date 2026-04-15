import React from 'react';
import { Modal } from './ui/Modal';
import { Button } from './ui/Button';

interface SingleFaceModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const SingleFaceModal: React.FC<SingleFaceModalProps> = ({ isOpen, onClose }) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="grid md:grid-cols-2 gap-6 p-6">
        <div className="bg-gray-800 p-8 rounded-lg text-center">
          <h3 className="text-2xl font-bold text-white mb-4">Without Pretrained</h3>
          <p className="text-gray-400 mb-6">
            A face recognition model without pretraining is built from scratch without relying on pre-existing weights,
            requiring full training on raw data for custom feature learning.
          </p>
          <Button onClick={() => window.location.href = "http://localhost:5003"}>GO</Button>
        </div>
        
        <div className="bg-purple-600 p-8 rounded-lg text-center">
          <h3 className="text-2xl font-bold text-white mb-4">With Pretrained</h3>
          <p className="text-white/80 mb-6">
            A face recognition model with pretraining uses pre-existing weights from a model trained on large datasets.
            It leverages learned facial features, reducing training time and data requirements.
          </p>
          <Button onClick={() => window.location.href = "http://localhost:5002"}>GO</Button>
        </div>
      </div>
    </Modal>
  );
};