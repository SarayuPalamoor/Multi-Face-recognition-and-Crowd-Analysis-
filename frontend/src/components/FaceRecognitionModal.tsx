import React from 'react';
import { X } from 'lucide-react';
import { Modal } from './ui/Modal';
import { Button } from './ui/Button';

interface FaceRecognitionModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const FaceRecognitionModal: React.FC<FaceRecognitionModalProps> = ({ isOpen, onClose }) => {
  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="grid md:grid-cols-2 gap-6 p-6">
        <div className="bg-gray-800 p-8 rounded-lg text-center">
          <h3 className="text-2xl font-bold text-white mb-4">Customized</h3>
          <p className="text-gray-400 mb-6">
           A face recognition model without pretraining is built from scratch without relying on pre-existing weights, requiring full training on raw data for custom feature learning. This approach allows for greater customization but demands extensive data, longer training time, and significant computational resources to ensure accurate and robust performance.
            <br/>
            <br/>
          </p>
          <Button onClick={onClose}>GO</Button>
        </div>
        
        <div className="bg-purple-600 p-8 rounded-lg text-center">
          <h3 className="text-2xl font-bold text-white mb-4">Trained</h3>
          <p className="text-white/80 mb-6">
           A face recognition model with pretraining uses pre-existing weights from a model trained on large datasets. It leverages learned facial features, reducing training time and data requirements. This approach is efficient and often achieves high accuracy with minimal fine-tuning, making it ideal for rapid deployment and scenarios with limited data. 
            <br />
            <br />
          </p>
          <Button onClick={onClose}>GO</Button>
        </div>
      </div>
    </Modal>
  );
};