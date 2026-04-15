// import React from 'react';
// import { Modal } from './ui/Modal';
// import { Button } from './ui/Button';
// import { Camera, Upload } from 'lucide-react';
// import { useNavigate } from 'react-router-dom';
// interface MultiFaceModalProps {
//   isOpen: boolean;
//   onClose: () => void;
// }

// export const MultiFaceModal: React.FC<MultiFaceModalProps> = ({ isOpen, onClose }) => {
//   const navigate = useNavigate(); // Correct use of useNavigate

//   const startCamera = () => {
//     navigate('/start-camera'); // Directly navigate to the desired route
//   };
//   return (
//     <Modal isOpen={isOpen} onClose={onClose}>
//       <div className="grid md:grid-cols-2 gap-6 p-6">
//         <div className="bg-gray-800 p-8 rounded-lg text-center">
//           <div className="flex justify-center mb-4">
//             <Camera className="w-12 h-12 text-purple-400" />
//           </div>
//           <h3 className="text-2xl font-bold text-white mb-4">Live Camera</h3>
//           <p className="text-gray-400 mb-6">
//             Use your device's camera to capture and analyze multiple faces in real-time
//             with our advanced recognition system.
//           </p>
//           <Button onClick={startCamera}>Start Camera</Button>
//         </div>
        
//         <div className="bg-purple-600 p-8 rounded-lg text-center">
//           <div className="flex justify-center mb-4">
//             <Upload className="w-12 h-12 text-white" />
//           </div>
//           <h3 className="text-2xl font-bold text-white mb-4">Upload Image</h3>
//           <p className="text-white/80 mb-6">
//             Upload an image containing multiple faces for instant analysis using our
//             advanced facial recognition algorithms.
//           </p>
//           <Button onClick={() => window.location.href = 'http://localhost:5000/'}>Upload Image</Button>

//         </div>
//       </div>
//     </Modal>
//   );
// };



// import React from 'react';
// import { Modal } from './ui/Modal';
// import { Button } from './ui/Button';
// import { Camera, Upload } from 'lucide-react';
// import { useNavigate } from 'react-router-dom';
// interface MultiFaceModalProps {
//   isOpen: boolean;
//   onClose: () => void;
// }

// export const MultiFaceModal: React.FC<MultiFaceModalProps> = ({ isOpen, onClose }) => {
//   const navigate = useNavigate(); // Correct use of useNavigate

//   const startCamera = () => {
//     navigate('/start-camera'); // Directly navigate to the desired route
//   };
//   return (
//     <Modal isOpen={isOpen} onClose={onClose}>
//       <div className="grid md:grid-cols-2 gap-6 p-6">
//         <div className="bg-gray-800 p-8 rounded-lg text-center">
//           <div className="flex justify-center mb-4">
//             <Camera className="w-12 h-12 text-purple-400" />
//           </div>
//           <h3 className="text-2xl font-bold text-white mb-4">Live Camera</h3>
//           <p className="text-gray-400 mb-6">
//             Use your device's camera to capture and analyze multiple faces in real-time
//             with our advanced recognition system.
//           </p>
//           <Button onClick={startCamera}>Start Camera</Button>
//         </div>
        
//         <div className="bg-purple-600 p-8 rounded-lg text-center">
//           <div className="flex justify-center mb-4">
//             <Upload className="w-12 h-12 text-white" />
//           </div>
//           <h3 className="text-2xl font-bold text-white mb-4">Upload Image</h3>
//           <p className="text-white/80 mb-6">
//             Upload an image containing multiple faces for instant analysis using our
//             advanced facial recognition algorithms.
//           </p>
//           <Button onClick={() => window.location.href = 'http://localhost:5000/'}>Upload Image</Button>

//         </div>
//       </div>
//     </Modal>
//   );
// };
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Modal } from './ui/Modal';
import { Button } from './ui/Button';
import { Camera, Upload } from 'lucide-react';

interface MultiFaceModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MultiFaceModal: React.FC<MultiFaceModalProps> = ({ isOpen, onClose }) => {
  const navigate = useNavigate(); // Correct use of useNavigate

  const startCamera = () => {
    navigate('/start-camera'); // Directly navigate to the desired route
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className="grid md:grid-cols-2 gap-6 p-6">
        <div className="bg-gray-800 p-8 rounded-lg text-center">
          <div className="flex justify-center mb-4">
            <Camera className="w-12 h-12 text-purple-400" />
          </div>
          <h3 className="text-2xl font-bold text-white mb-4">Live Camera</h3>
          <p className="text-gray-400 mb-6">
            Use your device's camera to capture and analyze multiple faces in real-time
            with our advanced recognition system.
          </p>
          <Button onClick={startCamera}>Start Camera</Button>
        </div>

        <div className="bg-purple-600 p-8 rounded-lg text-center">
          <div className="flex justify-center mb-4">
            <Upload className="w-12 h-12 text-white" />
          </div>
          <h3 className="text-2xl font-bold text-white mb-4">Upload Image</h3>
          <p className="text-white/80 mb-6">
            Upload an image containing multiple faces for instant analysis using our
            advanced facial recognition algorithms.
          </p>
          <Button onClick={() => window.location.href = 'http://localhost:5000/'}>Upload Image</Button>
        </div>
      </div>
    </Modal>
  );
};