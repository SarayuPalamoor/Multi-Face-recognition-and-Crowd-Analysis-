// import React, { useState } from 'react';
// import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import { Button } from './components/ui/Button';
// import { Scan } from 'lucide-react';
// import { PricingModal } from './components/PricingModal';
// import { FaceRecognition } from './pages/FaceRecognition';
// import LoginPage from './pages/LoginPage';
// import { CrowdAnalysis } from './pages/CrowdAnalysis';
// import CameraFeed  from './pages/camerafeed.tsx';
// function HomePage() {
//   const [isModalOpen, setIsModalOpen] = useState(false);

//   return (
//     <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center gap-8 p-4">
//       <div className="text-center space-y-4">
//         <h1 className="text-5xl font-bold text-white">
//           Multi-Level Face Recognition
//           <br />
//           and <span className="text-purple-400">Crowd Analysis</span> System
//         </h1>
//         <p className="text-gray-300 text-xl max-w-2xl mx-auto">
//           This project aims to develop a comprehensive, multi-level face recognition
//           and crowd analysis system.
//         </p>
//       </div>

//       <Button onClick={() => setIsModalOpen(true)}>
//         Get Started
//       </Button>

//       <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl w-full mt-8">
//         {[
//           {
//             title: 'Real-time Analysis',
//             description: 'Advanced algorithms for instant face detection and recognition',
//             icon: <Scan className="w-8 h-8 text-purple-400" />
//           },
//           {
//             title: 'Crowd Insights',
//             description: 'Detailed analytics and patterns in crowd behavior',
//             icon: <Scan className="w-8 h-8 text-purple-400" />
//           },
//           {
//             title: 'Security Focus',
//             description: 'Enhanced security measures with facial recognition',
//             icon: <Scan className="w-8 h-8 text-purple-400" />
//           }
//         ].map((feature, index) => (
//           <div
//             key={index}
//             className="bg-gray-800 p-6 rounded-lg text-center space-y-4 hover:bg-gray-700 transition-colors"
//           >
//             <div className="flex justify-center">{feature.icon}</div>
//             <h2 className="text-xl font-bold text-white">{feature.title}</h2>
//             <p className="text-gray-400">{feature.description}</p>
//           </div>
//         ))}
//       </div>

//       <PricingModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
//     </div>
//   );
// }

// function App() {
//   return (
//     <Router>
//       <Routes>
//         <Route path="/" element={<HomePage />} />
//         <Route path="/face-recognition" element={<FaceRecognition />} />
//         <Route path="/login" element={<LoginPage />} />
//         <Route path="/crowd-analysis" element={<CrowdAnalysis />} />
//         <Route path="/video_feed" element={<CameraFeed />} />

//       </Routes>
//     </Router>
//   );
// }

// export default App;

import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Button } from './components/ui/Button';
import { Scan } from 'lucide-react';
import { PricingModal } from './components/PricingModal';
import { FaceRecognition } from './pages/FaceRecognition';
import LoginPage from './pages/LoginPage';
import { CrowdAnalysis } from './pages/CrowdAnalysis';
import CameraFeed from './pages/camerafeed';
import ResultPage from './pages/resultpage';
function HomePage() {
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gray-900 flex flex-col items-center justify-center gap-8 p-4">
      <div className="text-center space-y-4">
        <h1 className="text-5xl font-bold text-white">
          Multi-Level Face Recognition
          <br />
          and <span className="text-purple-400">Crowd Analysis</span> System
        </h1>
        <p className="text-gray-300 text-xl max-w-2xl mx-auto">
          This project aims to develop a comprehensive, multi-level face recognition
          and crowd analysis system.
        </p>
      </div>

      <Button onClick={() => setIsModalOpen(true)}>
        Get Started
      </Button>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl w-full mt-8">
        {[
          {
            title: 'Real-time Analysis',
            description: 'Advanced algorithms for instant face detection and recognition',
            icon: <Scan className="w-8 h-8 text-purple-400" />
          },
          {
            title: 'Crowd Insights',
            description: 'Detailed analytics and patterns in crowd behavior',
            icon: <Scan className="w-8 h-8 text-purple-400" />
          },
          {
            title: 'Security Focus',
            description: 'Enhanced security measures with facial recognition',
            icon: <Scan className="w-8 h-8 text-purple-400" />
          }
        ].map((feature, index) => (
          <div key={index} className="bg-gray-800 p-6 rounded-lg text-center space-y-4 hover:bg-gray-700 transition-colors">
            <div className="flex justify-center">{feature.icon}</div>
            <h2 className="text-xl font-bold text-white">{feature.title}</h2>
            <p className="text-gray-400">{feature.description}</p>
          </div>
        ))}
      </div>

      <PricingModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </div>
  );
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/face-recognition" element={<FaceRecognition />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/crowd-analysis" element={<CrowdAnalysis />} />
        <Route path="/start-camera" element={<CameraFeed />} />
        <Route path="/result" element={<ResultPage />} /> {/* Add ResultPage */}
      </Routes>
    </Router>
  );
}

export default App;
