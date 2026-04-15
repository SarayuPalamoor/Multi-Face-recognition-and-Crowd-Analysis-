import React, { useEffect, useState } from 'react';

const ResultPage: React.FC = () => {
  const [rollNumbers, setRollNumbers] = useState<string>('Unknown');

  useEffect(() => {
    fetch('http://127.0.0.1:5006/result')
      .then((res) => res.json())
      .then((data) => setRollNumbers(data.roll_numbers))
      .catch((err) => console.error('Error fetching results:', err));
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-screen bg-gray-900 text-white">
      <h1 className="text-4xl font-bold mb-4">Recognition Results</h1>
      <p className="text-xl">Detected Roll Numbers: {rollNumbers}</p>
      <button
        onClick={() => window.history.back()} // Navigate back
        className="mt-6 bg-purple-500 text-white px-4 py-2 rounded-lg"
      >
        Back
      </button>
    </div>
  );
};

export default ResultPage;


