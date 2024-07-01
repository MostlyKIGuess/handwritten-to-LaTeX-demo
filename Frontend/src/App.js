import './App.css';
import Canvas from './components/Canvas';
import React from 'react';
import { useRef, useState } from 'react';
import Constants from './constants/constants';

function App() {
  const [contentToCopy, setContentToCopy] = useState('')
  const constants = new Constants();
  const canvasRef = useRef(null);
  const [LAPI_resp, setLAPI_Resp] = useState(null);
  const getLatex = async () => {
    const imgData = canvasRef.current.toDataURL('image/png');
    const resp = await fetch(`${constants.SERVER_BASE_URL}/${constants.EP_POST_CANVAS}`, {
      method: "POST",
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imageData: imgData }),
    });
    const sresp = await resp.json();
    setLAPI_Resp(sresp);
    console.log(LAPI_resp);
  }


  const handleCopyClick = () => {
    navigator.clipboard.writeText(contentToCopy)
      .then(() => {
        console.log('Content copied successfully!');
      })
      .catch(err => {
        console.error('Failed to copy: ', err);
      });
  };

  
  return (
    <div className="App bg-gray-100 m-2 p-4 sm:p-8">
      <h1 className='text-4xl mb-4 font-bold text-center md:text-left pl-8'>Canvas Drawing</h1>
      <div className='flex flex-col sm:flex-row gap-4'>
        <div className='w-full sm:w-3/5 shadow-lg bg-gray-200 rounded-lg p-4'>
          <Canvas canvasRef={canvasRef}/>
        </div>
        <div className='w-full sm:w-2/5 flex flex-col gap-4'>
          <div className='flex-1 bg-white p-4   shadow-md rounded-lg border'>
              <h1 className='text-2xl font-bold mb-2'>
              ASCII Output
              </h1>
              <div>
                Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec
                Content goes here
              </div>
          </div>
          <div className='flex-1 bg-white p-4 shadow-md rounded-lg border'>
            <h1 className='text-2xl font-bold mb-2'>
              LaTeX Code
            </h1>
            <div className='bg-gray-100 p-2 rounded font-mono text-sm overflow-auto'>
              Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam nec
              Content goes here/
            </div>
            <button 
            onClick={handleCopyClick}
            className='mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg block mx-auto'>
              Copy
            </button>
          </div>
        </div>
      </div>
      <button
        onClick={getLatex}
      className='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg my-8 mx-auto block'>
        Get LaTeX
      </button>
    </div>
  );
}

export default App;