import './App.css';
import Canvas from './components/Canvas';
import { useEffect, useRef, useState } from 'react';
import Constants from './constants/constants';

function App() {
  const [contentToCopy, setContentToCopy] = useState('')
  const constants = new Constants();
  const canvasRef = useRef(null);
  const ltxcbRef = useRef(null);
  const [LAPI_resp, setLAPI_Resp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [modelDownloaded, setModelDownloaded] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [modelDeatils, setModelDetails] = useState(null);

  const fetchModelDeatils = async () => {
    const resp = await fetch(`${constants.SERVER_BASE_URL}/model_details`)
    const data = await resp.json();
    if (data.error === undefined) {
      setModelDetails(data);
    }
    else{
      alert(data.error)
    }
  }
  useEffect(()=>{
    fetchModelDeatils();
  }, [])
  const checkPredictionStatus = async (fileId) => {
    setLoading(true);
    const intervalId = setInterval(async () => {
      const response = await fetch(`${constants.SERVER_BASE_URL}/prediction_status/${fileId}`);
      const data = await response.json();
      if (data.status === 'completed') {
        clearInterval(intervalId);
        setLoading(false);
        setLAPI_Resp(data.result);
        setContentToCopy(data.result.latex_text);

      } else if (data.status === -1) {
        clearInterval(intervalId);
        setLoading(false);
        setErrorMessage('Prediction data not found for the file ID');
      }
    
    }, 1000);
  };
  
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
    if (resp.ok) {
      checkPredictionStatus(sresp.file_id);
    } else {
      console.error(sresp.error);
    }
  }
  
  const getJobStatus = async (track_id) => {
    const resp = await fetch(`${constants.SERVER_BASE_URL}/${constants.QUEUE_STATUS}/${track_id}`, {
      method: "GET",
      headers: {
        'Content-Type': 'application/json',
      },
    });
    const sresp = await resp.json();
    setLAPIlv2_resp(sresp);
    console.log(LAPIlv2_resp);
  }

  const handleCopyClick = () => {
    setContentToCopy(ltxcbRef.current.innerText);
    navigator.clipboard.writeText(`$${contentToCopy}$`)
      .then(() => {
        alert('Content copied successfully!');
      })
      .catch(err => {
        console.error('Failed to copy: ', err);
      });
  };

  const downloadModel = async () => {
    try {
      setDownloadProgress(0);
      const interval = setInterval(() => {
        setDownloadProgress((oldProgress) => {
          if (oldProgress >= 100) {
            clearInterval(interval);
            setModelDownloaded(true);
            return 100;
          }
          return oldProgress + 10;
        });
      }, 1000);
    } catch (error) {
      console.error('Error downloading the model:', error);
      setErrorMessage('Failed to download the model. Please try again.');
    }
  };

  
  return (
    <div className="App bg-gray-100 p-4 sm:p-8">
      {modelDeatils && <div className='status-boxes flex justify-between mb-4'>
        <div className='status-box bg-green-200 p-2 rounded'>Model: {modelDeatils.name.replace(/^./, str => str.toUpperCase())}</div>
        <div className='status-box bg-blue-200 p-2 rounded'>Accuracy: {modelDeatils.accurcay}%</div>
        <div className='status-box bg-yellow-200 p-2 rounded'>Loss Rate: {modelDeatils.loss.toFixed(2)}</div>
        <div className='status-box bg-red-200 p-2 rounded'>Last Updated: {modelDeatils.updated}</div>
      </div>}
      <div className='main-content flex flex-col items-center gap-4'>
        <div className='flex flex-col sm:flex-row gap-4 w-full'>
          <div className='w-full sm:w-3/5 shadow-lg bg-gray-200 rounded-lg'>
            <Canvas canvasRef={canvasRef} />
          </div>
          <div className='w-full sm:w-2/5 flex flex-col gap-4'>
          {/* {llmPanelVisible && (
        <div className="llm-panel bg-white p-4 shadow-md rounded-lg">
          <textarea value={llmQuery} onChange={handleLlmQueryChange} placeholder="Ask the AI anything..." className="w-full p-2 mb-4 border rounded" />

          <button onClick={downloadModel} className="bg-green-500 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg">
            Download LLM
          </button>

          {downloadProgress > 0 && downloadProgress < 100 && (
            <div>Downloading model... {downloadProgress}%</div>
          )}

          {modelDownloaded && (
            <div>Model downloaded successfully!</div>
          )}

          <button onClick={submitLlmQuery} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg">
            Submit
          </button>
          <div className="mt-4">
            <h3 className="text-lg font-bold">AI Response:</h3>
            <p>{llmResponse}</p>
          </div>
        </div>
      )} */}
            <div className='flex-1 bg-white p-4 shadow-md rounded-lg border'>
              <h1 className='text-2xl font-bold mb-2'>
              Output
              </h1>
              {
                loading ? <div>Loading...</div> : errorMessage ? <div>{errorMessage}</div> : <div>{LAPI_resp&&LAPI_resp.plain_text}</div>
              }
            </div>
            <div className='flex items-center justify-center'>
            <button onClick={getLatex} className='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg'>
              Get LaTeX
            </button>
            </div>
            <div className='flex-1 bg-white p-4 shadow-md rounded-lg border'>
              <h1 className='text-2xl font-bold mb-2'>
                LaTeX Code
              </h1>
              <div ref={ltxcbRef} className='bg-gray-100 p-2 rounded font-mono text-sm overflow-auto'>
                {
                  loading ? <div>Loading...</div> : errorMessage ? <div>{errorMessage}</div> : <div>{LAPI_resp&&LAPI_resp.latex_text}</div>
                }
              </div>
              <button
                onClick={handleCopyClick}
                className='mt-4 bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg block mx-auto'>
                Copy
              </button>
            </div>
          </div>
        </div>
    
      </div>
    </div>
  );
}

export default App;