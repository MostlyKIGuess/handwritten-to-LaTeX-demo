import { useRef, useState } from 'react';
import './App.css';
import Canvas from './components/Canvas';
import Constants from './constants/constants';

function App() {
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

  return (
    <div className="App">
      <header className="App-header">
        <h1>Canvas Drawing</h1>
        <Canvas canvasRef={canvasRef} />
        <button className='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded'
          style={{
            padding: '10px',
            fontSize: '20px',
            borderRadius: '10px',
            cursor: 'pointer',
            backgroundColor: '#2196F3',
            color: 'white',
            margin: '2rem 0',
            marginBottom: '2rem',
          }}
          onClick={getLatex}
        >Get LaTeX</button>
      </header>
    </div>
  );
}

export default App;
