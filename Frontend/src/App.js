import logo from './logo.svg';
import './App.css';
import Canvas from './components/Canvas';

function App() {
  return (
    <div className="App">
      <header className="App-header">
      <h1>Canvas Drawing</h1>
        <Canvas />
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
        >Get LaTeX</button>
      </header>
    </div>
  );
}

export default App;
