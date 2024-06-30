import { useRef, useEffect, useState } from 'react';

const Canvas = ({ socket, roomId }) => {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [color, setColor] = useState('#000000');
  const [fillColor, setFillColor] = useState('#FFFFFF');
  const [lineSize, setLineSize] = useState(5);
  const [isFilling, setIsFilling] = useState(false);
  const [isErasing, setIsErasing] = useState(false);
  const [history, setHistory] = useState([]);
  const [step, setStep] = useState(-1);
  const [showLastFrame, setShowLastFrame] = useState(false);
  const overlayCanvasRef = useRef(null);

  function isMobile() {
    
    const hasTouchScreen = 'ontouchstart' in window || navigator.maxTouchPoints > 0 || navigator.msMaxTouchPoints > 0;

    
    const userAgent = navigator.userAgent || navigator.vendor || window.opera;
    const isMobileDevice = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent.toLowerCase());

    return hasTouchScreen || isMobileDevice;
}


  useEffect(() => {


    const canvas = canvasRef.current;
    if(isMobile()){
      const ratio = 4/3;
      canvas.width = window.innerWidth*3/4;
      canvas.height = canvas.width / ratio;
    }else{
    canvas.width = 800;
    canvas.height = 600;
    }
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctxRef.current = ctx;

    const overlayCanvas = overlayCanvasRef.current;
    overlayCanvas.width = canvas.width;
    overlayCanvas.height = canvas.height;
    overlayCanvas.style.position = 'absolute';
    overlayCanvas.style.left = canvas.offsetLeft + 'px';
    overlayCanvas.style.top = canvas.offsetTop + 'px';
    overlayCanvas.style.pointerEvents = 'none'; // canvas click-through
    overlayCanvas.style.display = 'none'; // hiding the overlay 
  }, []);

  const captureState = () => {
    setTimeout(() => {
      const canvas = canvasRef.current;
      const imageData = canvas.toDataURL();
      const newHistory = history.slice(0, step + 1); 
      setHistory([...newHistory, imageData]);
      setStep(newHistory.length);
    }, 0);
  };

  const undoLastAction = () => {
    if (step > 0) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const previousState = new Image();
      previousState.src = history[step - 1];
      previousState.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(previousState, 0, 0, canvas.width, canvas.height);
      };
      setStep(step - 1);
    }
  };

  const startDrawing = (e) => {
    setDrawing(true);
    const correctedX = e.clientX - canvasRef.current.offsetLeft + window.scrollX;
    const correctedY = e.clientY - canvasRef.current.offsetTop + window.scrollY;
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(correctedX, correctedY);
    captureState();
  };
  
  const draw = (e) => {
    if (!drawing) return;
    const correctedX = e.clientX - canvasRef.current.offsetLeft + window.scrollX;
    const correctedY = e.clientY - canvasRef.current.offsetTop + window.scrollY;
    const ctx = ctxRef.current;
    if (isErasing) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.lineWidth = 10;
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = color;
      ctx.lineWidth = lineSize;
    }
    ctx.lineTo(correctedX, correctedY);
    ctx.stroke();
  };


  const stopDrawing = () => {
    setDrawing(false);
    ctxRef.current.beginPath();
  };

  

  

  const toggleLastFrame = () => {
    setShowLastFrame(!showLastFrame);
    const overlayCanvas = overlayCanvasRef.current;
    const overlayCtx = overlayCanvas.getContext('2d');
    if (!showLastFrame && history.length > 0) {
      overlayCanvas.style.display = 'block'; 
      const lastState = new Image();
      lastState.src = history[step-1];
      lastState.onload = () => {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        overlayCtx.drawImage(lastState, 0, 0, overlayCanvas.width, overlayCanvas.height);
      };
    } else {
      overlayCanvas.style.display = 'none'; 
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  };


  const startDrawingTouch = (e) => {
    e.preventDefault(); 
    setDrawing(true);
    const touch = e.touches[0];
    const correctedX = touch.clientX - canvasRef.current.offsetLeft + window.scrollX;
    const correctedY = touch.clientY - canvasRef.current.offsetTop + window.scrollY;
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(correctedX, correctedY);
    captureState();
  };
  const drawTouch = (e) => {
    e.preventDefault(); 
    if (!drawing) return;
    const touch = e.touches[0];
    const correctedX = touch.clientX - canvasRef.current.offsetLeft + window.scrollX;
    const correctedY = touch.clientY - canvasRef.current.offsetTop + window.scrollY;
    const ctx = ctxRef.current;
    if (isErasing) {
      ctx.globalCompositeOperation = 'destination-out';
      ctx.lineWidth = 10;
    } else {
      ctx.globalCompositeOperation = 'source-over';
      ctx.strokeStyle = color;
      ctx.lineWidth = lineSize;
    }
    ctx.lineTo(correctedX, correctedY);
    ctx.stroke();
    socket.emit('draw', {
      roomId,
      x: correctedX,
      y: correctedY,
      color,
      lineSize,
      isErasing,
    });
  };
  
  const stopDrawingTouch = (e) => {
    e.preventDefault(); 
    setDrawing(false);
    ctxRef.current.beginPath();
  };


  useEffect(() => {
    const canvas = canvasRef.current;

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);
    // canvas.addEventListener('click', handleCanvasClick);
    canvas.addEventListener('touchstart', startDrawingTouch, { passive: false });
    canvas.addEventListener('touchmove', drawTouch, { passive: false });
    canvas.addEventListener('touchend', stopDrawingTouch);
    canvas.addEventListener('touchcancel', stopDrawingTouch); 

    return () => {
      canvas.removeEventListener('touchstart', startDrawingTouch);
      canvas.removeEventListener('touchmove', drawTouch);
      canvas.removeEventListener('touchend', stopDrawingTouch);
      canvas.removeEventListener('touchcancel', stopDrawingTouch);
      
      canvas.removeEventListener('mousedown', startDrawing);
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('mouseup', stopDrawing);
      canvas.removeEventListener('mouseleave', stopDrawing);
    };
  }, []);

  useEffect(() => {
    if (socket) {
      const handleDraw = ({ x, y, color, lineSize, isErasing }) => {
        const ctx = ctxRef.current;
        if (isErasing) {
          ctx.globalCompositeOperation = 'destination-out';
        } else {
          ctx.globalCompositeOperation = 'source-over';
          ctx.strokeStyle = color;
          ctx.lineWidth = lineSize;
        }
        ctx.lineTo(x, y);
        ctx.stroke();
      };

      socket.on('draw', handleDraw);
      return () => {
        socket.off('draw', handleDraw);
      };
    }
  }, [socket]);

  const fillCanvas = () => {
    const ctx = ctxRef.current;
    ctx.fillStyle = fillColor;
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  };

  return (
    <div style={{ padding: '20px', backgroundColor: '#f2f2f2', borderRadius: '8px', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
      <canvas
        ref={canvasRef}
        onMouseDown={startDrawing}
        onMouseUp={stopDrawing}
        onMouseLeave={stopDrawing}
        onMouseMove={draw}
        onTouchStart={startDrawingTouch}
        onTouchMove={drawTouch}
        onTouchEnd={stopDrawingTouch}
        onTouchCancel={stopDrawingTouch}
        
        style={{ display: 'block', margin: '0 auto', backgroundColor: 'white', border: '1px solid #ccc', borderRadius: '5px' }}
      ></canvas>
       <canvas
      ref={overlayCanvasRef}
      style={{ display: 'none' }} 
    ></canvas>
      <div style={{ marginTop: '10px', display: 'flex', justifyContent: 'center', gap: '10px', flexWrap: 'wrap' }}>
        <input type="color" value={color} onChange={(e) => setColor(e.target.value)} style={{ width: '40px', height: '40px', border: 'none', cursor: 'pointer' }} />
        <input type="range" min="1" max="20" value={lineSize} onChange={(e) => setLineSize(e.target.value)} style={{ cursor: 'pointer' }} />
        <button onClick={() => setIsErasing(!isErasing)} style={{ backgroundColor: isErasing ? '#f44336' : '#4CAF50', color: 'white', border: 'none', borderRadius: '5px', padding: '10px 15px', cursor: 'pointer', fontSize: '16px' }}>{isErasing ? 'Draw' : 'Erase'}</button>
        <input type="color" value={fillColor} onChange={(e) => setFillColor(e.target.value)} style={{ width: '40px', height: '40px', border: 'none', cursor: 'pointer' }} />
        <button onClick={fillCanvas} style={{ backgroundColor: '#2196F3', color: 'white', border: 'none', borderRadius: '5px', padding: '10px 15px', cursor: 'pointer', fontSize: '16px' }}>Fill Canvas</button>
        <button onClick={undoLastAction} style={{ backgroundColor: '#FFC107', color: 'white', border: 'none', borderRadius: '5px', padding: '10px 15px', cursor: 'pointer', fontSize: '16px' }}>Undo</button>
        {/* <button onClick={toggleLastFrame} style={{ backgroundColor : showLastFrame? '#FFB6C1' :'pink', color: 'tumbleweed', border:'none',borderRadius:'5px',padding:'10px 15px',cursor:'pointer', }}>
        {showLastFrame ? 'Hide Last Frame' : 'Show Last Frame'}
        </button> */}
        {/* <button onClick={() => setIsFilling(!isFilling)} style={{ backgroundColor: isFilling ? '#FFB6C1' : 'pink', color: 'tumbleweed', border: 'none', borderRadius: '5px', padding: '10px 15px', cursor: 'pointer', fontSize: '16px' }}>{isFilling ? 'Stop Filling' : 'Fill'}</button> */}
     
      </div>
    </div>
  );
};

export default Canvas;