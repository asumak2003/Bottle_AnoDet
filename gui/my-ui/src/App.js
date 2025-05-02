import React, { useState, useEffect } from "react";
import "./App.css"; // Import external CSS file for styling

function App() {
  const [time, setTime] = useState(new Date().toLocaleTimeString());

  // Update the time every second
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(interval); // Cleanup interval
  }, []);

  return (
    <div className="app-container">
      {/* Top Title & Image */}
      <div className="header">
        <div className="clock">{time}</div> {/* Add Time Here */}
        <h1 className="title">Real-Time Anomaly Detection</h1>
        <img src="/images/Logo_HHN.png" alt="Logo" className="small-image" />
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Left: Live Stream */}
        <div className="live-container">
          <img src="http://localhost:5000/video" alt="Live Stream" className="live-stream" />
        </div>

        {/* Right: Buttons & Info */}
        <div className="right-panel">
          <h2>Controls</h2>
          <button className="btn btn-green" onClick={() => console.log("Start clicked")}>
            START
          </button>
          <button className="btn btn-red" onClick={() => console.log("Stop clicked")}>
            STOP
          </button>
          <div className="info-box">
            <p>Live anomaly detection in progress...</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
