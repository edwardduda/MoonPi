import React from 'react';
import EpisodeVisualizer from './components/EpisodeVisualizer'; // Adjust the path if necessary

function App() {
  return (
    <div className="App">
      <h1 style={{ textAlign: 'center', margin: '20px 0' }}>DQN Episode Visualizer</h1>
      <EpisodeVisualizer />
    </div>
  );
}

export default App;