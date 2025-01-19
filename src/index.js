import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Optional: Add global styles here
import App from './App';

// Render the root React component (App) into the DOM
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);