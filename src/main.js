import './style.css';
import { io } from 'socket.io-client';

// Initialize Socket.IO
const socket = io();

socket.on('connect', () => {
  console.log('Connected to Socket.IO server');
});

socket.on('progress_update', (data) => {
  console.log('Progress update:', data);
  updateProgress(data);
});

function updateProgress(data) {
  const progressBar = document.getElementById('progressBar');
  const statusMessage = document.getElementById('statusMessage');
  
  if (progressBar) {
    progressBar.style.width = `${data.progress}%`;
    progressBar.textContent = `${data.progress}%`;
  }
  
  if (statusMessage) {
    statusMessage.textContent = data.status;
  }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
  initializeApp();
});

function initializeApp() {
  // Add event listeners and initialize UI components
  const uploadForm = document.getElementById('uploadForm');
  if (uploadForm) {
    uploadForm.addEventListener('submit', handleUpload);
  }
}

async function handleUpload(event) {
  event.preventDefault();
  const formData = new FormData(event.target);
  
  try {
    const response = await fetch('/api/upload', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    if (data.success) {
      window.location.href = `/analysis/${data.analysisId}`;
    } else {
      alert('Upload failed: ' + data.message);
    }
  } catch (error) {
    console.error('Upload error:', error);
    alert('An error occurred during upload');
  }
}