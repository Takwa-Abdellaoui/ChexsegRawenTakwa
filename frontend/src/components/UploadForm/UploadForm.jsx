import React from 'react';
import './UploadForm.css';

export default function UploadForm({ onFileChange, onSubmit, filename }) {
  return (
    <div className="upload-card">
      <input type="file" onChange={onFileChange} />
      <span>{filename}</span>
      <button onClick={onSubmit}>Analyser</button>
    </div>
  );
}
