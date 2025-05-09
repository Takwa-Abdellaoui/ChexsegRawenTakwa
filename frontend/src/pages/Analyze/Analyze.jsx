import React, { useState } from 'react';
import axios from 'axios';
import './Analyze.css';

export default function Analyze() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [chart, setChart] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setPredictions([]);
    setChart(null);
  };

  const handleSubmit = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      const res = await axios.post('http://localhost:5000/predict', formData);
      const reader = new FileReader();

      reader.onloadend = () => {
        const base64Image = reader.result;
        const entry = {
          date: new Date().toLocaleString(),
          filename: file.name,
          image: base64Image,
          predictions: res.data.predictions || [],
        };
        const saved = JSON.parse(localStorage.getItem('history') || '[]');
        saved.unshift(entry);
        localStorage.setItem('history', JSON.stringify(saved));
      };

      reader.readAsDataURL(file);

      setPredictions(res.data.predictions || []);
      setChart(res.data.chart || null);
    } catch (err) {
      alert("Erreur pendant l'analyse.");
    }
    setLoading(false);
  };

  return (
    <div className="analyze-container">
      <h1>Analyse de radiographie</h1>

      <div className="upload-section">
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleSubmit} disabled={loading}>
          {loading ? "Analyse..." : "Analyser"}
        </button>
      </div>

      {predictions.length > 0 && (
        <div className="results-section">
          <p style={{ fontStyle: 'italic', color: 'gray' }}>
            Voici l’analyse de la radiographie. Les maladies considérées comme positives sont celles dont la probabilité dépasse 50 %.
          </p>

          {predictions.map((pred, idx) => (
            <div key={idx} className="prediction-block">
              <h3>{pred.label} — {(pred.score * 100).toFixed(1)}%</h3>
              <img
                src={`data:image/png;base64,${pred.cam}`}
                alt={`CAM pour ${pred.label}`}
                style={{ width: '300px', borderRadius: '8px' }}
              />
              <p>
                <strong>Interprétation :</strong> Cette pathologie a été détectée avec une forte probabilité. 
                Il est recommandé de consulter un professionnel de santé.
              </p>
            </div>
          ))}

          <h2 style={{ marginTop: '40px' }}>Vue globale des prédictions</h2>
          {chart && (
            <img
              src={`data:image/png;base64,${chart}`}
              alt="Graphique des probabilités"
              style={{ width: '90%', marginTop: '20px', borderRadius: '8px' }}
            />
          )}
        </div>
      )}
    </div>
  );
}
