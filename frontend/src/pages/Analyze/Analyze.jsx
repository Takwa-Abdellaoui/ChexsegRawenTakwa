import React, { useState } from 'react';
import axios from 'axios';
import './Analyze.css';

export default function Analyze() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [chart, setChart] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setPredictions([]);
    setChart(null);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError(null);

    try {
      const res = await axios.post('http://localhost:5000/predict', formData);

      // ✅ Affichage du message d'erreur si image non autorisée
      if (res.data.error) {
        setError(res.data.error);
        setLoading(false);
        return;
      }

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
      setError("Erreur pendant l'analyse,Ce fichier ne semble pas être une radiographie thoracique .");
    }

    setLoading(false);
  };

  return (
    <div className="analyze-container">
      <h1>Analyse de radiographie</h1>

      <div className="upload-section">
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleSubmit} disabled={loading || !file}>
          {loading ? "Analyse en cours..." : "Analyser"}
        </button>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {predictions.length > 0 && (
        <div className="results-section">
          <p className="note">
            Analyse des pathologies avec une probabilité supérieure à 50 %.
          </p>

          {predictions.map((pred, idx) => (
            <div key={idx} className="prediction-block">
              <h3>{pred.label} — {(pred.score * 100).toFixed(1)}%</h3>
              <img
                src={`data:image/png;base64,${pred.cam}`}
                alt={`CAM pour ${pred.label}`}
                className="cam-image"
              />
              <p>
                <strong>Remarque :</strong> Résultat suggéré par l’IA. Une confirmation médicale est conseillée.
              </p>
            </div>
          ))}

          <h2 className="chart-title">Diagramme des prédictions</h2>
          {chart && (
            <img
              src={`data:image/png;base64,${chart}`}
              alt="Graphique des probabilités"
              className="chart-image"
            />
          )}
        </div>
      )}
    </div>
  );
}
