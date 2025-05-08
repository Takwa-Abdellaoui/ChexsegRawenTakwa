import React, { useEffect, useState } from 'react';
import './History.css';

export default function History() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem('history') || '[]');
    setHistory(saved);
  }, []);

  return (
    <div className="history-container">
      <h1>Historique des analyses</h1>

      {history.length === 0 ? (
        <p>Aucune analyse enregistrée pour le moment.</p>
      ) : (
        <div className="history-list">
          {history.map((entry, idx) => (
            <div key={idx} className="history-card">
              <h3>{entry.date}</h3>
              <p><strong>Fichier analysé :</strong> {entry.filename}</p>
              <h4>Pathologies détectées :</h4>
              <ul>
                {entry.predictions
                  .filter(p => p.score > 0.5)
                  .map((pred, i) => (
                    <li key={i}>
                      ✅ <strong>{pred.pathology}</strong> — {(pred.score * 100).toFixed(1)}%
                    </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
