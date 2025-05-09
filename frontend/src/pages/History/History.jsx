import React, { useEffect, useState } from 'react';
import './History.css';

export default function History() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem('history') || '[]');
    setHistory(saved);
  }, []);

  const handleDelete = (index) => {
    const updated = [...history];
    updated.splice(index, 1);
    setHistory(updated);
    localStorage.setItem('history', JSON.stringify(updated));
  };

  return (
    <div className="history-container">
      <h1>Historique des analyses</h1>

      {history.length === 0 ? (
        <p>Aucune analyse enregistrée pour le moment.</p>
      ) : (
        <div className="history-list">
          {history.map((entry, idx) => (
            <div key={idx} className="history-card">
              <img src={entry.image} alt="Radiographie" className="history-img" />
              <div className="history-info">
                <h3>{entry.date}</h3>
                <p><strong>Fichier :</strong> {entry.filename}</p>
                <h4>Pathologies détectées :</h4>
                <ul>
                  {entry.predictions
                    .filter(p => p.score > 0.5)
                    .map((pred, i) => (
                      <li key={i}>
                        ✅ <strong>{pred.label}</strong> — {(pred.score * 100).toFixed(1)}%
                      </li>
                  ))}
                </ul>
                <button onClick={() => handleDelete(idx)} className="delete-btn">🗑 Supprimer</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
