import React from "react";
import "./PredictionCard.css";

const PredictionCard = ({ pathology, score }) => {
  return (
    <div className="prediction-card">
      <h3>{pathology}</h3>
      <div className="progress-bar">
        <div
          className="progress-bar-fill"
          style={{ width: `${Math.round(score * 100)}%` }}
        ></div>
      </div>
      <p>{(score * 100).toFixed(1)}%</p>
    </div>
  );
};

export default PredictionCard;
