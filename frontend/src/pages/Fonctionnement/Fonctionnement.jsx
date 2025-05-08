import React from 'react';
import './Fonctionnement.css';

export default function Fonctionnement() {
  return (
    <div className="fonctionnement-container">
      <h1>Fonctionnement du Modèle CheXNet</h1>
      <p>
        Ce site utilise un modèle d’intelligence artificielle appelé <strong>CheXNet</strong>, basé sur une architecture DenseNet121.
        Il a été entraîné pour détecter <strong>14 pathologies thoraciques</strong> à partir de radiographies pulmonaires.
      </p>

      <h2>🧠 Étapes du pipeline d’analyse</h2>
      <ol>
        <li>Chargement de l’image radiographique.</li>
        <li>Prétraitement (mise à l’échelle, normalisation).</li>
        <li>Passage dans le modèle CheXNet pour classification.</li>
        <li>Si une pathologie est détectée (score &gt; 50%), génération d’une <strong>CAM</strong> (Class Activation Map).</li>
        <li>Superposition de la CAM à l’image pour visualisation.</li>
      </ol>

      <h2>📊 Interprétation des résultats</h2>
      <p>
        Chaque pathologie détectée est associée à une probabilité. Nous considérons une pathologie comme <strong>positive si le score &gt; 50%</strong>.
      </p>

      <h2>🌈 Que sont les CAMs ?</h2>
      <p>
        Les <strong>Class Activation Maps</strong> sont des images colorées superposées sur les radiographies. Elles montrent les zones qui ont contribué à la décision du modèle pour une pathologie spécifique.
      </p>

      <div className="cam-example">
       <img src="/assets/cams/00000001_000_Atelectasis.png" alt="Exemple de CAM" />
       <p>Exemple : zone en rouge indiquant une consolidation pulmonaire.</p>
      </div>


      <h2>🧩 À propos de CheXNet</h2>
      <p>
        CheXNet est une version modifiée du réseau DenseNet121, optimisée pour la détection multi-label sur le jeu de données <strong>ChestX-ray14</strong>
        développé par les NIH. Il est capable de produire des résultats comparables à ceux d’un radiologue sur certaines tâches.
      </p>

      <h2>🔍 Transparence & Limites</h2>
      <ul>
        <li>Le modèle ne remplace pas un diagnostic médical.</li>
        <li>Les résultats peuvent varier en fonction de la qualité des images.</li>
        <li>Les seuils peuvent être ajustés selon le besoin clinique.</li>
      </ul>
    </div>
  );
}
