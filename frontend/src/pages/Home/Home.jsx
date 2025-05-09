import React from "react";
import { useNavigate } from "react-router-dom";
import "./Home.css";

const Home = () => {
  const navigate = useNavigate();

  const handleAnalyzeClick = () => {
    navigate("/analyze");
  }

  return (
    <div className="home">
      <section className="hero">
        <div className="hero-content">
          <h1>Bienvenue sur <span className="highlight">CheXSeg AI</span></h1>
          <p>Analyse intelligente de vos radiographies thoraciques grâce à l’Intelligence Artificielle</p>
          <div className="buttons">
            <button onClick={handleAnalyzeClick}>Commencer l'analyse</button>
            
          </div>
        </div>
      </section>

      <section className="features">
        <h2>Pourquoi utiliser CheXSeg ?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3>Rapide</h3>
            <p>Résultats instantanés, en moins de 3 secondes par image.</p>
          </div>
          <div className="feature-card">
            <h3>Fiable</h3>
            <p>Modèles CheXNet entraînés sur la base ChestXray14.</p>
          </div>
          <div className="feature-card">
            <h3>Visualisé</h3>
            <p>Masques colorés superposés sur vos images pour une compréhension claire.</p>
          </div>
          <div className="feature-card">
            <h3>Sécurisé</h3>
            <p>Traitement local, aucune donnée médicale partagée.</p>
          </div>
        </div>
      </section>

      <section className="how-it-works">
        <h2>Comment ça marche ?</h2>
        <ol>
          <li>Importez votre radiographie thoracique.</li>
          <li>Notre IA détecte 14 pathologies pulmonaires possibles.</li>
          <li>Visualisez les zones suspectes via des cartes de chaleur (CAMs).</li>
        </ol>
      </section>

      <section className="testimonials">
        <h2>Ils nous font confiance</h2>
        <div className="testimonials-grid">
          <blockquote>
            <p>“L’outil m’a permis de détecter un œdème pulmonaire non vu en première lecture.”</p>
            <footer>— Dr. Salma, pneumologue</footer>
          </blockquote>
          <blockquote>
            <p>“J’ai pu avoir une idée de mon état avant même le rendez-vous médical.”</p>
            <footer>— Amine, patient</footer>
          </blockquote>
        </div>
      </section>
    </div>
  );
};

export default Home;
