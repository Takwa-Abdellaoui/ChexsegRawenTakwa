import React from "react";
import { Link, useLocation } from "react-router-dom";
import "./Footer.css";

const Footer = () => {
  const location = useLocation();
  const isAnalyzePage = location.pathname === "/analyze";

  return (
    <footer className={`footer ${isAnalyzePage ? "footer-analyze" : ""}`}>
      <div className="footer-content">
        <div className="footer-column">
          <h4>Navigation</h4>
          <ul>
            <li><Link to="/">Accueil</Link></li>
            <li><Link to="/analyze">Analyse</Link></li>
            <li><Link to="/history">Historique</Link></li>
            <li><Link to="/pathologies">Pathologies</Link></li>
            <li><Link to="/fonctionnement">Fonctionnement</Link></li>
          </ul>
        </div>

        <div className="footer-column">
          <h4>À propos du projet</h4>
          <p><strong>CheXSeg</strong> est un outil de détection et de segmentation des pathologies pulmonaires à partir de radiographies thoraciques, combinant intelligence artificielle et expertise médicale.</p>
          <p>Développé dans le cadre du PCD ENSI 2025.</p>
        </div>

        <div className="footer-column">
          <h4>Contact & Équipe</h4>
          <p>Rawen Sahraoui </p>
          <p>Takwa Abdellaoui</p>
          <p><Link to="/contact">Nous contacter</Link></p>
        </div>
      </div>

      <div className="footer-bottom">
        <p>&copy; 2025 CheXSeg – Tous droits réservés | <Link to="https://www.inserm.fr/dossier/intelligence-artificielle-et-sante/">Mentions légales</Link></p>
      </div>
    </footer>
  );
};

export default Footer;
