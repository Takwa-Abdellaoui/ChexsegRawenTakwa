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
          <h4>Projet</h4>
          <p>CheXSeg</p>
          <p>PCD ENSI 2025</p>
        </div>
        <div className="footer-column">
          <h4>Développé par</h4>
          <p>Rawen Sahraoui</p>
          <p>Takwa Abdellaoui</p>
        </div>
      </div>
      <div className="footer-bottom">
        <p>&copy; 2025 Tous droits réservés.</p>
      </div>
    </footer>
  );
};

export default Footer;
