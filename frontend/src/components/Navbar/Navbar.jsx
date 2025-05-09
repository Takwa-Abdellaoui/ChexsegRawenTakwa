import React from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">CheXSeg</div>
      <ul className="navbar-links">
        <li><Link to="/">Accueil</Link></li>
        <li><Link to="/analyze">Analyser</Link></li>
        <li><Link to="/history">Histoire</Link></li>
        <li><Link to="/pathologies">Pathologies</Link></li>
        <li><Link to="/fonctionnement">Fonctionnement</Link></li>
        
      </ul>
    </nav>
  );
};

export default Navbar;
