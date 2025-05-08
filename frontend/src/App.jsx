import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Footer from "./components/Footer/Footer";
import Home from "./pages/Home/Home";
import Analyze from "./pages/Analyze/Analyze";
import History from "./pages/History/History";
import Pathologies from "./pages/pathologies/pathologies";
import Fonctionnement from "./pages/fonctionnement/fonctionnement";

const App = () => {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/history" element={<History />} />
        <Route path="/Pathologies" element={<Pathologies />}/>
        <Route path="/fonctionnement" element={<Fonctionnement />}/>
      </Routes>
      <Footer />
    </Router>
  );
};

export default App;
