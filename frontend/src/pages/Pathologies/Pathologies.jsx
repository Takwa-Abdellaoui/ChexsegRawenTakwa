import React from 'react';
import './Pathologies.css';

const pathologies = [
  {
    name: 'Atelectasis',
    description: 'Affaissement partiel ou complet d’un ou plusieurs lobes du poumon.',
    symptoms: 'Essoufflement, toux, douleurs thoraciques.',
    link: 'https://www.mayoclinic.org/diseases-conditions/atelectasis/symptoms-causes/syc-20369684'
  },
  {
    name: 'Cardiomegaly',
    description: 'Augmentation anormale de la taille du cœur.',
    symptoms: 'Fatigue, essoufflement, palpitations.',
    link: 'https://my.clevelandclinic.org/health/diseases/21490-enlarged-heart-cardiomegaly'
  },
  {
    name: 'Effusion',
    description: 'Présence de liquide dans la cavité pleurale entourant les poumons.',
    symptoms: 'Essoufflement, toux sèche, douleur à la poitrine.',
    link: 'https://my.clevelandclinic.org/health/diseases/17373-pleural-effusion'
  },
  {
    name: 'Infiltration',
    description: 'Accumulation de substances ou de cellules dans le tissu pulmonaire.',
    symptoms: 'Toux, fièvre, douleurs thoraciques.',
    link: 'https://accessemergencymedicine.mhmedical.com/content.aspx?sectionid=109429397&bookid=1658'
  },
  {
    name: 'Mass',
    description: 'Formation anormale (souvent une tumeur) visible dans les poumons.',
    symptoms: 'Toux persistante, perte de poids, hémoptysie.',
    link: 'https://www.verywellhealth.com/lung-mass-possible-causes-and-what-to-expect-2249388'
  },
  {
    name: 'Nodule',
    description: 'Petite masse ronde dans le poumon, souvent bénigne.',
    symptoms: 'Souvent asymptomatique, découverte fortuite.',
    link: 'https://www.chu-lyon.fr/nodule-pulmonaire'
  },
  {
    name: 'Pneumonia',
    description: 'Infection provoquant une inflammation des alvéoles pulmonaires.',
    symptoms: 'Fièvre, toux, douleur thoracique, difficultés respiratoires.',
    link: 'https://my.clevelandclinic.org/health/diseases/4471-pneumonia'
  },
  {
    name: 'Pneumothorax',
    description: 'Présence d’air dans la cavité pleurale entraînant un affaissement pulmonaire.',
    symptoms: 'Douleur thoracique soudaine, essoufflement.',
    link: 'https://www.msdmanuals.com/fr/accueil/troubles-pulmonaires-et-des-voies-a%C3%A9riennes/maladies-de-la-pl%C3%A8vre-et-du-m%C3%A9diastin/pneumothorax'
  },
  {
    name: 'Consolidation',
    description: 'Durcissement du tissu pulmonaire dû à une accumulation de liquide ou cellules.',
    symptoms: 'Toux, fièvre, respiration sifflante.',
    link: 'https://radiopaedia.org/articles/lung-consolidation'
  },
  {
    name: 'Edema',
    description: 'Accumulation de liquide dans les poumons.',
    symptoms: 'Essoufflement, toux, respiration difficile.',
    link: 'https://www.healthline.com/health/pulmonary-edema'
  },
  {
    name: 'Emphysema',
    description: 'Maladie pulmonaire chronique qui détruit les alvéoles.',
    symptoms: 'Essoufflement progressif, toux chronique.',
    link: 'https://www.lung.org/lung-health-diseases/lung-disease-lookup/emphysema'
  },
  {
    name: 'Fibrosis',
    description: 'Formation de tissu cicatriciel dans les poumons.',
    symptoms: 'Essoufflement, toux sèche persistante.',
    link: 'https://www.pulmonaryfibrosis.org/'
  },
  {
    name: 'Pleural_Thickening',
    description: 'Épaississement anormal de la plèvre.',
    symptoms: 'Essoufflement, douleur à la poitrine.',
    link: 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7121719/'
  },
  {
    name: 'Hernia',
    description: 'Protrusion d’un organe à travers une ouverture anormale (ex. : hernie hiatale).',
    symptoms: 'Reflux, douleurs thoraciques.',
    link: 'https://www.mayoclinic.org/diseases-conditions/hiatal-hernia'
  }
];

export default function Pathologies() {
  return (
    <div className="pathologies-container">
      <h1>Fiches Pathologies</h1>
      <p>
        Voici les 14 pathologies détectées par notre modèle. Cliquez sur les liens pour en savoir plus via des sources médicales fiables.
      </p>
      <div className="pathology-grid">
        {pathologies.map((item, idx) => (
          <div key={idx} className="pathology-card">
            <h2>{item.name}</h2>
            <p><strong>Description :</strong> {item.description}</p>
            <p><strong>Symptômes :</strong> {item.symptoms}</p>
            <a href={item.link} target="_blank" rel="noopener noreferrer">En savoir plus 🔗</a>
          </div>
        ))}
      </div>
    </div>
  );
}
