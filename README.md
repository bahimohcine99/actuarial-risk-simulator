# 🛡️ Actuarial Risk Simulator
### TP Assurance Non-Vie — Modélisation du Risque (Chapitre 6)

---

## 📋 Contenu de l'application

Cette interface Streamlit couvre **tous les concepts du TP** :

| Module | Contenu |
|--------|---------|
| 📊 Vue d'ensemble | KPIs, histogrammes, statistiques descriptives |
| ⚠️ VaR & TVaR | Calcul multi-niveaux, CDF, courbe de risque |
| 🔄 Modèles | Individuel vs Collectif, densités, convergence LGN |
| 📐 Panjer | Récursion, distribution agrégée, VaR Panjer |
| 🔒 Stop-Loss | Prime pure et chargée, courbe de réassurance |
| 📋 Rapport | Rapport actuariel automatique complet |

---

## 🚀 Installation et lancement

### Étape 1 — Installer Python (si pas déjà fait)
Télécharger Python 3.10+ sur : https://www.python.org/downloads/

### Étape 2 — Créer un environnement virtuel (recommandé)

```bash
# Ouvrir un terminal dans le dossier actuarial_simulator/
python -m venv venv

# Activer l'environnement (Windows)
venv\Scripts\activate

# Activer l'environnement (Mac/Linux)
source venv/bin/activate
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 — Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse :
**http://localhost:8501**

---

## 🎛️ Utilisation

1. **Sidebar gauche** : Configurer les paramètres du portefeuille
   - Distribution des sinistres (Log-Normale, Gamma, Exponentielle, Pareto)
   - Fréquence (Poisson ou Binomiale Négative)
   - Taille du portefeuille et nombre de simulations
   - Niveaux de confiance VaR/TVaR

2. **Onglets** : Explorer les résultats par thème

3. **Rapport final** : L'onglet "Rapport Actuariel" génère automatiquement 
   un rapport complet à copier dans votre TP.

---

## 📚 Théorie couverte

### Value-at-Risk
```
VaR_α(X) = F⁻¹(α)
```

### Tail Value-at-Risk
```
TVaR_α(X) = E[X | X > VaR_α(X)]
```

### Modèle Collectif
```
S = Σ_{i=1}^{N} X_i,  N ~ Poisson(λ),  X_i iid
```

### Récursion de Panjer (Poisson)
```
g(s) = (λ/s) · Σ_{k=1}^{s} k · h(k) · g(s-k),  g(0) = e^{-λ}
```

### Prime Stop-Loss
```
π(d) = E[(S - d)⁺] = E[max(S - d, 0)]
```

---

## 🏗️ Structure des fichiers

```
actuarial_simulator/
├── app.py              ← Application Streamlit principale
├── requirements.txt    ← Dépendances Python
└── README.md          ← Ce fichier
```

---

## ⚡ Astuce pour impressionner votre professeur

- Changer la loi de sévérité de **Log-Normale** à **Pareto** pour montrer 
  l'impact des queues lourdes sur la VaR et TVaR
- Augmenter λ pour voir la convergence du modèle collectif
- Comparer les primes Stop-Loss à différents niveaux de franchise
- Le rapport final (onglet 6) génère automatiquement toutes les interprétations actuarielles

---

*Développé pour le TP Assurance Non-Vie — Chapitre 6 : Modélisation de la Charge Sinistre*
