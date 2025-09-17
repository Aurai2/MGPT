# SteelsGPT – Predicting Steel Properties Project Deakin University

This is **SteelsGPT**, a Streamlit-powered web app that predicts the **mechanical properties of steels** (Ultimate Tensile Strength, Yield Strength, and Ductility) based on chemical composition and processing clusters.

This project combines several parts:
- **Phase 1**: Feature engineering and clustering of processing methods  
- **Phase 2**: Random Forest model trained on Phase-1 features + composition  
- **Phase 3**: Streamlit UI for interactive predictions and alloy suggestions  

There are several dependencies based on Phase 1 and Phase 2 that are found in our google drive folder and this app uses those models to then produce the following predictions.
---

## 📖 About Us
Adithya Umanath Rai, Kiran Devraju, Jose Thomson  
- Masters Project for Deakin University
- Guidance under Dr.Nick Birbilis/ Dr. Marzie Ghorbani 

---

## How It Works
1. **Upload/enter composition** (18 elements supported: C, Mn, Cr, Ni, Mo, etc.)  
2. **Select processing cluster (K=12)**  
3. App returns:
   - Predicted UTS, Yield, Ductility  
   - Closest matching alloys from existing dataset and closeness percentage
   - Optional: feature importance ranking for additional information

---
