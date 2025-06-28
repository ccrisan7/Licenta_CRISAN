# PREDICȚIA CONSTANTEI CHIMICE PKA FOLOSIND O REȚEA NEURONALĂ ȘI DETERMINAREA AUTOMATIZATĂ A ACESTEIA

Nume: Crișan
Prenume: Cristian
Link GitHub: https://github.com/ccrisan7/Licenta_CRISAN

Titlu: PREDICȚIA CONSTANTEI CHIMICE PKA FOLOSIND O REȚEA NEURONALĂ ȘI DETERMINAREA AUTOMATIZATĂ A ACESTEIA

Spec: CTIRO

Aplicatie 1: Aplicație Predictor pKa – Licență

Această aplicație permite:
- vizualizarea structurii și grafului unei molecule pornind de la SMILES;
- predicția valorii pKa pentru compuși organici folosind un model antrenat cu rețea neuronală de tip GNN (Graph Neural Network);
- analiza setului de date și vizualizarea statisticilor de performanță ale modelului.

Livrabile:

1. `predict_pKa1.ipynb` – codul folosit pentru antrenarea și validarea modelului;
2. `predict_pKa1_app.py` – aplicația Streamlit pentru predicții interactive;
3. `model.pth` – modelul neuronal salvat;
4. `dataset_IUPAC.csv` – setul de date brut;
5. `used_dataset.csv` – setul de date filtrat;
6. `output2.png`, `output3.png`, `output4.png` – grafice de performanță ale modelului.

Instalare:

1. Instalează Python 3.8+ și pip.
2. (Opțional) Creează un mediu virtual:

   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate   # Windows

3. Instalează dependințele:

   pip install streamlit torch torchvision torchaudio torch-geometric matplotlib rdkit pandas networkx requests

Lansare aplicație:

1. Asigură-te că fișierul `model.pth` este în același folder cu `predict_pKa1_app.py`.
2. Rulează aplicația:

   streamlit run predict_pKa1_app.py

3. Aplicația va fi disponibilă la adresa:

   http://localhost:8501



Aplicație 2: Titrare automatizată

Această aplicație integrează un sistem fizic (cu Arduino și pompă peristaltică) și o interfață grafică realizată cu Tkinter pentru a efectua:
- calibrarea pH-metrului pe baza a 3 soluții tampon (pH 4.01, 6.86, 9.21);
- titrarea automată a unei substanțe necunoscute și înregistrarea valorilor de pH în timp real;
- analiza numerică (cu spline-uri) pentru determinarea punctului de inflexiune;
- generarea automată a unui raport PDF cu grafice și rezultatul valorii pKa.

Livrabile:

1. `cod_aplicatie_20martie.py` – aplicația principală pentru rularea experimentului (Python GUI + generare PDF);
2. `sketch_mar31a.ino` – codul pentru Arduino, ce controlează pompa și trimite datele de la senzorul de pH;
3. `calibrare.png`, `titrare.png` – grafice salvate automat;
4. `Titrare_*.pdf` – raportul generat automat la finalul titrării cu toate datele experimentale.

Instalare:

1. Instalează Python 3.8+ și următoarele librării:

pip install numpy matplotlib tkinter fpdf pyserial scipy scikit-learn

2. Încarcă fișierul `sketch_mar31a.ino` pe Arduino din Arduino IDE.

3. Verifică portul COM la care este conectat Arduino (înlocuiește `'COM3'` în cod dacă este necesar).

Lansare aplicație:

1. Conectează Arduino-ul și pornește senzorul de pH și pompa.
2. Rulează aplicația:

python cod_aplicatie_20martie.py

3. Introdu substanțele și pornește procesul. Aplicația te va ghida prin toți pașii (calibrare, titrare, analiză).

Raport generat:

La final, un fișier PDF se va salva automat și va conține:
- valorile de calibrare și titrare,
- curbele de calibrare și titrare,
- valoarea pKa estimată numeric,
- detalii despre substanțele utilizate și momentul experimentului.
