# Intelligence Artificielle 2025, projet semestriel
## Sujet 4 --- Apprentissage de représentations par reconstruction

This project is a practical assignment for the course *Intelligence Artificielle* taught in Spring 2025 at the University of Strasbourg. We chose Topic 4, the full description of which can be found in this repository (see the corresponding file).

---

## ✍️ Authors

- **Aleksandr Akhmetshin**
- **Luka Beklemishev**
- **Noah Jamart**

Groupe --- L2 MPA

---

## 📝 Project Description

In this project, we implemented a neural network designed to compress and then reconstruct data (an autoencoder).

We trained three models with different compression sizes (4, 8, and 12) and compared their reconstruction quality using the mean-squared-error metric.

The training and evaluation were performed using the iris dataset, available in the `iris_extended.csv` file.

---

## 📁 Project Structure

```
├── README.md # This file 
├── 4.RepresentationsReconstruction.pdf # File with task description
├── main.py # Main script, entry point of the project
├── NeuralNet.py # Contains the neural network class implementation
├── Utility.py # Helper functions
├── items_a_evalue.ipynb # Jupyter Notebook implementing the required tasks
├── iris_extended.csv # Dataset used for training and evaluation
├── .gitignore # Specifies untracked files
└── LICENSE # License file 
```
