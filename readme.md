# Quantum Enhanced Tumor Analysis Framework (QETAF)

**QETAF** is an advanced hybrid quantum-classical framework designed for automated **brain tumor classification** and **temporal tumor progression simulation** using the **BraTS dataset**.  
The system integrates **Convolutional Neural Networks (CNNs)**, **Quantum Convolutional Neural Networks (QCNNs)**, **Recurrent Quantum Neural Networks (RQNNs)**, and **Large Language Models (LLMs)** (GPT-based) to predict, simulate, and interpret tumor evolution across time.

Results of this research are published in the Journal of Electronics Electromedical Engineering and Medical Informatics (JEEEMI), visit https://jeeemi.org/index.php/jeeemi/article/view/720/246 for more information.

---

## 🧠 Overview

QETAF bridges traditional deep learning with quantum computing by embedding **quantum-enhanced feature learning** into a medically interpretable prediction system.  
It classifies MRI brain scans into four tumor types and projects **possible tumor growth patterns** over time.

### **Tumor Categories**
- **Meningioma**
- **Pituitary**
- **Glioma**
- **No Tumor**

---

## ⚙️ Architecture Summary

The architecture comprises **four main stages**:

1. **Classical Tumor Detection**
2. **Quantum Feature Projection (QCNN)**
3. **Temporal Simulation (RQNN)**
4. **Natural Language Interpretation (GPT Analysis)**

Each module is designed to work both independently and as a continuous data pipeline.

---

## 🩻 1. Classical Tumor Detection

**Input:** MRI slices from the **BraTS dataset**  
**Model:** Pretrained CNN classifier with 4 output categories  
**Output:** Detected tumor type with prediction confidence  

### Key Details:
- CNN backbone trained on labeled MRI images.
- Performs coarse-level feature extraction.
- Serves as the **trigger** for quantum feature analysis.

---

## ⚛️ 2. Quantum Convolutional Neural Network (QCNN)

**Purpose:** To extract *quantum-enhanced spatial features* of the tumor from the MRI image.  
**Implementation:** Built using `Pennylane` with **8 qubits** and **3 variational layers**.

### Circuit Composition:
- **RY Encoding:** Maps classical pixel intensities into qubit states.
- **RX Variational Layers:** Apply parameterized rotations for nonlinear transformations.
- **CNOT Entanglement:** Distributes information across qubits in a cyclic ring topology.
- **Z-Measurement:** Produces a feature vector representing the encoded spatial features.

### Output:
A dense **quantum feature vector** capturing high-order spatial correlations of the tumor region.

---

## 🔁 3. Recurrent Quantum Neural Network (RQNN)

**Purpose:** Simulate and model **temporal tumor progression** — predicting how a tumor might evolve.  
**Design:** Quantum equivalent of an RNN cell with memory state.

### Circuit Flow:
1. **Input Encoding:**  
   - Combines current quantum feature vector with previous hidden state via RY rotations.
2. **Variational Layer:**  
   - Applies RZ phase rotations and CNOT entanglement to model temporal evolution.
3. **Measurement:**  
   - Z-basis readout produces a new quantum hidden state for the next timestep.

### Iterative Simulation:
- Repeated for several time steps (`num_iterations`), generating **future states**.
- Each future MRI state is re-evaluated for tumor existence.

---

## 🤖 4. GPT-Based Medical Interpretation

**Purpose:** To provide a **contextual medical explanation** of tumor progression across simulated stages.

### Workflow:
1. Quantum outputs from RQNN (expectation values) are converted into textual descriptors.
2. These descriptors are passed to **GPT-3** (or newer LLMs) with clinical prompts.
3. The model generates **human-readable diagnostic summaries**, describing:
   - Tumor growth pattern (shrinking/spreading)
   - Potential treatment relevance
   - Severity progression

---

## 🔄 Full Pipeline Flow

```mermaid
flowchart TD
    A["Brain MRI Image (BraTS Dataset)"] --> B["CNN-Based Tumor Classifier"]
    B -->|Detected Tumor| C["Quantum Convolutional Neural Network (QCNN)"]
    C -->|Quantum Feature Map| D["Recurrent Quantum Neural Network (RQNN)"]
    D -->|Temporal State Evolution| E["GPT-based Clinical Report Generator"]
    E --> F["Final Patient Evolution Report"]

