# HorizonSAGE: Exploring an Adaptive Greedy Approach to Skyline Detection using Neural Networks
**Horizon S*tructure Tensor* A*daptive* G*reedy* E*xtraction***
<!-- SPDX-License-Identifier: MIT -->
---
## Abstract
In mountain peak identification applications, synthetic mountain line overlays are often misaligned with live camera feeds, leading to poor user experiences. We developed a ridge detection algorithm that uses a new adaptive greedy reconstruction strategy to accurately and efficiently detect mountain ridge lines. Our approach trains a single-hidden-layer neural network on local pixel neighborhoods. Additionally, we compared adding structure tensor values to our training features and comparing training on grayscale versus RGB pixels. Our proposed greedy algorithm reconstructs continuous ridgelines by searching for high probability pixels within a small search radius in each column—essentially tracing the line from left to right. This approach avoids spending time classifying unnecessary pixels and removes the need for computationally intensive dynamic programming. Experiments on our limited dataset revealed that our method runs much faster than dynamic programming, only suffering a minimal reduction in accuracy.

---
## How to Use    
These instructions assume you’re using Python 3.7+ and have `git`, `pip`, and Jupyter or VS Code installed.
### 1. Clone the repo
```bash
git clone https://github.com/csci701s25/snowbowl-savant.git 
cd snowbowl-savant
```
### 2. Create a virtual environment
We recommend naming it `.venv` and adding `.venv/` to your `.gitignore`.
```bash
python3 -m venv .venv
```
### 3. Activate the virtual environment
- **macOS/Linux (bash/zsh):**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Windows (CMD.exe):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
Once activated, your prompt should show `(.venv)`.
### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 5. Launch the notebook
To replicate our results:
- Open `final_tests.ipynb` directly in **VS Code** or **Jupyter Lab**. (***Recommended***)
- Or from terminal:
  ```bash
  jupyter notebook final_tests.ipynb
  ```
Make sure your environment is selected as the kernel when running the notebook.
To train and test on images of your own, simply change the directory routes listed at the top of the notebook.
### 6. Deactivate when you’re done
```bash
deactivate
```
**Pro tips:**
- If `jupyter` isn’t found, install it:
  ```bash
  pip install notebook
  ```
- Be sure to **not commit `.venv/`** by listing it in your `.gitignore`.

---
## Examples
If you want to run the program on test images of your own, navigate to `EdgeDetection/user_images.ipynb` and follow the instructions in the file! If you want to replicate our results, run all the cells in `EdgeDetection/final_tets.ipynb`.

> Complete "How to Use" steps before you run the files!

This is an example of a generated line that was not a part of our training or testing suite, taken nearby Middlebury College:
![Middlebury Mountain Range with Line](/media/midd_line.png)
This image shows a very accurate horizon line. This was computed training on color images with structure tensor data.

This is an example of an image that was in our training dataset, that performed poorly:
![Decent line with poor detection in a region](/media/image3_line.png)

---
## File Organization
The most important file is `EdgeDetection/final_tests.ipynb`. Open the notebook and run all the cells to replicate our results.

snowbowl-savant/

├─ EdgeDetection/  
│  ├─ **edge_detection.ipynb**        ← Initial edge‐detection experiments  
│  ├─ **edge_det_tensor.ipynb**       ← Exploring structure‐tensor features  
│  ├─ **final_tests.ipynb**           ← Final evaluation & plots  
│  ├─ data-figures/                   ← Old generated plots & raw results  
│  │   ├─ `eval.png`  
│  │   ├─ `web_dataset_evaluation.png`  
│  │   ├─ `results.txt`  
│  │   └─ `web_dataset_results.csv`  
│  ├─ models/                         ← Saved PyTorch model checkpoints (`.pth`)  
│  │   ├─ `ridge_orig.pth`  
│  │   ├─ `ridge_st_cont.pth`  
│  │   └─ …  
│  ├─ test_images/                    ← Sample input images for demo & testing  
│  │   ├─ `canyon.jpg`  
│  │   └─ …  
│  ├─ utils/                          ← Python modules  
│  │   ├─ **data.py**                 ← Dataset prep & I/O  
│  │   ├─ **features.py**             ← Feature extraction routines  
│  │   ├─ **training.py**             ← Model training / evaluation loops  
│  │   ├─ **ridge_evaluation.py**     ← Post-training metrics & analysis  
│  │   └─ **line_prediction.py**      ← Greedy / DP skyline extraction  
│  └─ web_dataset/                    ← Raw web-scraped train/test splits  
│      ├─ train/  
│      └─ test/  
├─ requirements.txt                   ← Project dependencies   
├─ report/                            ← Final written report  
│   ├─ Final Poster.pdf               ← Project Poster  
│   └─ README.md                      ← Project report  
└─ README.md                          ← This file   

Each notebook in `EdgeDetection/` can be run in Jupyter Lab or VS Code. The `utils/` folder houses all reusable code for loading data, extracting features (incl. structure‐tensor maps), training models, and evaluating results. Pretrained checkpoints live in `models/` (*these were trained without a train-test split*), sample inputs in `test_images/`, and final output plots & CSVs in `data-figures/`.

---
## Acknowledgements
- Professor **Philip Caplan** - Middlebury College Dept. of Computer Science
- Professor **Andrea Vaccari** - Middlebury College Dept. of Computer Science

> We employed ChatGPT as a tool to help format, proofread, and improve the readability of this text