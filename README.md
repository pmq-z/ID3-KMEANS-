# ID3 & K-Means

Two classic machine learning algorithms implemented in Python, each with a
terminal interface: a supervised classifier (ID3) and an unsupervised
clustering algorithm (k-means).

<!-- ============================================================= -->
```text

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⣶⣶⣶⠦⢄⣠⡟⢁⡔⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣾⣿⡀⠀⠀⠀⠀⢹⣿⣿⣄⣤⣶⡒⢦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⣀⠀⠹⣿⠿⣿⣦⡀⢀⡀⢛⣿⢿⣭⣿⣻⣿⣾⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢀⣤⠖⠋⠀⣀⣼⣧⣼⣧⠀⢈⣹⣿⣶⣷⣾⣿⠀⠈⠻⣿⣿⣿⠟⢰⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⣠⣀⣠⣾⠛⠁⣠⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣶⣦⠀⠹⡿⠷⠴⣿⣿⠸⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣾⣿⣿⣿⣷⣤⡴⠛⣹⣿⠿⣿⠟⠁⠀⠀⣸⣿⣿⣿⣿⣿⣾⣿⣿⣿⣦⣇⠀⠀⠘⣿⣦⠘⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠈⠉⢻⣿⣿⣿⣷⠞⠛⠁⠀⠀⠀⠀⠀⠀⠛⣿⣿⣿⣿⣿⣿⣯⣽⣿⣿⡏⠀⣿⣿⣿⣿⡎⢠⣤⣶⣦⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠰⣿⡋⠙⠉⠏⠀⠀⠀⠀⠀⠀⠐⠗⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⡟⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠙⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠈⣹⣿⡿⠻⣿⣿⣿⣿⠛⠟⠰⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠈⠃⠀⠀⠀⠀⠓⠀⠀⣰⠟⣽⠃⣹⠛⠉⠻⣶⣄⢹⠻⡿⣷⡀⠀⠀⠈⠙⠛⠛⠛⠛⠻⡿⠏⠈⠛⠙⠛⠿⢷⣶⣤⣄⡀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⠂⠀⠀⠀⠀⣠⣿⡟⣏⠀⡟⠀⠀⢀⡿⣿⣾⠀⣰⣿⣃⣶⠦⠤⢤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠙⠻⢷⣶⣤⣀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠒⠖⠚⣿⣿⣸⣿⠀⠛⠳⢶⠞⣠⣿⣏⣰⣿⣿⣛⠻⣷⣄⠀⠻⡙⢦⣄⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠛
⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠶⣤⣿⣿⣿⣿⣿⣼⣧⣤⢀⡿⢿⣿⣿⣿⣿⣿⣿⣷⣿⣿⣷⣦⠙⣦⣸⡷⣿⡆⠠⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢁⣿⣿⣿⣿⣿⡿⣿⣿⣿⡿⠃⠈⠙⠛⠛⠿⠛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣄⣾⡟⠀⠈⠓⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠟⢩⣿⣿⡇⠁⢠⣿⣿⡿⠀⠀⠀⠀⠀⢠⣤⣶⣾⣿⣿⣿⢿⣿⣿⣿⣿⣿⠁⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠏⠀⠈⢿⣿⣧⣴⣿⣿⣿⠇⠀⠀⡀⢀⣶⣼⣿⣿⣿⢿⣿⣿⣶⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣿⣿⣿⣿⠉⠀⠟⠀⣿⣿⣷⣿⣿⣿⣿⣿⣿⣿⠿⠟⣿⡿⠟⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⡇⠀⠀⠀⠀⣾⣿⣿⣿⣿⡟⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⡅⠀⠀⠀⠀⠈⠙⣿⣿⣿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⢹⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠐⠻⠿⠿⠿⠿⠿⠿⠿⠿⣷⣾⣿⣿⣿⣿⣿⣿⣿⣿⡿⠶⠶⠾⠿⠿⡶⠲⠗⠒⠒⠒⠶⠶⠒⠒⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

```
<!-- ============================================================= -->

## Contents
|              Folder              |       Algorithm      |     Type     |              Documentation                  |
|----------------------------------|----------------------|--------------|---------------------------------------------|
| [`AlgoritmoID3/`](AlgoritmoID3/) | ID3 (decision tree)  | Supervised   | [README_ID3.md](AlgoritmoID3/README_ID3.md) |
| [`Kmeans/`](Kmeans/)             | k-means (clustering) | Unsupervised | [README_kmeans.md](Kmeans/README_kmeans.md) |

- **ID3** learns a decision tree and rules from labeled data to classify new examples.
- **k-means** groups numeric data into `k` clusters by proximity, without labels.


## Requirements
```bash
pip install -r requirements.txt
```

Python 3.9+. Both read `.xlsx`, `.xls`, `.csv`, and `.tsv`.


## Quick start
```bash
python AlgoritmoID3/main.py     # ID3 decision tree
python Kmeans/kmeans.py         # k-means clustering
```

Each program is interactive. Usage and implementation details are in each
folder's README.


## Structure
```
.
├── README.md              general index
├── requirements.txt       dependencies
├── .gitignore
├── AlgoritmoID3/
│   ├── main.py
│   └── README_ID3.md
└── Kmeans/
    ├── kmeans.py
    └── README_kmeans.md
```
