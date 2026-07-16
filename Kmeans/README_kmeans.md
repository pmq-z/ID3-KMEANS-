# =============
#    K-Means
# =============

Implementation of the k-means algorithm for unsupervised clustering of numerical data, with a terminal interface.

<!-- ============================================================= --> 
```text

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠚⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠈⠀⠐⠂⠄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡤⠞⠁⠀⠀⠀⠀⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠠⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠃⠀⠀⠀⠀⠀⠀⢁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡏⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠖⠂⡄⢀⣀⠀⢀⣸⠏⠀⠀⠀⠀⠀⠀⠠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠠⠀⢳⣦⠀⠀⠁⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⡉⠛⠛⠶⣄⠀⠀⠀⠀⠁⠢⠤⠀⠀⠘⢧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣆⠀⠈⣯⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣆⣤⣠⡄⣄⣹⣷⣤⣀⣀⣀⠄⢀⠀⠀⠀⢸⡁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡆⠀⢸⣯⢿⡄⠀⢰⠁⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠈⠙⠏⣁⣀⣼⠟⠿⣿⠿⣿⣿⣷⡶⢶⣿⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣄⠀⢻⡽⣿⡄⠀⠀⠀⠡⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠂⠀⠤⠼⠛⠁⢀⠀⠀⠀⢸⠊⠛⢠⡞⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣿⣦⡀⢻⣽⣿⡄⠀⠀⠄⠣⢤⣤⣀⡀⠠⢊⠜⠀⠄⢂⠀⠀⠀⢀⡠⠚⠀⣴⣾⣿⡇⠀⠘⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⣾⠿⠋⠉⠳⣄⠉⠻⠇⠀⡸⠀⠀⠀⠀⠉⠛⣇⠁⠀⠀⢀⡎⡷⠋⠈⠀⡀⠀⣠⠀⠀⠀⠰⡘⡄⢤⡶⢃⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣀⣀⠤⠖⠚⠋⠉⢠⠾⠋⠀⠀⠀⠀⠀⠈⢻⡦⠄⠠⡇⠀⠀⠀⠀⠀⠀⢸⠀⢀⡴⠋⠁⠈⣅⠠⠊⠀⢜⡌⠁⠀⣇⠀⠡⠰⣼⡧⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⢀⠀⠀⠄⠂⠉⠀⠀⠀⠀⠀⠀⠴⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⣧⠀⠀⠐⡄⠀⠀⠀⠀⠀⠰⢡⠏⠀⠀⡆⠈⠽⡄⡆⣄⡈⠀⠀⠀⠘⠂⡰⡇⠜⡀⡹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⡆⠀⠀⢰⠀⠀⠀⠀⠀⠀⠸⠈⠉⠹⠷⣆⣀⢸⡁⠆⠈⠉⢹⡀⡶⡸⠀⡏⣰⣀⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣷⠀⠀⢨⡀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⠀⠉⠙⠓⠻⠤⢔⠡⠴⠜⠓⠚⠋⠉⣠⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⡀⠀⠀⠀⠀⠀⡀⠀⠀⣀⠠⠤⢀⡀⢀⣴⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠤⠤⠠⠶⣤⣤⣤⣤⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⢷⣄⡀⠀⠠⡀⠀⣀⡀⠀⠀⠈⠀⠤⠠⠴⠋⠉⠁⠀⠀⠀⠀⠀⠀⠀⢹⣿⠿⣟⡿⣿⣶⣤⣄⡀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠛⠛⠶⢦⣤⣀⡀⠀⠀⠀⠀⠀⠀⠹⣿⣿⢶⣤⡈⠀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⡏⢀⣿⠻⠷⠭⣟⣿⣻⣶⣄⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠓⠶⣤⡀⠀⠀⠀⠹⣏⡇⠀⢽⠀⠀⠈⠛⢷⣄⡀⠀⠀⠀⠀⠀⠀⠀⣀⣠⡾⣞⣿⣿⠃⠈⣿⡀⠀⠀⠀⠀⠁⠛⠿⣷
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠳⢦⣄⣀⠙⣃⠀⢈⣷⣄⠀⠀⠀⠈⠿⠿⣶⣦⣄⣠⣶⣟⣭⣷⣽⣾⡿⠁⠈⢶⣿⠀⠀⠀⠀⠀⠀⠀⠀⠸
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠃⡻⡀⠀⠀⠙⢧⡀⠀⠀⠀⠀⠀⠈⠉⠛⠿⠿⢿⣯⣾⡝⠀⠀⠀⢸⡧⠀⠀⠀⠀⠀⠀⠀⠀⠠
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢨⡁⠹⣄⠀⠀⠈⢷⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠀⠀⣸⡗⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡅⠀⢻⣆⠀⠀⠀⠀⠉⠳⣤⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣟⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡟⡄⠀⢹⣦⠀⠀⠀⠀⠀⠀⠛⣶⣶⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢳⡄⠀⠉⠻⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣯⠀⠀⠀⠀⠀⠀⢀⣤⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡌⠳⣄⡀⠀⠀⠱⣴⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⡆⠀⠀⠀⣠⣾⣿⣻⠄
⢹⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⣆⠈⠻⣤⠀⠀⠈⠶⠤⢤⠀⠀⠀⠀⠀⠀⠠⠀⠀⠀⠀⠀⠀⣿⣧⡀⠀⣼⣿⣛⡶⣽⠀
⢸⢿⣷⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢳⣀⠘⢇⡀⢀⠀⠀⠠⠀⠀⠀⠀⠀⠀⠀⠡⠀⠀⠀⠀⢀⣿⣿⣷⣴⣿⣯⡽⣞⡃⠀
⣞⣻⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠹⣆⣀⠉⠫⠠⣀⠀⠀⠁⠐⠠⢀⡀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠿⣿⣿⣾⣿⣆

```
<!-- ============================================================= -->


## REQUIREMENTS
```bash
pip install pandas openpyxl
```

Python 3.9+. Supports `.xlsx`, `.xls`, `.csv`, `.tsv`.


## USAGE
```bash
python kmeans.py
```

The program is interactive and guides you through these steps:

   1. Provide the dataset path.
   2. Choose the row containing the headers.
   3. Select the numerical columns to be used as features.
   4. Define the number of clusters k (default is 2).
   5. Choose random centroids or enter them manually.
   6. The program normalizes, iterates, and displays the clusters with their centers.


## HOW DOES IT WORK?
k-means separates the data into k groups by minimizing the distance of each point to its center. The cycle is:
   
   1. Normalize each column by dividing it by its maximum (rule of three, maximum = 1).
   2. Initialize k centroids (random or given).
   3. Assign each point to the closest centroid using Euclidean distance: d(A,B)=∑(bi​−ai​)2​.
   4. Update each centroid as the average of the points in its cluster.
   5. Repeat steps 3–4 until the centroids stop moving (convergence).

In each iteration, the inertia (tau_va) is reported: the sum of the intra-cluster distances that the algorithm is reducing.


## NOTE
The result depends on the initial centroids; with random initialization, you can set a seed for reproducibility.
