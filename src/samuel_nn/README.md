# samuel_nn: dense neural network baseline

[dense_nn.ipynb](dense_nn.ipynb) trains a fully connected network (200 → 128 → 64 → 4) on `src/data/{train,val,test}.npz`. It uses square-root class weights and early stopping on validation macro-F1. The notebook saves the model, its training history and its predictions to `outputs/`.

The predictions are used by the model comparison in `src/EDA.ipynb` (section 7), so run this notebook first. It takes under a minute on a CPU.
