Aerial Image Segmentation
=========================

.. |python| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/
   :alt: Python

.. |pytorch| image:: https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg
   :target: https://pytorch.org/
   :alt: PyTorch

|python| |pytorch|

U-Net semantic segmentation for aerial imagery (ISPRS Potsdam). Dataset: `deasadiqbal/private-data-1` via kagglehub.

Install
-------

::

   pip install -r requirements.txt

Usage
-----

Basic training and evaluation::

   python -m src.train
   python -m src.evaluate --checkpoint results/checkpoints/best.pth
   python -m src.visualize --checkpoint results/checkpoints/best.pth

K-fold cross-validation
~~~~~~~~~~~~~~~~~~~~~~~

Use ``--n-folds`` to train with K-fold cross-validation. Each fold trains a fresh model and saves its best checkpoint as ``best_fold_1.pth``, ``best_fold_2.pth``, etc. At the end, mean and standard deviation of validation loss across folds are printed::

   python -m src.train --n-folds 5
   python -m src.train --n-folds 3 --epochs 20

Evaluate or visualize any fold checkpoint::

   python -m src.evaluate --checkpoint results/checkpoints/best_fold_1.pth
   python -m src.visualize --checkpoint results/checkpoints/best_fold_1.pth
