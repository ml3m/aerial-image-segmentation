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

::

   python -m src.train -help
   python -m src.evaluate --checkpoint results/checkpoints/best.pth
   python -m src.visualize --checkpoint results/checkpoints/best.pth
