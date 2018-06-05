# MLG-LSC
Metric learning-guided least squares classifier for multicategory classification
============================MLG-LSC Matlab Code============================

This package contains the source code for the following paper:

    Chuanxing Geng and Songcan Chen.  Metric Learning-Guided Least Squares Classifier Learning. TNNLS 2018.
______________________________________________________________________________________
---- * Manual of MLG-LSC * ----------------------------------------------------------------------

   input varables:
   
       Parameters: lamda, mu, and alpha  correspond the formula (2), (10), (12) in our MLG-LSC paper
       
       Training data: X (n*d) is stored by rows and divided sequentially class by class
       
       Training label: Y (n*c) is coded by zero-one vector corresponding the training data

   output varables:
   
       W: coefficient matrix of LSR
       
       t: bias vector of LSR
       
       A_alpha: dragging (or metric) matrix for LSR error

_________________________________________________________________________________________
----- * PACKING LIST * ------------------------------------------------------------------

1. A demo script named 'MLG_LSC_Iris.m' is provided. It runs MLG-LSC on the Iris dataset. 
2. 'CS_GeometricMean.m' is used for fast computation of Riemannian geodesics for SPD matrices implemented by  Cholesky-Schur method. 
3. 'Iris.mat' is the preprocessed dataset for demo.
4. 'README.md' file.
________________________________________________________________________________________
---- * Attention* -----------------------------------------------------------------------

This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Songcan Chen (s.chen@nuaa.edu.cn).

This package was developed by Chuanxing Geng. For any problem concerning the codes, please feel free to contact Mr. Geng (gengchuanxing@126.com or gengchuanxing@nuaa.edu.cn).
