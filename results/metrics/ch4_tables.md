# Chapter 4 -- Consolidated Results

```
TABLE 4.1 -- PREDICTIVE PERFORMANCE: CENTRALISED vs FEDERATED
==========================================================================================
Model               Setting           Accuracy            Recall              AUC-ROC        
------------------------------------------------------------------------------------------
random_forest       centralised       1.0000              1.0000              1.0000         
                    iid               0.9862 ± 0.0088   0.9900 ± 0.0100   0.9993
                    non_iid_equal     0.9925 ± 0.0083   0.9960 ± 0.0080   0.9997
                    non_iid           0.9925 ± 0.0100   0.9980 ± 0.0060   0.9994
------------------------------------------------------------------------------------------
xgboost             centralised       0.9875              0.9800              1.0000         
                    iid               0.9613 ± 0.0247   0.9620 ± 0.0316   0.9963
                    non_iid_equal     0.9663 ± 0.0210   0.9720 ± 0.0271   0.9963
                    non_iid           0.9763 ± 0.0142   0.9880 ± 0.0133   0.9981
------------------------------------------------------------------------------------------
logistic_reg        centralised       0.9875              0.9800              1.0000         
                    fedavg (iid)      0.9863 ± 0.0131   0.9780 ± 0.0209   1.0000
------------------------------------------------------------------------------------------

FEDERATION COST (centralised accuracy - federated accuracy):
  random_forest     iid             +0.0138
  random_forest     non_iid_equal   +0.0075
  random_forest     non_iid         +0.0075
  xgboost           iid             +0.0262
  xgboost           non_iid_equal   +0.0212
  xgboost           non_iid         +0.0112
  logistic_reg      fedavg          +0.0012
```

```

TABLE 4.2 -- DIFFERENTIAL PRIVACY: PRIVACY-UTILITY TRADE-OFF (TRACK A)
==========================================================================================
Model             Epsilon     Accuracy              Privacy Cost          
------------------------------------------------------------------------------------------
random_forest     inf (none)  0.9862 ± 0.0088   —                     
                  10.0        0.9512 ± 0.0205   +0.0350 ± 0.0255
                  5.0         0.7950 ± 0.0363   +0.1913 ± 0.0362
                  1.0         0.5363 ± 0.0385   +0.4500 ± 0.0395
                  0.5         0.5050 ± 0.0404   +0.4813 ± 0.0408
------------------------------------------------------------------------------------------
xgboost           inf (none)  0.9613 ± 0.0247   —                     
                  10.0        0.9400 ± 0.0200   +0.0212 ± 0.0177
                  5.0         0.7988 ± 0.0303   +0.1625 ± 0.0411
                  1.0         0.5413 ± 0.0451   +0.4200 ± 0.0388
                  0.5         0.5075 ± 0.0408   +0.4537 ± 0.0395
------------------------------------------------------------------------------------------
```

```

TABLE 4.3 -- HOMOMORPHIC ENCRYPTION: COMPUTATIONAL/COMMUNICATION OVERHEAD (TRACK B)
==============================================================================
  plaintext FedAvg-LR accuracy:  0.9750
  HE-aggregated FedAvg-LR accuracy: 0.9750
  accuracy difference:           +0.000000  (numerical only)
------------------------------------------------------------------------------
  encrypt time / round (all clients): 17.06 ms ± 0.55 ms
  aggregate time / round (server):    3.44 ms ± 0.34 ms
  decrypt time / round:                1.18 ms ± 0.09 ms
  total wall time, plaintext:          0.075 s
  total wall time, HE:                 0.542 s
  slowdown factor:                     7.2x
  ciphertext size / client update:     1,655,761 bytes
  plaintext size / client update:      1,000 bytes
  size blow-up factor:                 1655.8x
==============================================================================
```
