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
==============================================================================
Model             Epsilon     Accuracy      Privacy Cost    
------------------------------------------------------------------------------
random_forest     inf (none)  1.0000        —               
                  10.0        0.9250        +0.0750         
                  5.0         0.8000        +0.2000         
                  1.0         0.4875        +0.5125         
                  0.5         0.4625        +0.5375         
------------------------------------------------------------------------------
xgboost           inf (none)  0.9000        —               
                  10.0        0.9000        +0.0000         
                  5.0         0.8000        +0.1000         
                  1.0         0.4625        +0.4375         
                  0.5         0.4500        +0.4500         
------------------------------------------------------------------------------
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
