# E12 — which filter finds the noise? (DT and NB, separately)

Generated 2026-10-02 17:21

## Detection — asymmetric noise

```
                     f1                                                                                precision                                                                              recall                                                                      
arm  DT confident_joint DT hard_vote NB confident_joint NB hard_vote committee consensus none DT confident_joint DT hard_vote NB confident_joint NB hard_vote committee consensus DT confident_joint DT hard_vote NB confident_joint NB hard_vote committee consensus none
rate                                                                                                                                                                                                                                                                      
0.2               0.537        0.537              0.801        0.755               0.644  0.0              0.419        0.419              0.822        0.666               0.881              0.759        0.759              0.791         0.88               0.511  0.0
```

## Detection — symmetric noise

```
                     f1                                                                                precision                                                                              recall                                                                      
arm  DT confident_joint DT hard_vote NB confident_joint NB hard_vote committee consensus none DT confident_joint DT hard_vote NB confident_joint NB hard_vote committee consensus DT confident_joint DT hard_vote NB confident_joint NB hard_vote committee consensus none
rate                                                                                                                                                                                                                                                                      
0.2                 0.6        0.599              0.836        0.786               0.769  0.0              0.466        0.465              0.823         0.67               0.913              0.846        0.846              0.855        0.955               0.671  0.0
```

## Accuracy delta vs no cleaning — DT (points)

```
arm              DT confident_joint  DT hard_vote  NB confident_joint  NB hard_vote  committee consensus  none
kind       rate                                                                                               
asymmetric 0.2                 7.33          7.33                9.33         10.67                 7.33   0.0
symmetric  0.2                 8.67          8.67                8.67         13.33                 9.33   0.0
```

## Accuracy delta vs no cleaning — LR (points)

```
arm              DT confident_joint  DT hard_vote  NB confident_joint  NB hard_vote  committee consensus  none
kind       rate                                                                                               
asymmetric 0.2                -0.67         -0.67               -3.33         -5.33                -0.67   0.0
symmetric  0.2                 6.00          6.00                8.00          6.67                 5.33   0.0
```

## Accuracy delta vs no cleaning — NB (points)

```
arm              DT confident_joint  DT hard_vote  NB confident_joint  NB hard_vote  committee consensus  none
kind       rate                                                                                               
asymmetric 0.2                 1.33          1.33                2.00          0.67                 1.33   0.0
symmetric  0.2                -0.67         -0.67                0.67          0.67                -0.67   0.0
```
