# E12v2 results

Reference labels are unverified; detection is against injected corruption only.
No novelty claim. Full thresholded Confident Learning is not implemented.
Primary metrics pool confusion matrices across outer folds per seed, using all dataset classes.

## Primary: dual agreement vs no cleaning at 20% added noise

                    delta_accuracy  delta_macro_f1
dataset                                           
breast-cancer               -0.583          -0.029
contact-lenses               0.694           1.522
diabetes                     1.693           1.837
glass                        0.000           1.670
image-segmentation           4.978           5.208
iris                         4.556           4.587
nsl-kdd                      1.038           1.855
soybean                      6.881           9.718
tic-tac-toe                 -1.009          -1.087
vote                         2.912           3.005

Descriptive run only: this configuration does not satisfy the locked main protocol.

## All downstream means

     kind  rate handling                 arm classifier  accuracy  macro_f1  delta_accuracy  delta_macro_f1
 pairflip   0.0   delete        DT hard_vote         DT    86.284    81.233           0.438           0.540
 pairflip   0.0   delete        DT hard_vote         LR    86.713    81.267           0.674           1.567
 pairflip   0.0   delete        DT hard_vote         NB    78.428    73.196           1.098           1.400
 pairflip   0.0   delete        DT threshold         DT    86.249    81.275           0.403           0.582
 pairflip   0.0   delete        DT threshold         LR    86.503    81.147           0.465           1.447
 pairflip   0.0   delete        DT threshold         NB    78.461    73.261           1.131           1.465
 pairflip   0.0   delete        NB hard_vote         DT    80.899    75.779          -4.947          -4.914
 pairflip   0.0   delete        NB hard_vote         LR    81.326    76.104          -4.712          -3.596
 pairflip   0.0   delete        NB hard_vote         NB    77.749    72.869           0.419           1.072
 pairflip   0.0   delete        NB threshold         DT    83.518    78.994          -2.327          -1.699
 pairflip   0.0   delete        NB threshold         LR    82.534    77.530          -3.504          -2.170
 pairflip   0.0   delete        NB threshold         NB    77.330    72.541           0.000           0.745
 pairflip   0.0   delete committee consensus         DT    87.011    81.896           1.165           1.202
 pairflip   0.0   delete committee consensus         LR    86.906    81.413           0.868           1.713
 pairflip   0.0   delete committee consensus         NB    78.007    73.111           0.677           1.315
 pairflip   0.0   delete      dual agreement         DT    86.214    81.395           0.368           0.702
 pairflip   0.0   delete      dual agreement         LR    86.523    81.128           0.485           1.428
 pairflip   0.0   delete      dual agreement         NB    77.196    72.482          -0.134           0.686
 pairflip   0.0   delete                none         DT    85.846    80.693           0.000           0.000
 pairflip   0.0   delete                none         LR    86.038    79.700           0.000           0.000
 pairflip   0.0   delete                none         NB    77.330    71.796           0.000           0.000
 pairflip   0.0   delete    reference labels         DT    85.846    80.693           0.000           0.000
 pairflip   0.0   delete    reference labels         LR    86.038    79.700           0.000           0.000
 pairflip   0.0   delete    reference labels         NB    77.330    71.796           0.000           0.000
 pairflip   0.2   delete        DT hard_vote         DT    77.195    70.857           6.243           4.728
 pairflip   0.2   delete        DT hard_vote         LR    82.356    76.293           0.088           0.702
 pairflip   0.2   delete        DT hard_vote         NB    74.349    69.024           3.932           4.196
 pairflip   0.2   delete        DT threshold         DT    76.938    70.837           5.987           4.708
 pairflip   0.2   delete        DT threshold         LR    82.248    76.171          -0.019           0.580
 pairflip   0.2   delete        DT threshold         NB    74.457    69.177           4.040           4.349
 pairflip   0.2   delete        NB hard_vote         DT    76.142    70.693           5.190           4.564
 pairflip   0.2   delete        NB hard_vote         LR    76.535    71.007          -5.732          -4.584
 pairflip   0.2   delete        NB hard_vote         NB    72.844    67.840           2.427           3.012
 pairflip   0.2   delete        NB threshold         DT    77.409    72.592           6.458           6.463
 pairflip   0.2   delete        NB threshold         LR    77.871    72.567          -4.397          -3.024
 pairflip   0.2   delete        NB threshold         NB    72.610    67.994           2.193           3.166
 pairflip   0.2   delete committee consensus         DT    79.062    73.527           8.111           7.398
 pairflip   0.2   delete committee consensus         LR    82.217    76.399          -0.051           0.808
 pairflip   0.2   delete committee consensus         NB    72.598    68.380           2.182           3.553
 pairflip   0.2   delete      dual agreement         DT    77.165    72.014           6.213           5.885
 pairflip   0.2   delete      dual agreement         LR    81.571    76.201          -0.697           0.610
 pairflip   0.2   delete      dual agreement         NB    71.529    67.374           1.112           2.546
 pairflip   0.2   delete                none         DT    70.951    66.129           0.000           0.000
 pairflip   0.2   delete                none         LR    82.268    75.591           0.000           0.000
 pairflip   0.2   delete                none         NB    70.417    64.828           0.000           0.000
 pairflip   0.2   delete    reference labels         DT    85.846    80.693          14.894          14.564
 pairflip   0.2   delete    reference labels         LR    86.038    79.700           3.771           4.109
 pairflip   0.2   delete    reference labels         NB    77.330    71.796           6.913           6.969
symmetric   0.0   delete        DT hard_vote         DT    86.284    81.233           0.438           0.540
symmetric   0.0   delete        DT hard_vote         LR    86.713    81.267           0.674           1.567
symmetric   0.0   delete        DT hard_vote         NB    78.428    73.196           1.098           1.400
symmetric   0.0   delete        DT threshold         DT    86.249    81.275           0.403           0.582
symmetric   0.0   delete        DT threshold         LR    86.503    81.147           0.465           1.447
symmetric   0.0   delete        DT threshold         NB    78.461    73.261           1.131           1.465
symmetric   0.0   delete        NB hard_vote         DT    80.899    75.779          -4.947          -4.914
symmetric   0.0   delete        NB hard_vote         LR    81.326    76.104          -4.712          -3.596
symmetric   0.0   delete        NB hard_vote         NB    77.749    72.869           0.419           1.072
symmetric   0.0   delete        NB threshold         DT    83.518    78.994          -2.327          -1.699
symmetric   0.0   delete        NB threshold         LR    82.534    77.530          -3.504          -2.170
symmetric   0.0   delete        NB threshold         NB    77.330    72.541           0.000           0.745
symmetric   0.0   delete committee consensus         DT    87.011    81.896           1.165           1.202
symmetric   0.0   delete committee consensus         LR    86.906    81.413           0.868           1.713
symmetric   0.0   delete committee consensus         NB    78.007    73.111           0.677           1.315
symmetric   0.0   delete      dual agreement         DT    86.214    81.395           0.368           0.702
symmetric   0.0   delete      dual agreement         LR    86.523    81.128           0.485           1.428
symmetric   0.0   delete      dual agreement         NB    77.196    72.482          -0.134           0.686
symmetric   0.0   delete                none         DT    85.846    80.693           0.000           0.000
symmetric   0.0   delete                none         LR    86.038    79.700           0.000           0.000
symmetric   0.0   delete                none         NB    77.330    71.796           0.000           0.000
symmetric   0.0   delete    reference labels         DT    85.846    80.693           0.000           0.000
symmetric   0.0   delete    reference labels         LR    86.038    79.700           0.000           0.000
symmetric   0.0   delete    reference labels         NB    77.330    71.796           0.000           0.000
symmetric   0.2   delete        DT hard_vote         DT    79.371    72.348           9.693           9.092
symmetric   0.2   delete        DT hard_vote         LR    82.765    76.298           0.216           0.722
symmetric   0.2   delete        DT hard_vote         NB    72.980    67.618          -0.490           0.336
symmetric   0.2   delete        DT threshold         DT    79.319    72.530           9.641           9.273
symmetric   0.2   delete        DT threshold         LR    82.765    76.330           0.215           0.754
symmetric   0.2   delete        DT threshold         NB    73.157    67.841          -0.314           0.559
symmetric   0.2   delete        NB hard_vote         DT    75.417    69.770           5.740           6.514
symmetric   0.2   delete        NB hard_vote         LR    77.292    70.300          -5.257          -5.277
symmetric   0.2   delete        NB hard_vote         NB    72.635    67.685          -0.835           0.403
symmetric   0.2   delete        NB threshold         DT    76.399    69.991           6.722           6.735
symmetric   0.2   delete        NB threshold         LR    79.948    73.555          -2.601          -2.021
symmetric   0.2   delete        NB threshold         NB    72.068    67.552          -1.402           0.270
symmetric   0.2   delete committee consensus         DT    80.310    73.781          10.632          10.525
symmetric   0.2   delete committee consensus         LR    84.244    78.055           1.695           2.478
symmetric   0.2   delete committee consensus         NB    71.862    68.257          -1.608           0.975
symmetric   0.2   delete      dual agreement         DT    76.842    69.885           7.164           6.629
symmetric   0.2   delete      dual agreement         LR    83.476    76.991           0.926           1.415
symmetric   0.2   delete      dual agreement         NB    71.448    67.169          -2.023          -0.113
symmetric   0.2   delete                none         DT    69.678    63.256           0.000           0.000
symmetric   0.2   delete                none         LR    82.549    75.576           0.000           0.000
symmetric   0.2   delete                none         NB    73.471    67.282           0.000           0.000
symmetric   0.2   delete    reference labels         DT    85.846    80.693          16.168          17.437
symmetric   0.2   delete    reference labels         LR    86.038    79.700           3.489           4.124
symmetric   0.2   delete    reference labels         NB    77.330    71.796           3.859           4.514