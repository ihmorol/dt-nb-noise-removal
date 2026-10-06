# E12v2 results

Reference labels are unverified; detection is against injected corruption only.
No novelty claim. Full thresholded Confident Learning is not implemented.
Primary metrics pool confusion matrices across outer folds per seed, using all dataset classes.

## Primary: dual agreement vs no cleaning at 20% added noise

          delta_accuracy  delta_macro_f1
dataset                                 
diabetes          -1.562          -0.676
iris               3.333           3.432
vote               1.379           1.545

Descriptive run only: this configuration does not satisfy the locked main protocol.

## All downstream means

     kind  rate handling                 arm classifier  accuracy  macro_f1  delta_accuracy  delta_macro_f1
 pairflip   0.0  relabel        DT hard_vote        MLP    87.194    85.940           0.097           0.138
 pairflip   0.0  relabel        DT threshold        MLP    87.194    85.940           0.097           0.138
 pairflip   0.0  relabel        NB hard_vote        MLP    85.322    84.188          -1.775          -1.614
 pairflip   0.0  relabel        NB threshold        MLP    85.424    84.516          -1.673          -1.287
 pairflip   0.0  relabel committee consensus        MLP    87.324    86.131           0.227           0.328
 pairflip   0.0  relabel      dual agreement        MLP    86.765    85.641          -0.332          -0.162
 pairflip   0.0  relabel                none        MLP    87.097    85.803           0.000           0.000
 pairflip   0.0  relabel    reference labels        MLP    87.097    85.803           0.000           0.000
 pairflip   0.2  relabel        DT hard_vote        MLP    84.678    83.223           0.657           0.578
 pairflip   0.2  relabel        DT threshold        MLP    83.559    82.039          -0.462          -0.606
 pairflip   0.2  relabel        NB hard_vote        MLP    81.674    80.222          -2.347          -2.423
 pairflip   0.2  relabel        NB threshold        MLP    82.694    81.805          -1.328          -0.839
 pairflip   0.2  relabel committee consensus        MLP    84.228    82.911           0.207           0.266
 pairflip   0.2  relabel      dual agreement        MLP    84.849    83.865           0.828           1.221
 pairflip   0.2  relabel                none        MLP    84.021    82.645           0.000           0.000
 pairflip   0.2  relabel    reference labels        MLP    87.097    85.803           3.075           3.158
symmetric   0.0  relabel        DT hard_vote        MLP    87.194    85.940           0.097           0.138
symmetric   0.0  relabel        DT threshold        MLP    87.194    85.940           0.097           0.138
symmetric   0.0  relabel        NB hard_vote        MLP    85.322    84.188          -1.775          -1.614
symmetric   0.0  relabel        NB threshold        MLP    85.424    84.516          -1.673          -1.287
symmetric   0.0  relabel committee consensus        MLP    87.324    86.131           0.227           0.328
symmetric   0.0  relabel      dual agreement        MLP    86.765    85.641          -0.332          -0.162
symmetric   0.0  relabel                none        MLP    87.097    85.803           0.000           0.000
symmetric   0.0  relabel    reference labels        MLP    87.097    85.803           0.000           0.000
symmetric   0.2  relabel        DT hard_vote        MLP    84.234    82.781          -0.454          -0.617
symmetric   0.2  relabel        DT threshold        MLP    84.004    82.557          -0.684          -0.841
symmetric   0.2  relabel        NB hard_vote        MLP    83.674    82.662          -1.014          -0.736
symmetric   0.2  relabel        NB threshold        MLP    84.471    83.689          -0.217           0.292
symmetric   0.2  relabel committee consensus        MLP    86.228    85.140           1.540           1.742
symmetric   0.2  relabel      dual agreement        MLP    85.960    85.044           1.272           1.647
symmetric   0.2  relabel                none        MLP    84.688    83.398           0.000           0.000
symmetric   0.2  relabel    reference labels        MLP    87.097    85.803           2.409           2.405