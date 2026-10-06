# E12v2 results

Reference labels are unverified; detection is against injected corruption only.
No novelty claim. Full thresholded Confident Learning is not implemented.
Primary metrics pool confusion matrices across outer folds per seed, using all dataset classes.

## Primary: dual agreement vs no cleaning at 20% added noise

         delta_accuracy  delta_macro_f1
dataset                                
iris              5.444           5.483

Descriptive run only: this configuration does not satisfy the locked main protocol.

## All downstream means

     kind  rate handling                 arm classifier  accuracy  macro_f1  delta_accuracy  delta_macro_f1
 pairflip   0.0  relabel        DT hard_vote         DT    96.000    96.000           2.000           2.005
 pairflip   0.0  relabel        DT hard_vote         LR    96.000    96.000           0.667           0.667
 pairflip   0.0  relabel        DT hard_vote         NB    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel        DT threshold         DT    96.000    96.000           2.000           2.005
 pairflip   0.0  relabel        DT threshold         LR    96.000    96.000           0.667           0.667
 pairflip   0.0  relabel        DT threshold         NB    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel        NB hard_vote         DT    94.667    94.667           0.667           0.672
 pairflip   0.0  relabel        NB hard_vote         LR    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel        NB hard_vote         NB    94.667    94.667          -0.667          -0.666
 pairflip   0.0  relabel        NB threshold         DT    93.333    93.333          -0.667          -0.661
 pairflip   0.0  relabel        NB threshold         LR    96.000    95.998           0.667           0.666
 pairflip   0.0  relabel        NB threshold         NB    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel committee consensus         DT    96.667    96.666           2.667           2.672
 pairflip   0.0  relabel committee consensus         LR    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel committee consensus         NB    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel      dual agreement         DT    94.000    93.999           0.000           0.005
 pairflip   0.0  relabel      dual agreement         LR    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel      dual agreement         NB    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel                none         DT    94.000    93.995           0.000           0.000
 pairflip   0.0  relabel                none         LR    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel                none         NB    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel    reference labels         DT    94.000    93.995           0.000           0.000
 pairflip   0.0  relabel    reference labels         LR    95.333    95.333           0.000           0.000
 pairflip   0.0  relabel    reference labels         NB    95.333    95.333           0.000           0.000
 pairflip   0.2  relabel        DT hard_vote         DT    78.000    77.999          -2.000          -2.048
 pairflip   0.2  relabel        DT hard_vote         LR    89.333    89.231          -3.333          -3.285
 pairflip   0.2  relabel        DT hard_vote         NB    88.667    88.406          -0.667          -0.714
 pairflip   0.2  relabel        DT threshold         DT    79.333    79.324          -0.667          -0.723
 pairflip   0.2  relabel        DT threshold         LR    89.333    89.231          -3.333          -3.285
 pairflip   0.2  relabel        DT threshold         NB    88.667    88.406          -0.667          -0.714
 pairflip   0.2  relabel        NB hard_vote         DT    88.000    87.825           8.000           7.778
 pairflip   0.2  relabel        NB hard_vote         LR    86.000    85.681          -6.667          -6.835
 pairflip   0.2  relabel        NB hard_vote         NB    90.667    90.530           1.333           1.410
 pairflip   0.2  relabel        NB threshold         DT    94.000    93.985          14.000          13.938
 pairflip   0.2  relabel        NB threshold         LR    91.333    91.312          -1.333          -1.205
 pairflip   0.2  relabel        NB threshold         NB    92.000    91.987           2.667           2.867
 pairflip   0.2  relabel committee consensus         DT    93.333    93.323          13.333          13.276
 pairflip   0.2  relabel committee consensus         LR    94.000    94.006           1.333           1.490
 pairflip   0.2  relabel committee consensus         NB    93.333    93.323           4.000           4.203
 pairflip   0.2  relabel      dual agreement         DT    92.000    91.996          12.000          11.949
 pairflip   0.2  relabel      dual agreement         LR    93.333    93.360           0.667           0.844
 pairflip   0.2  relabel      dual agreement         NB    93.333    93.323           4.000           4.203
 pairflip   0.2  relabel                none         DT    80.000    80.047           0.000           0.000
 pairflip   0.2  relabel                none         LR    92.667    92.516           0.000           0.000
 pairflip   0.2  relabel                none         NB    89.333    89.120           0.000           0.000
 pairflip   0.2  relabel    reference labels         DT    94.000    93.995          14.000          13.948
 pairflip   0.2  relabel    reference labels         LR    95.333    95.333           2.667           2.817
 pairflip   0.2  relabel    reference labels         NB    95.333    95.333           6.000           6.213
symmetric   0.0  relabel        DT hard_vote         DT    96.000    96.000           2.000           2.005
symmetric   0.0  relabel        DT hard_vote         LR    96.000    96.000           0.667           0.667
symmetric   0.0  relabel        DT hard_vote         NB    95.333    95.333           0.000           0.000
symmetric   0.0  relabel        DT threshold         DT    96.000    96.000           2.000           2.005
symmetric   0.0  relabel        DT threshold         LR    96.000    96.000           0.667           0.667
symmetric   0.0  relabel        DT threshold         NB    95.333    95.333           0.000           0.000
symmetric   0.0  relabel        NB hard_vote         DT    94.667    94.667           0.667           0.672
symmetric   0.0  relabel        NB hard_vote         LR    95.333    95.333           0.000           0.000
symmetric   0.0  relabel        NB hard_vote         NB    94.667    94.667          -0.667          -0.666
symmetric   0.0  relabel        NB threshold         DT    93.333    93.333          -0.667          -0.661
symmetric   0.0  relabel        NB threshold         LR    96.000    95.998           0.667           0.666
symmetric   0.0  relabel        NB threshold         NB    95.333    95.333           0.000           0.000
symmetric   0.0  relabel committee consensus         DT    96.667    96.666           2.667           2.672
symmetric   0.0  relabel committee consensus         LR    95.333    95.333           0.000           0.000
symmetric   0.0  relabel committee consensus         NB    95.333    95.333           0.000           0.000
symmetric   0.0  relabel      dual agreement         DT    94.000    93.999           0.000           0.005
symmetric   0.0  relabel      dual agreement         LR    95.333    95.333           0.000           0.000
symmetric   0.0  relabel      dual agreement         NB    95.333    95.333           0.000           0.000
symmetric   0.0  relabel                none         DT    94.000    93.995           0.000           0.000
symmetric   0.0  relabel                none         LR    95.333    95.333           0.000           0.000
symmetric   0.0  relabel                none         NB    95.333    95.333           0.000           0.000
symmetric   0.0  relabel    reference labels         DT    94.000    93.995           0.000           0.000
symmetric   0.0  relabel    reference labels         LR    95.333    95.333           0.000           0.000
symmetric   0.0  relabel    reference labels         NB    95.333    95.333           0.000           0.000
symmetric   0.2  relabel        DT hard_vote         DT    80.000    80.141          -2.000          -1.929
symmetric   0.2  relabel        DT hard_vote         LR    82.667    82.576          -4.667          -4.726
symmetric   0.2  relabel        DT hard_vote         NB    90.667    90.667          -2.667          -2.664
symmetric   0.2  relabel        DT threshold         DT    80.000    80.141          -2.000          -1.929
symmetric   0.2  relabel        DT threshold         LR    82.667    82.576          -4.667          -4.726
symmetric   0.2  relabel        DT threshold         NB    90.667    90.667          -2.667          -2.664
symmetric   0.2  relabel        NB hard_vote         DT    89.333    89.333           7.333           7.263
symmetric   0.2  relabel        NB hard_vote         LR    88.000    88.053           0.667           0.751
symmetric   0.2  relabel        NB hard_vote         NB    91.333    91.332          -2.000          -1.998
symmetric   0.2  relabel        NB threshold         DT    91.333    91.249           9.333           9.179
symmetric   0.2  relabel        NB threshold         LR    93.333    93.333           6.000           6.032
symmetric   0.2  relabel        NB threshold         NB    94.000    93.999           0.667           0.669
symmetric   0.2  relabel committee consensus         DT    93.333    93.331          11.333          11.261
symmetric   0.2  relabel committee consensus         LR    92.000    91.997           4.667           4.695
symmetric   0.2  relabel committee consensus         NB    92.000    91.997          -1.333          -1.334
symmetric   0.2  relabel      dual agreement         DT    92.000    91.945          10.000           9.875
symmetric   0.2  relabel      dual agreement         LR    93.333    93.331           6.000           6.029
symmetric   0.2  relabel      dual agreement         NB    93.333    93.331           0.000           0.000
symmetric   0.2  relabel                none         DT    82.000    82.070           0.000           0.000
symmetric   0.2  relabel                none         LR    87.333    87.302           0.000           0.000
symmetric   0.2  relabel                none         NB    93.333    93.331           0.000           0.000
symmetric   0.2  relabel    reference labels         DT    94.000    93.995          12.000          11.925
symmetric   0.2  relabel    reference labels         LR    95.333    95.333           8.000           8.031
symmetric   0.2  relabel    reference labels         NB    95.333    95.333           2.000           2.002