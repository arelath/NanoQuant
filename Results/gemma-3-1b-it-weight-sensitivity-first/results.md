# Gemma 3 1B IT weight sensitivity

Measured 78 of 78 trials.
Scored 4,080 next-token positions per trial.
Teacher dataset at `D:\dev\research\NanoQuantRewrite\Datasets\gemma-3-1b-it-fp16-sensitivity-16x256`.
Unchanged FP16 model KL against the stored teacher is 2.56581e-07 nats/token.

Gaussian weight noise is scaled to the requested relative Frobenius error per projection.
Measured weight error includes FP16 rounding. KL is full-vocabulary KL(teacher ‖ perturbed).
KL above the control is the signed difference from the unchanged model.

## Requested weight error 10.000%

| Target | Actual weight error | Mean KL | KL above control | Seed SD | Seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| block 0 | 10.000% | 0.209449 | 0.209449 | 0.0138612 | 3 |
| block 1 | 10.000% | 0.144022 | 0.144022 | 0.00644179 | 3 |
| block 2 | 10.000% | 0.118865 | 0.118865 | 0.00468341 | 3 |
| block 3 | 10.000% | 0.105843 | 0.105843 | 0.0019445 | 3 |
| block 4 | 10.000% | 0.0825182 | 0.0825179 | 0.00229566 | 3 |
| block 5 | 10.000% | 0.0759625 | 0.0759623 | 0.0128196 | 3 |
| block 6 | 10.000% | 0.0541169 | 0.0541166 | 0.00328102 | 3 |
| block 7 | 10.000% | 0.0428748 | 0.0428746 | 0.00281731 | 3 |
| block 8 | 10.000% | 0.0365713 | 0.0365711 | 0.00403873 | 3 |
| block 10 | 10.000% | 0.0352863 | 0.0352861 | 0.00210625 | 3 |
| block 17 | 10.000% | 0.0302049 | 0.0302046 | 0.0024388 | 3 |
| block 9 | 10.000% | 0.0273357 | 0.0273354 | 0.00122103 | 3 |
| block 11 | 10.000% | 0.0264086 | 0.0264084 | 0.00240597 | 3 |
| block 16 | 10.000% | 0.0255966 | 0.0255963 | 0.00336687 | 3 |
| block 25 | 10.000% | 0.0219445 | 0.0219442 | 0.000139315 | 3 |
| block 12 | 10.000% | 0.0214462 | 0.0214459 | 0.00128647 | 3 |
| block 15 | 10.000% | 0.0205343 | 0.020534 | 0.00200579 | 3 |
| block 23 | 10.000% | 0.0183058 | 0.0183056 | 0.000954897 | 3 |
| block 19 | 10.000% | 0.0154607 | 0.0154605 | 0.000183055 | 3 |
| block 14 | 10.000% | 0.0152829 | 0.0152827 | 0.00127618 | 3 |
| block 18 | 10.000% | 0.01492 | 0.0149198 | 0.00118507 | 3 |
| block 13 | 10.000% | 0.0144665 | 0.0144663 | 0.000560083 | 3 |
| block 21 | 10.000% | 0.0122703 | 0.01227 | 0.000834981 | 3 |
| block 22 | 10.000% | 0.0120707 | 0.0120705 | 0.000735129 | 3 |
| block 20 | 10.000% | 0.0116545 | 0.0116543 | 0.000453165 | 3 |
| block 24 | 10.000% | 0.0111155 | 0.0111152 | 0.000460044 | 3 |
