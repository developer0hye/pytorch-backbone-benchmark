# pytorch-backbone-benchmark

Benchmarks for popular neural network models supported by [timm](https://github.com/rwightman/pytorch-image-models)

- CPU: Intel(R) Core(TM) i5-10400 CPU @ 2.90GHz   2.90 GHz
- GPU: RTX 3070 8GB
- RAM: 32.0 GB
- Pytorch Version: 1.8.1
- Input Tensor Shape: 1x3x**608x608**

All timing experiments are averaged over 100 times.

sorted by execution time

The graph is [here](https://colab.research.google.com/drive/1NmkUIcA9Vt8U2WLXgZeIrIuY3yFW3Byx#scrollTo=4eXxfjPrg_PB).

|     | model                            |   top1 |   param_count |   execution time(sec) |   fps |
|----:|:---------------------------------|-------:|--------------:|----------------------:|------:|
|   0 | swsl_resnet18                    | 73.276 |         11.69 |                0.0044 |   227 |
|   1 | resnet18                         | 69.748 |         11.69 |                0.0045 |   222 |
|   2 | gluon_resnet18_v1b               | 70.836 |         11.69 |                0.0045 |   222 |
|   3 | ssl_resnet18                     | 72.61  |         11.69 |                0.0045 |   222 |
|   4 | resnet18d                        | 72.26  |         11.71 |                0.0048 |   208 |
|   5 | tf_mobilenetv3_small_minimal_100 | 62.906 |          2.04 |                0.0055 |   181 |
|   6 | legacy_seresnet18                | 71.742 |         11.78 |                0.0065 |   153 |
|   7 | tf_mobilenetv3_large_minimal_100 | 72.248 |          3.92 |                0.0073 |   136 |
|   8 | efficientnet_lite0               | 75.484 |          4.65 |                0.0074 |   135 |
|   9 | ese_vovnet19b_dw                 | 76.798 |          6.54 |                0.0074 |   135 |
|  10 | gluon_resnet34_v1b               | 74.588 |         21.8  |                0.0074 |   135 |
|  11 | tv_resnet34                      | 73.312 |         21.8  |                0.0075 |   133 |
|  12 | resnet34                         | 75.11  |         21.8  |                0.0076 |   131 |
|  13 | mobilenetv2_140                  | 76.516 |          6.11 |                0.0076 |   131 |
|  14 | regnetx_002                      | 68.762 |          2.68 |                0.0076 |   131 |
|  15 | dla34                            | 74.63  |         15.74 |                0.0077 |   129 |
|  16 | tf_efficientnet_lite0            | 74.83  |          4.65 |                0.0077 |   129 |
|  17 | mnasnet_100                      | 74.658 |          4.38 |                0.0078 |   128 |
|  18 | efficientnet_es_pruned           | 75     |          5.44 |                0.0079 |   126 |
|  19 | efficientnet_es                  | 78.066 |          5.44 |                0.0079 |   126 |
|  20 | mobilenetv2_100                  | 72.97  |          3.5  |                0.008  |   125 |
|  21 | tf_efficientnet_es               | 76.594 |          5.44 |                0.008  |   125 |
|  22 | resnet26                         | 75.292 |         16    |                0.008  |   125 |
|  23 | dla46x_c                         | 65.97  |          1.07 |                0.0083 |   120 |
|  24 | resnet34d                        | 77.116 |         21.82 |                0.0083 |   120 |
|  25 | dla46_c                          | 64.866 |          1.3  |                0.0084 |   119 |
|  26 | gernet_s                         | 76.916 |          8.17 |                0.0084 |   119 |
|  27 | resnest14d                       | 75.506 |         10.61 |                0.0084 |   119 |
|  28 | hardcorenas_a                    | 75.916 |          5.26 |                0.0086 |   116 |
|  29 | selecsls42b                      | 77.174 |         32.46 |                0.0086 |   116 |
|  30 | resnet26d                        | 76.696 |         16.01 |                0.0087 |   114 |
|  31 | gernet_m                         | 80.732 |         21.14 |                0.0088 |   113 |
|  32 | semnasnet_100                    | 75.448 |          3.89 |                0.0093 |   107 |
|  33 | ecaresnet26t                     | 79.854 |         16.01 |                0.0094 |   106 |
|  34 | regnetx_006                      | 73.852 |          6.2  |                0.0094 |   106 |
|  35 | skresnet18                       | 73.038 |         11.96 |                0.0094 |   106 |
|  36 | spnasnet_100                     | 74.084 |          4.42 |                0.0095 |   105 |
|  37 | fbnetc_100                       | 75.124 |          5.57 |                0.0096 |   104 |
|  38 | mobilenetv2_110d                 | 75.036 |          4.52 |                0.0096 |   104 |
|  39 | tf_efficientnet_lite1            | 76.642 |          5.42 |                0.0097 |   103 |
|  40 | regnetx_008                      | 75.038 |          7.26 |                0.0097 |   103 |
|  41 | tf_efficientnet_lite2            | 77.468 |          6.09 |                0.0098 |   102 |
|  42 | tf_mobilenetv3_small_100         | 67.922 |          2.54 |                0.0105 |    95 |
|  43 | hardcorenas_b                    | 76.538 |          5.18 |                0.0106 |    94 |
|  44 | efficientnet_em                  | 79.252 |          6.9  |                0.0106 |    94 |
|  45 | gernet_l                         | 81.354 |         31.08 |                0.0107 |    93 |
|  46 | dla60x_c                         | 67.892 |          1.32 |                0.0108 |    92 |
|  47 | cspresnet50                      | 79.574 |         21.62 |                0.0108 |    92 |
|  48 | tf_efficientnet_em               | 78.13  |          6.9  |                0.0108 |    92 |
|  49 | selecsls60                       | 77.982 |         30.67 |                0.011  |    90 |
|  50 | tf_mobilenetv3_small_075         | 65.716 |          2.04 |                0.011  |    90 |
|  51 | tf_efficientnet_lite3            | 79.82  |          8.2  |                0.0111 |    90 |
|  52 | regnety_002                      | 70.252 |          3.16 |                0.0112 |    89 |
|  53 | repvgg_a2                        | 76.46  |         28.21 |                0.0112 |    89 |
|  54 | regnetx_016                      | 76.95  |          9.19 |                0.0113 |    88 |
|  55 | efficientnet_b0                  | 77.698 |          5.29 |                0.0114 |    87 |
|  56 | selecsls60b                      | 78.412 |         32.77 |                0.0114 |    87 |
|  57 | mobilenetv3_large_100            | 75.766 |          5.48 |                0.0115 |    86 |
|  58 | mobilenetv3_large_100_miil       | 77.916 |          5.48 |                0.0115 |    86 |
|  59 | hardcorenas_c                    | 77.054 |          5.52 |                0.0115 |    86 |
|  60 | mobilenetv2_120d                 | 77.284 |          5.83 |                0.0116 |    86 |
|  61 | legacy_seresnext26_32x4d         | 77.104 |         16.79 |                0.0116 |    86 |
|  62 | tf_efficientnet_b0_ns            | 78.658 |          5.29 |                0.0116 |    86 |
|  63 | tf_mobilenetv3_large_100         | 75.518 |          5.48 |                0.0116 |    86 |
|  64 | resnest26d                       | 78.478 |         17.07 |                0.0117 |    85 |
|  65 | tf_efficientnet_b0               | 76.848 |          5.29 |                0.012  |    83 |
|  66 | legacy_seresnet34                | 74.808 |         21.96 |                0.012  |    83 |
|  67 | tf_efficientnet_b0_ap            | 77.086 |          5.29 |                0.012  |    83 |
|  68 | mobilenetv3_rw                   | 75.634 |          5.48 |                0.0122 |    81 |
|  69 | seresnext26d_32x4d               | 77.602 |         16.81 |                0.0123 |    81 |
|  70 | seresnext26t_32x4d               | 77.986 |         16.81 |                0.0123 |    81 |
|  71 | regnetx_004                      | 72.396 |          5.16 |                0.0124 |    80 |
|  72 | ese_vovnet39b                    | 79.32  |         24.57 |                0.0126 |    79 |
|  73 | ecaresnet50d_pruned              | 79.716 |         19.94 |                0.0127 |    78 |
|  74 | rexnet_100                       | 77.858 |          4.8  |                0.0127 |    78 |
|  75 | ecaresnetlight                   | 80.462 |         30.16 |                0.0127 |    78 |
|  76 | rexnet_130                       | 79.5   |          7.56 |                0.0127 |    78 |
|  77 | tv_resnet50                      | 76.138 |         25.56 |                0.0127 |    78 |
|  78 | swsl_resnet50                    | 81.166 |         25.56 |                0.0127 |    78 |
|  79 | tf_mobilenetv3_large_075         | 73.438 |          3.99 |                0.0128 |    78 |
|  80 | repvgg_b0                        | 75.152 |         15.82 |                0.0128 |    78 |
|  81 | resnet50                         | 79.038 |         25.56 |                0.0128 |    78 |
|  82 | ssl_resnet50                     | 79.222 |         25.56 |                0.0128 |    78 |
|  83 | rexnet_150                       | 80.31  |          9.73 |                0.0128 |    78 |
|  84 | regnety_008                      | 76.316 |          6.26 |                0.0128 |    78 |
|  85 | gluon_resnet50_v1b               | 77.58  |         25.56 |                0.0129 |    77 |
|  86 | dla60                            | 77.032 |         22.04 |                0.013  |    76 |
|  87 | rexnet_200                       | 81.632 |         16.37 |                0.0131 |    76 |
|  88 | gluon_resnet50_v1c               | 78.012 |         25.58 |                0.0133 |    75 |
|  89 | regnety_006                      | 75.246 |          6.06 |                0.0133 |    75 |
|  90 | gluon_resnet50_v1d               | 79.074 |         25.58 |                0.0135 |    74 |
|  91 | resnet50d                        | 80.53  |         25.58 |                0.0136 |    73 |
|  92 | regnety_004                      | 74.034 |          4.34 |                0.0137 |    72 |
|  93 | tf_efficientnet_lite4            | 81.536 |         13.01 |                0.0139 |    71 |
|  94 | cspdarknet53                     | 80.058 |         27.64 |                0.014  |    71 |
|  95 | cspresnext50                     | 80.04  |         20.57 |                0.0141 |    70 |
|  96 | hardcorenas_e                    | 77.794 |          8.07 |                0.0143 |    69 |
|  97 | resnetblur50                     | 79.286 |         25.56 |                0.0143 |    69 |
|  98 | tf_efficientnet_cc_b0_4e         | 77.306 |         13.31 |                0.0144 |    69 |
|  99 | tf_efficientnet_cc_b0_8e         | 77.908 |         24.01 |                0.0145 |    68 |
| 100 | res2net50_48w_2s                 | 77.522 |         25.29 |                0.0145 |    68 |
| 101 | legacy_seresnet50                | 77.63  |         28.09 |                0.0147 |    68 |
| 102 | regnetx_032                      | 78.172 |         15.3  |                0.0147 |    68 |
| 103 | hardcorenas_f                    | 78.104 |          8.2  |                0.0147 |    68 |
| 104 | ecaresnet50t                     | 82.346 |         25.57 |                0.0147 |    68 |
| 105 | ecaresnet50d                     | 80.592 |         25.58 |                0.0148 |    67 |
| 106 | mixnet_s                         | 75.992 |          4.13 |                0.0148 |    67 |
| 107 | gluon_resnet50_v1s               | 78.712 |         25.68 |                0.015  |    66 |
| 108 | ghostnet_100                     | 73.978 |          5.18 |                0.0151 |    66 |
| 109 | seresnet50                       | 80.274 |         28.09 |                0.0152 |    65 |
| 110 | efficientnet_el_pruned           | 80.3   |         10.59 |                0.0153 |    65 |
| 111 | efficientnet_el                  | 81.316 |         10.59 |                0.0154 |    64 |
| 112 | tf_efficientnet_el               | 80.25  |         10.59 |                0.0154 |    64 |
| 113 | dpn68                            | 76.318 |         12.61 |                0.0156 |    64 |
| 114 | hardcorenas_d                    | 77.432 |          7.5  |                0.0156 |    64 |
| 115 | efficientnet_b1                  | 78.794 |          7.79 |                0.0157 |    63 |
| 116 | tf_mixnet_s                      | 75.65  |          4.13 |                0.0158 |    63 |
| 117 | tf_inception_v3                  | 77.856 |         23.83 |                0.0159 |    62 |
| 118 | inception_v3                     | 77.438 |         23.83 |                0.016  |    62 |
| 119 | adv_inception_v3                 | 77.582 |         23.83 |                0.016  |    62 |
| 120 | xception                         | 79.052 |         22.86 |                0.016  |    62 |
| 121 | regnetx_040                      | 78.482 |         22.12 |                0.016  |    62 |
| 122 | hrnet_w18_small                  | 72.342 |         13.19 |                0.016  |    62 |
| 123 | gluon_inception_v3               | 78.806 |         23.83 |                0.0161 |    62 |
| 124 | efficientnet_b2                  | 80.612 |          9.11 |                0.0161 |    62 |
| 125 | tf_efficientnet_b2_ap            | 80.3   |          9.11 |                0.0161 |    62 |
| 126 | efficientnet_b1_pruned           | 78.236 |          6.33 |                0.0161 |    62 |
| 127 | efficientnet_b2_pruned           | 79.916 |          8.31 |                0.0162 |    61 |
| 128 | dla60x                           | 78.246 |         17.35 |                0.0162 |    61 |
| 129 | resnetv2_50x1_bitm               | 80.172 |         25.55 |                0.0163 |    61 |
| 130 | tf_efficientnet_b1               | 78.826 |          7.79 |                0.0163 |    61 |
| 131 | tf_efficientnet_b1_ns            | 81.388 |          7.79 |                0.0164 |    60 |
| 132 | resnetrs50                       | 79.892 |         35.69 |                0.0164 |    60 |
| 133 | tf_efficientnet_b1_ap            | 79.28  |          7.79 |                0.0164 |    60 |
| 134 | tf_efficientnet_b2_ns            | 82.38  |          9.11 |                0.0165 |    60 |
| 135 | tf_efficientnet_b2               | 80.086 |          9.11 |                0.0165 |    60 |
| 136 | res2net50_26w_4s                 | 77.964 |         25.7  |                0.0165 |    60 |
| 137 | tv_resnext50_32x4d               | 77.62  |         25.03 |                0.0167 |    59 |
| 138 | ssl_resnext50_32x4d              | 80.318 |         25.03 |                0.0167 |    59 |
| 139 | dpn68b                           | 79.216 |         12.61 |                0.0167 |    59 |
| 140 | resnest50d_1s4x24d               | 80.988 |         25.68 |                0.0168 |    59 |
| 141 | swsl_resnext50_32x4d             | 82.182 |         25.03 |                0.0168 |    59 |
| 142 | gluon_resnext50_32x4d            | 79.354 |         25.03 |                0.0168 |    59 |
| 143 | resnext50_32x4d                  | 79.768 |         25.03 |                0.0168 |    59 |
| 144 | res2next50                       | 78.246 |         24.67 |                0.0172 |    58 |
| 145 | nf_resnet50                      | 80.694 |         25.56 |                0.0173 |    57 |
| 146 | skresnet34                       | 76.912 |         22.28 |                0.0174 |    57 |
| 147 | resnext50d_32x4d                 | 79.676 |         25.05 |                0.0174 |    57 |
| 148 | repvgg_b1g4                      | 77.594 |         39.97 |                0.0175 |    57 |
| 149 | mixnet_l                         | 78.976 |          7.33 |                0.0177 |    56 |
| 150 | dla60_res2net                    | 78.464 |         20.85 |                0.0178 |    56 |
| 151 | vgg11                            | 69.024 |        132.86 |                0.0178 |    56 |
| 152 | mixnet_m                         | 77.26  |          5.01 |                0.0179 |    55 |
| 153 | resnest50d                       | 80.974 |         27.48 |                0.018  |    55 |
| 154 | tf_efficientnet_b3_ns            | 84.048 |         12.23 |                0.0182 |    54 |
| 155 | efficientnet_b3                  | 82.242 |         12.23 |                0.0185 |    54 |
| 156 | tf_efficientnet_b3               | 81.636 |         12.23 |                0.0186 |    53 |
| 157 | vgg11_bn                         | 70.36  |        132.87 |                0.0187 |    53 |
| 158 | xception41                       | 78.516 |         26.97 |                0.0188 |    53 |
| 159 | tf_efficientnet_b3_ap            | 81.822 |         12.23 |                0.0189 |    52 |
| 160 | efficientnet_b3_pruned           | 80.858 |          9.86 |                0.0189 |    52 |
| 161 | tf_mixnet_m                      | 76.942 |          5.01 |                0.019  |    52 |
| 162 | tf_mixnet_l                      | 78.774 |          7.33 |                0.0191 |    52 |
| 163 | eca_nfnet_l0                     | 82.588 |         24.14 |                0.0191 |    52 |
| 164 | dla60_res2next                   | 78.44  |         17.03 |                0.0192 |    52 |
| 165 | gluon_seresnext50_32x4d          | 79.918 |         27.56 |                0.0194 |    51 |
| 166 | seresnext50_32x4d                | 81.266 |         27.56 |                0.0194 |    51 |
| 167 | legacy_seresnext50_32x4d         | 79.078 |         27.56 |                0.0195 |    51 |
| 168 | regnety_032                      | 82.724 |         19.44 |                0.0196 |    51 |
| 169 | regnety_040                      | 79.22  |         20.65 |                0.0199 |    50 |
| 170 | regnetx_064                      | 79.072 |         26.21 |                0.02   |    50 |
| 171 | dla102                           | 78.032 |         33.27 |                0.0201 |    49 |
| 172 | regnetx_080                      | 79.194 |         39.57 |                0.0203 |    49 |
| 173 | tf_efficientnet_cc_b1_8e         | 79.308 |         39.72 |                0.0204 |    49 |
| 174 | nfnet_l0                         | 82.76  |         35.07 |                0.0205 |    48 |
| 175 | resnest50d_4s2x40d               | 81.108 |         30.42 |                0.0214 |    46 |
| 176 | tv_resnet101                     | 77.374 |         44.55 |                0.0219 |    45 |
| 177 | tv_densenet121                   | 74.738 |          7.98 |                0.0219 |    45 |
| 178 | densenet121                      | 75.578 |          7.98 |                0.022  |    45 |
| 179 | res2net50_26w_6s                 | 78.57  |         37.05 |                0.0221 |    45 |
| 180 | densenetblur121d                 | 76.588 |          8    |                0.0221 |    45 |
| 181 | gluon_resnet101_v1b              | 79.306 |         44.55 |                0.0221 |    45 |
| 182 | mixnet_xl                        | 80.476 |         11.9  |                0.0224 |    44 |
| 183 | gluon_resnet101_v1c              | 79.534 |         44.57 |                0.0226 |    44 |
| 184 | vgg13                            | 69.926 |        133.05 |                0.0227 |    44 |
| 185 | gluon_resnet101_v1d              | 80.414 |         44.57 |                0.0228 |    43 |
| 186 | resnet101d                       | 83.022 |         44.57 |                0.0228 |    43 |
| 187 | efficientnet_b4                  | 83.428 |         19.34 |                0.023  |    43 |
| 188 | tf_efficientnet_b4_ns            | 85.162 |         19.34 |                0.0232 |    43 |
| 189 | wide_resnet50_2                  | 81.456 |         68.88 |                0.0232 |    43 |
| 190 | tf_efficientnet_b4               | 83.022 |         19.34 |                0.0232 |    43 |
| 191 | repvgg_b1                        | 78.366 |         57.42 |                0.0233 |    42 |
| 192 | tf_efficientnet_b4_ap            | 83.248 |         19.34 |                0.0235 |    42 |
| 193 | regnety_016                      | 77.862 |         11.2  |                0.0238 |    42 |
| 194 | regnety_064                      | 79.722 |         30.58 |                0.0239 |    41 |
| 195 | skresnext50_32x4d                | 80.156 |         27.48 |                0.0243 |    41 |
| 196 | repvgg_b2g4                      | 79.366 |         61.76 |                0.0243 |    41 |
| 197 | ecaresnet101d_pruned             | 80.818 |         24.88 |                0.0244 |    40 |
| 198 | gluon_resnet101_v1s              | 80.302 |         44.67 |                0.0244 |    40 |
| 199 | nf_regnet_b1                     | 79.306 |         10.22 |                0.0245 |    40 |
| 200 | vgg13_bn                         | 71.594 |        133.05 |                0.0246 |    40 |
| 201 | regnety_080                      | 79.876 |         39.18 |                0.0247 |    40 |
| 202 | ecaresnet101d                    | 82.172 |         44.57 |                0.025  |    40 |
| 203 | dla102x                          | 78.51  |         26.31 |                0.0252 |    39 |
| 204 | res2net50_14w_8s                 | 78.15  |         25.06 |                0.0263 |    38 |
| 205 | gluon_xception65                 | 79.716 |         39.92 |                0.0265 |    37 |
| 206 | xception65                       | 79.552 |         39.92 |                0.0266 |    37 |
| 207 | inception_v4                     | 80.168 |         42.68 |                0.027  |    37 |
| 208 | vgg16                            | 71.594 |        138.36 |                0.0273 |    36 |
| 209 | legacy_seresnet101               | 78.382 |         49.33 |                0.0274 |    36 |
| 210 | tf_efficientnet_b5_ns            | 86.088 |         30.39 |                0.0276 |    36 |
| 211 | tf_efficientnet_b5               | 83.812 |         30.39 |                0.0277 |    36 |
| 212 | tf_efficientnet_b5_ap            | 84.252 |         30.39 |                0.0277 |    36 |
| 213 | res2net50_26w_8s                 | 79.198 |         48.4  |                0.0282 |    35 |
| 214 | resnetrs101                      | 82.288 |         63.62 |                0.0283 |    35 |
| 215 | ssl_resnext101_32x4d             | 80.924 |         44.18 |                0.0285 |    35 |
| 216 | swsl_resnext101_32x4d            | 83.23  |         44.18 |                0.0285 |    35 |
| 217 | gluon_resnext101_32x4d           | 80.334 |         44.18 |                0.0287 |    34 |
| 218 | regnetx_120                      | 79.596 |         46.11 |                0.0288 |    34 |
| 219 | hrnet_w18_small_v2               | 75.114 |         15.6  |                0.029  |    34 |
| 220 | vgg16_bn                         | 73.35  |        138.37 |                0.0293 |    34 |
| 221 | resnetv2_101x1_bitm              | 82.212 |         44.54 |                0.0294 |    34 |
| 222 | densenet169                      | 75.906 |         14.15 |                0.0302 |    33 |
| 223 | res2net101_26w_4s                | 79.198 |         45.21 |                0.0305 |    32 |
| 224 | densenet161                      | 77.358 |         28.68 |                0.0312 |    32 |
| 225 | repvgg_b3g4                      | 80.212 |         83.83 |                0.0313 |    31 |
| 226 | regnety_120                      | 80.366 |         51.82 |                0.0314 |    31 |
| 227 | dla169                           | 78.688 |         53.39 |                0.0315 |    31 |
| 228 | tv_resnet152                     | 78.312 |         60.19 |                0.0318 |    31 |
| 229 | vgg19                            | 72.368 |        143.67 |                0.0321 |    31 |
| 230 | gluon_resnet152_v1b              | 79.686 |         60.19 |                0.0322 |    31 |
| 231 | dpn92                            | 80.008 |         37.67 |                0.0324 |    30 |
| 232 | tf_efficientnet_b6               | 84.11  |         43.04 |                0.0325 |    30 |
| 233 | tf_efficientnet_b6_ns            | 86.452 |         43.04 |                0.0325 |    30 |
| 234 | gluon_resnet152_v1c              | 79.91  |         60.21 |                0.0326 |    30 |
| 235 | tf_efficientnet_b6_ap            | 84.788 |         43.04 |                0.0326 |    30 |
| 236 | resnet152d                       | 83.68  |         60.21 |                0.0327 |    30 |
| 237 | gluon_resnet152_v1d              | 80.474 |         60.21 |                0.0327 |    30 |
| 238 | repvgg_b2                        | 78.792 |         89.02 |                0.0332 |    30 |
| 239 | gluon_seresnext101_32x4d         | 80.904 |         48.96 |                0.0333 |    30 |
| 240 | dm_nfnet_f0                      | 83.342 |         71.49 |                0.0334 |    29 |
| 241 | legacy_seresnext101_32x4d        | 80.228 |         48.96 |                0.0337 |    29 |
| 242 | xception71                       | 79.874 |         42.34 |                0.034  |    29 |
| 243 | gluon_resnet152_v1s              | 81.016 |         60.32 |                0.0341 |    29 |
| 244 | vgg19_bn                         | 74.214 |        143.68 |                0.0342 |    29 |
| 245 | resnest101e                      | 82.89  |         48.28 |                0.0342 |    29 |
| 246 | eca_nfnet_l1                     | 84.008 |         41.41 |                0.0352 |    28 |
| 247 | densenet201                      | 77.286 |         20.01 |                0.0361 |    27 |
| 248 | dla102x2                         | 79.448 |         41.28 |                0.0364 |    27 |
| 249 | regnetx_160                      | 79.856 |         54.28 |                0.0365 |    27 |
| 250 | regnety_160                      | 83.686 |         83.59 |                0.0396 |    25 |
| 251 | legacy_seresnet152               | 78.66  |         66.82 |                0.0408 |    24 |
| 252 | resnetrs152                      | 83.712 |         86.62 |                0.041  |    24 |
| 253 | seresnet152d                     | 84.362 |         66.84 |                0.0413 |    24 |
| 254 | inception_resnet_v2              | 80.458 |         55.84 |                0.0426 |    23 |
| 255 | ens_adv_inception_resnet_v2      | 79.982 |         55.84 |                0.0427 |    23 |
| 256 | resnet200d                       | 83.962 |         64.69 |                0.0429 |    23 |
| 257 | dpn98                            | 79.642 |         61.57 |                0.0435 |    22 |
| 258 | tf_efficientnet_b7               | 84.936 |         66.35 |                0.0441 |    22 |
| 259 | tf_efficientnet_b7_ap            | 85.12  |         66.35 |                0.0441 |    22 |
| 260 | tf_efficientnet_b7_ns            | 86.84  |         66.35 |                0.0441 |    22 |
| 261 | wide_resnet101_2                 | 78.856 |        126.89 |                0.0441 |    22 |
| 262 | repvgg_b3                        | 80.492 |        123.09 |                0.0443 |    22 |
| 263 | swsl_resnext101_32x8d            | 84.284 |         88.79 |                0.0453 |    22 |
| 264 | ig_resnext101_32x8d              | 82.688 |         88.79 |                0.0454 |    22 |
| 265 | resnext101_32x8d                 | 79.308 |         88.79 |                0.0454 |    22 |
| 266 | ssl_resnext101_32x8d             | 81.616 |         88.79 |                0.0455 |    21 |
| 267 | gluon_resnext101_64x4d           | 80.604 |         83.46 |                0.0459 |    21 |
| 268 | gluon_seresnext101_64x4d         | 80.894 |         88.23 |                0.0505 |    19 |
| 269 | resnetrs200                      | 84.066 |         93.21 |                0.0537 |    18 |
| 270 | tf_efficientnet_b8               | 85.37  |         87.41 |                0.0551 |    18 |
| 271 | tf_efficientnet_b8_ap            | 85.37  |         87.41 |                0.0552 |    18 |
| 272 | hrnet_w30                        | 78.206 |         37.71 |                0.0556 |    17 |
| 273 | pnasnet5large                    | 82.782 |         86.06 |                0.0556 |    17 |
| 274 | hrnet_w18                        | 76.758 |         21.3  |                0.0564 |    17 |
| 275 | hrnet_w32                        | 78.45  |         41.23 |                0.0572 |    17 |
| 276 | nasnetalarge                     | 82.62  |         88.75 |                0.0588 |    17 |
| 277 | dpn131                           | 79.822 |         79.25 |                0.0593 |    16 |
| 278 | dm_nfnet_f1                      | 84.604 |        132.63 |                0.0601 |    16 |
| 279 | hrnet_w48                        | 79.3   |         77.47 |                0.0609 |    16 |
| 280 | hrnet_w40                        | 78.92  |         57.56 |                0.061  |    16 |
| 281 | hrnet_w44                        | 78.896 |         67.06 |                0.0614 |    16 |
| 282 | hrnet_w64                        | 79.474 |        128.06 |                0.0629 |    15 |
| 283 | dpn107                           | 80.156 |         86.92 |                0.0634 |    15 |
| 284 | ecaresnet269d                    | 84.976 |        102.09 |                0.0635 |    15 |
| 285 | regnety_320                      | 80.812 |        145.05 |                0.0651 |    15 |
| 286 | resnest200e                      | 83.832 |         70.2  |                0.0651 |    15 |
| 287 | regnetx_320                      | 80.246 |        107.81 |                0.0657 |    15 |
| 288 | resnetrs270                      | 84.434 |        129.86 |                0.0722 |    13 |
| 289 | gluon_senet154                   | 81.234 |        115.09 |                0.0723 |    13 |
| 290 | legacy_senet154                  | 81.31  |        115.09 |                0.0727 |    13 |
| 291 | ig_resnext101_32x16d             | 84.17  |        194.03 |                0.0784 |    12 |
| 292 | ssl_resnext101_32x16d            | 81.844 |        194.03 |                0.0785 |    12 |
| 293 | swsl_resnext101_32x16d           | 83.346 |        194.03 |                0.0785 |    12 |
| 294 | resnetv2_50x3_bitm               | 83.784 |        217.32 |                0.0823 |    12 |
| 295 | resnest269e                      | 84.518 |        110.93 |                0.0866 |    11 |
| 296 | dm_nfnet_f2                      | 84.99  |        193.78 |                0.0873 |    11 |
| 297 | resnetrs350                      | 84.72  |        163.96 |                0.0931 |    10 |
| 298 | resnetrs420                      | 85.008 |        191.89 |                0.1106 |     9 |
| 299 | resnetv2_152x2_bitm              | 84.44  |        236.34 |                0.1108 |     9 |
| 300 | dm_nfnet_f3                      | 85.56  |        254.92 |                0.1143 |     8 |
| 301 | dm_nfnet_f4                      | 85.658 |        316.07 |                0.1413 |     7 |
| 302 | resnetv2_101x3_bitm              | 84.394 |        387.93 |                0.1557 |     6 |
| 303 | ig_resnext101_32x32d             | 85.094 |        468.53 |                0.1678 |     5 |
| 304 | dm_nfnet_f5                      | 85.714 |        377.21 |                0.1686 |     5 |
| 305 | tf_efficientnet_l2_ns_475        | 88.234 |        480.31 |                0.1875 |     5 |
| 306 | tf_efficientnet_l2_ns            | 88.352 |        480.31 |                0.1877 |     5 |
| 307 | dm_nfnet_f6                      | 86.296 |        438.36 |                0.1959 |     5 |
| 308 | ig_resnext101_32x48d             | 85.428 |        828.41 |                0.2854 |     3 |
| 309 | resnetv2_152x4_bitm              | 84.932 |        936.53 |                0.3693 |     2 |
