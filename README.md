# codecformer

This repository contains a number of scripts required for replicating Codecformer within the speechbrain framework. Unfortunately, they will have to be manually copied into the respective directories manually.

train_cdf.py -> recipes/WSJ02Mix/Separation

DAC_original_L4nq.yaml -> recipes/WSJ02Mix/Separation/hparams

codecformer3.py -> speechbrain/lobes/models

For replication efforts, please note that the activation function of the simpleseparator2 model has a big impact on performance. Ensure that the activation function of the separator matches the activation function used in the final layer of the neural audio codec's encoder.

If you found this useful, please cite our paper below
```
@inproceedings{yip2024towards,
  title={Towards Audio Codec-based Speech Separation},
  author={Yip, Jia Qi and Zhao, Shengkui and Ng, Dianwen and Chng, Eng Siong and Ma, Bin},
  booktitle={Proc. Interspeech},
  year={2024}
}
```
