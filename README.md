# Poisoning-Assisted Membership Inference in Federated Learning
## IEEE Transactions on Dependable and Secure Computing (TDSC)

## Getting Started
To run this repository, we kindly advise you to install python 3.7 and PyTorch 1.9.1
with Anaconda. You may download Anaconda and read the installation instruction on the official
website (https://www.anaconda.com/download/).
Create a new environment and install PyTorch and torchvision on it:


```shell
conda create --name xxx - pytorch python == 3.7
conda activate xxx - pytorch
conda install pytorch == 1.9.1
conda install torchvision -c pytorch
```


Install other requirements:
```shell
pip install numpy scikit-learn matplotlib xgboost os random copy 
```

```shell
python FL/main.py
```

If you find this code or our paper useful for your research, please cite:
```bibtex
@ARTICLE{11544110,
  author={Luan, Xukun and Bi, Yuanguo and Zhang, Kuan and Huang, Zixuan and Su, Zhou and Luan, Tom H. and Hu, Bing},
  journal={IEEE Transactions on Dependable and Secure Computing}, 
  title={Poisoning-Assisted Membership Inference in Federated Learning}, 
  year={2026},
  volume={},
  number={},
  pages={1-17},
  keywords={Modeling;Toxicology;Labeling;Privacy;Training;Federated learning;Accuracy;Machine learning;Educational institutions;Computers;Federated learning;membership inference attack;membership inference defense;poisoning attack;temporal evolution},
  doi={10.1109/TDSC.2026.3699355}}
```
