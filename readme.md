## 基础部分

运行bi-loss方法的训练，xxx为模型（cnn，snn，mnn）

```bash
python xxx_reverse.py 
```

运行bi-model方法的训练，xxx为模型（cnn，snn，mnn）

```bash
python xxx_bimodel.py 
```

运行数据集后门调整方法的训练，xxx为模型（cnn，snn，mnn）

```bash
python xxx_backdoor.py 
```

请注意，在运行mnn方法前，请先训练对应的cnn模型

## 提高部分

运行提高部分的方法，xxx为方法名（dro，dann，irm，cnc）

```bash
python xxx.py 
```

# Reference

CNC: [Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations](https://arxiv.org/abs/2203.01517)

DANN: [Domain-Adversarial Training of Neural Networks](https://arxiv.org/abs/1505.07818)

IRM: [Invarient Risk Minimization](https://arxiv.org/abs/1907.02893)

DRO: [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731)
