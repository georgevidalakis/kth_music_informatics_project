


# Adversarial Attack on Music Classifiers
Project for [Music Informatics(DT2470)](https://www.kth.se/student/kurser/kurs/DT2470?l=en) at KTH.

In the project we explore how we can perform an Adversarial attack using Projected Gradient Descent(PGD) method for fooling a Music classifier.
The Music classifier first learns to discriminate between Human & AI generated music. 

**Dataset:**
Visualization of CLAP embeddings using TSNE. By rotating our points in reduced 3D space we notice we could train a network to learn non-linear decision boundary. 

![image](https://github.com/user-attachments/assets/36342758-fefe-4b03-8c96-63fb8c86b498)





**Baseline models:**
We first trained baseline models  based on which we decided to adopt the architecture for Adversarial attack process.
- LR
- kNN
- SVC
- Gaussian Naive Bayes
- Random Forest
- MLP 

### Training Results

We trained MLP and sequence based models such as LSTM and Transformer for the classification task. We provide the necessary hyperparmeters that were obtained after hyper-parameter tuning
for 5-fold evaluation.

| Model            | Fit Scaler     | Hidden Size  | No of Layers              | Dropout  | Batch Size | Learning Rate | Weight Decay | Bidirectional |
|------------------|----------------|--------------|---------------------------|----------|------------|---------------|--------------|---------------|
| MLP              | No             | 64           | 2                         | 33.6%    | 16         | 0.010         | 0.000076     | No            |
| LSTM             | No             | 128          | 2                         | 38.5%    | 16         | 0.00063       | 0.000019     | No            |
| Transformer      | No (LayerNorm) | 1,024        | 3 (pre & post Linear layers) | 10.0% | 16         | 0.00013       | 0.0047       | No            |



| Model       | CV Accuracy (mean ± std) | Test Accuracy | Test F1 Score | Test AUC  |
|-------------|--------------------------|---------------|---------------|-----------|
| MLP         | 97.14% ± 1.28%           | 96.50%        | 96.52%        | 96.50%    |
| LSTM        | 96.86% ± 2.42%           | 96.50%        | 96.55%        | 96.50%    |
| Transformer | 97.29% ± 1.83%           | 94.50%        | 94.58%        | 94.50%    |






We noticed that it is super easy to overfit on validation sets, and cross-validation is super important for training. A simple model such as MLP performs very well and gives a good trade-off
between CV and Test scores. 

### Adversarial Attack Process

![image](https://github.com/user-attachments/assets/a90a1362-f0c0-47e9-a6ea-721361a8b5bb)


The adversarial attack was performed using Projected Gradient Descent(PGD) method by projecting the added noise to a constrained SNR space. 
The PGD attack was performed on ~50 audio files to fool the classifier from AI to Human label.This was adopted from the method proposed in [Deep Music Adversaries](https://ieeexplore.ieee.org/document/7254179) that was done for MFCC's. We adopt this method to do it on the Waveform since backpropogration was done through CLAP embeddings.
INSERT EQUATION 


## Results
In the plot below we showcase results for MLP only since comprehensively evaluating attack on Transformer takes ~20 mins per attack iteration
We notice that higher learning rate leads to faster convergence of the PGD attack ut when the require min SNR is high it may lead to failures. 
Another interesting observation was how the attack seem to struggle on audio's of short length probably due to the classifier associating shorter audio lengths with AI gen music,
hence being difficult to fool. We also notice that by constraining based on SNR the model learns temporally what regions are best to add noise.

![d251fbcf-c9d5-4057-812a-5badfa81a0ea](https://github.com/user-attachments/assets/b7649cf2-2829-4ea7-9a71-89e46e325add)


Results for Transformer....loading. (Seriously tho can we have some more GPUs)
# Conclusion
We notice that the Adversarial attack sometimes fail to attack audio-lengths that are very short. This
would mean that audio-length could play an important factor learned by our classifier. Moreover, we also
observe that due to constraints on just the amount of noise added, the model doesn’t seem to utilize the
frequency aspect as well. The attack learns to add noise at intervals without showing much variation in the
types of noise it adds. This means we might need more rigorous constraints for our attack to utilize this
dimension as well.

# References
Based on "Deep Music Adversaries" https://ieeexplore.ieee.org/document/7254179
- [1] Duan, R., Qu, Z., Zhao, S., Ding, L., Liu, Y., & Lu, Z. (2022). Perception-aware attack: Creating adversarial music via reverse-engineering human perception. In Proceedings of the 2022 ACM SIGSAC conference on computer and communications security (pp. 905-919). Association for Computing Machinery. URL: https://doi.org/10.1145/3548606.3559350

- [2]Elizalde, B., Deshmukh, S., Al Ismail, M., & Wang, H. (2023). Clap learning audio concepts from natural language supervision. In 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1-5). IEEE. URL: https://doi.org/10.48550/arXiv.2206.04769

- [3]Ren, I. Y. (2015). Comparing Artificial Intelligence Music and Human Music: A Case Study of Prof. David Cope’s Emmy System. University of Rochester. URL: https://hajim.rochester.edu/ece/sites/zduan/teaching/ece477/projects/2015/Iris_Ren_ReportFinal.pdf

- [4]Saadatpanah, P., Shafahi, A., & Goldstein, T. (2020). Adversarial attacks on copyright detection systems. In H. Larochelle, M. Balcan, R. Pineau, D. Prelec (Eds.), Proceedings of the 37th International Conference on Machine Learning (pp. 8507-8516). URL: https://proceedings.mlr.press/v119/saadatpanah20a.html

- [5]Saijo, K., Ebbers, J., Germain, F. G., Khurana, S., Wichern, G., & Roux, J. L. (2024). Leveraging Audio-Only Data for Text-Queried Target Sound Extraction. URL: https://arxiv.org/abs/2409.13152

- [6]Subramanian, V. (2019). Adversarial Attacks in Sound Event Classification. URL:. https://arxiv.org/abs/1907.02477

- [7]Szegedy, C., et al. (2014). Intriguing properties of neural networks. URL: https://arxiv.org/abs/1312.6199


**Authors:** Refer report.


