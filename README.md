This project contains IPyhton Notebook files for a Kaggle competition <a href="https://www.kaggle.com/c/tgs-salt-identification-challenge">TGS Salt Identification Challenge</a>.

The results should score around 0.836 at Private Leaderboard, top 37% of the leaderboard.

U-Net is used in this competition. This model architecture is good at image segmentation. Details can be found on this <a href="https://arxiv.org/pdf/1505.04597.pdf">paper</a>.

Conclusion:
1) EDA and error analysis is hard to perform in this dataset due to lack of domain knowledge
2) vertical flipping works but other data augmentation doesn't help
3) test-time augmentation will improve the result a little bit
4) pre-trained model as encoder will get a better result according to the discussion on Kaggle
