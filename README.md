# TMT-MLiP

Bengali Digit Recognition: http://dspace.daffodilvarsity.edu.bd:8080/bitstream/handle/123456789/3678/P13399%20%2821%25%29.pdf?sequence=1&isAllowed=y

Max:

 I'm going to experiment with:
 
 - CutMix
 - Dropout
 - S&E
 - The 'Rolling Trainingset' approach
 - Oversampling

Ruben:

- Pretrain on isolated components
- preprocessing https://www.kaggle.com/iafoss/image-preprocessing-128x128
  - Is praktisch hetzelfde als wat we al hebben, helaas.
- More data augmentation
  - https://www.kaggle.com/corochann/bengali-albumentations-data-augmentation-tutorial
  - add noise to data
- Use feather files
  - Speeds up data reading 30x
  -https://www.kaggle.com/corochann/bangali-ai-super-fast-data-loading-with-feather
- Use class_map_corrected.csv
  - Improves dense_5 accuracy from 0.6791 to 0.6873 after one epoch
  
Taras:
- Chinese character recognition https://www.researchgate.net/publication/220412438_Chinese_character_recognition_History_status_and_prospects

S&E blocks: http://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
Iterative stratification https://www.kaggle.com/yiheng/iterative-stratification

