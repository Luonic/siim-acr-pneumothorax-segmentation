# TODO:
- [x] Save model
- [x] Cyclical LR
- [x] Tensorboard logging
- [x] Calc multiple dices with different threshold per batch
- [x] Crossvalidation (support simultaneous training and validation for all folds)
- [ ] Warmstart lr, 1cycle policy
- [x] Training on random crops, testing on full resolution
- [ ] Lovasz loss
- [ ] AdamW
- [x] Evaluate model perfomance (incapsulated fn in separate file)
 
- [ ] Submission creation code
- [x] DataParallel training
- [ ] Hyperparameter search 
- [x] Integrate AMP


# Important notes
Geometric mean of probabilities thresholded by mean mask and class thresold is a bad idea
Best of all is median of pixels thresolded by mean thresholds

This task should be separated to two different tasks:
1. Classification - has pneumotorax or not
2. If has - Predict semantic label map

This can be solved with single model by applying semantic label map loss only if sample has mask

Loss proposals:
1. OHEM but for each gt mask pixels separately. min_keep is num of 
elements for class with min pixels per batch.
2. OHEM normalized to have same magnitude as default loss

Improvement proposals:
1. Harder augmentations
2. Dropout2d(0.1) after scales merge (0.2 too)
3. Dropout for classificator (looks like it not hurts perfomance but slows down training, may improve final perfomance)
4. UNet++
5. Classic OHEM loss with threshold 0.7 and small min_keep value around 0.1-0.01
6. Group norm (SUCCESS with g=8 but without pretrained model trains very slowly)
7. Upsampling instead of transposed conv (SUCCESS)
8. Medium augmentations (SUCCESS have to train one more epochs round)
9. Lovasz
10. Log Dice loss
11. Boundary F1 Loss
12. IC layer (Relu BN Dropout Conv) (FAILURE)
13. Take not sum but mean of mask bce and dice losses
14. [Use without experiment] SWA
15. Relu for resnet features
16. Image normalization (as for imagenet)

What should I use in my model:
1. Dropout2d(0.1) after concatenation with lower level features
2. Upsampling with bilinear interpolation
3. SWA
4. No Relu for backbone resnet outputs
5. Log dice with gamma=0.3