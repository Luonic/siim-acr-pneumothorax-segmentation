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
This task should be separated to two different tasks:
1. Classification - has pneumotorax or not
2. If has - Predict semantic label map

This can be solved with single model by applying semantic label map loss only if sample has mask

Loss proposals:
1. OHEM but for each gt mask pixels separately. min_keep is num of 
elements for class with min pixels per batch.
2. OHEM normalized to have same magnitude as default loss