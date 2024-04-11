import torch

dict = torch.load('D:\\Bert_ex\\Transformer-Explainability\\bert_models\\movies\\classifier\\classifier_epoch_data.pt')

print(dict['epoch'])

print(dict['results'])

print(dict['best_val_acc'])

print(dict['done'])