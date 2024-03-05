from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from transformers import AutoTokenizer

from captum.attr import visualization
import torch

# class BERTClass(torch.nn.Module):
#     def __init__(self):
#         super(BERTClass, self).__init__()
#         self.bert_model = AutoModel.from_pretrained('jackaduma/SecBERT')
#         self.dropout = torch.nn.Dropout(0.3)
#         self.linear = torch.nn.Linear(768, 7)

#     def forward(self, input_ids, attention_mask, token_type_ids):
#         output = self.bert_model(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids
#         )
#         output_dropout = self.dropout(output.pooler_output)
#         output = self.linear(output_dropout)
#         return output
#     def relprop(self, cam=None, **kwargs):
#         cam = self.classifier.relprop(cam, **kwargs)
#         cam = self.dropout.relprop(cam, **kwargs)
#         cam = self.bert.relprop(cam, **kwargs)
#         # print("conservation: ", cam.sum())
#         return cam
    
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
from transformers import BertTokenizer, BertModel

# checkpoint = torch.load('/data/AI/backup/namlt/namlt/Hidden/Transformer-Explainability/best_model.pt')
# model = BERTClass()
# # model.load_state_dict(torch.load('/kaggle/working/best_model.pt'))
# model.load_state_dict(checkpoint['state_dict'])
# model.eval()
model = BertForSequenceClassification.from_pretrained("jackaduma/SecBERT").to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')
# initialize the explanations generator


classifications = ["000 - Normal", "272 - Protocol Manipulation",
            "88 - OS Command Injection", "126 - Path Traversal", "66 - SQL Injection",
            "310 - Scanning for Vulnerable Software",
            "194 - Fake the Source of Data"]
model.to('cuda')
explanations = Generator(model)
# encode a sentence
text_batch = ["GET /blog/index.php/2020/04/04/voluptatum-reprehenderit-maiores-ab-sequi-quaerat/"]
encoding = tokenizer(text_batch, return_tensors='pt')
input_ids = encoding['input_ids'].to("cuda")
attention_mask = encoding['attention_mask'].to("cuda")
token_type_ids = encoding['token_type_ids'].to("cuda")

# true class is positive - 1
true_class = 1

# generate an explanation for the input
expl = explanations.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
# normalize scores
expl = (expl - expl.min()) / (expl.max() - expl.min())

# get the model classification
output = torch.nn.functional.softmax(model(input_ids=input_ids, attention_mask=attention_mask)[0], dim=-1)
classification = output.argmax(dim=-1).item()
# get class name
class_name = classifications[classification]
# if the classification is negative, higher explanation scores are more negative
# flip for visualization
if class_name == "272 - Protocol Manipulation":
  expl *= (-1)

tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
print([(tokens[i], expl[i].item()) for i in range(len(tokens))])
vis_data_records = [visualization.VisualizationDataRecord(
                                expl,
                                output[0][classification],
                                classification,
                                true_class,
                                true_class,
                                1,
                                tokens,
                                1)]
visualization.visualize_text(vis_data_records)
