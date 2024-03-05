from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from transformers import BertTokenizer
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator
from transformers import AutoTokenizer

from captum.attr import visualization
import torch
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").to("cuda")
model.eval()
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
# initialize the explanations generator
explanations = Generator(model)

classifications = ["NEGATIVE", "POSITIVE"]
# encode a sentence
text_batch = ["This movie was the best movie I have ever seen! some scenes were ridiculous, but acting was great."]
encoding = tokenizer(text_batch, return_tensors='pt')
input_ids = encoding['input_ids'].to("cuda")
attention_mask = encoding['attention_mask'].to("cuda")

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
if class_name == "NEGATIVE":
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