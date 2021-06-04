import torch.nn as nn
from transformers import AutoModel, BertModel


# BERT/BERTweet
class stance_classifier(nn.Module):

    def __init__(self,num_labels,model_select):

        super(stance_classifier, self).__init__()
        
        self.dropout = nn.Dropout(0.)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        if model_select == 'Bertweet':
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")
        elif model_select == 'Bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, x_input_ids, x_seg_ids, x_atten_masks, x_len):
        
        last_hidden = self.bert(input_ids=x_input_ids, \
                                attention_mask=x_atten_masks, token_type_ids=x_seg_ids, \
                               )
        
        query = last_hidden[0][:,0]
        query = self.dropout(query)
        
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out