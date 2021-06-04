import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer, BertweetTokenizer


# Tokenization
def convert_data_to_ids(tokenizer, target, text):
    
    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []
    for tar, sent in zip(target, text):
        encoded_dict = tokenizer.encode_plus(
                            ' '.join(tar),                  # Target to encode
                            ' '.join(sent),                 # Sentence to encode
                            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'
                            max_length = 128,               # Pad & truncate all sentences
                            padding = 'max_length',
                            return_attention_mask = True,   # Construct attention masks
                       )

        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))
    
    return input_ids, seg_ids, attention_masks, sent_len
    

# BERT/BERTweet tokenizer
def data_helper_bert(x_train_all,x_val_all,x_test_all,model_select):
    
    print('Loading data')
    
    x_train,y_train,x_train_target = x_train_all[0],x_train_all[1],x_train_all[2]                                                
    x_val,y_val,x_val_target = x_val_all[0],x_val_all[1],x_val_all[2]
    x_test,y_test,x_test_target = x_test_all[0],x_test_all[1],x_test_all[2]
                                                         
    print("Length of x_train: %d, the sum is: %d"%(len(x_train), sum(y_train)))
    print("Length of x_val: %d, the sum is: %d"%(len(x_val), sum(y_val)))
    print("Length of x_test: %d, the sum is: %d"%(len(x_test), sum(y_test)))
    
    # get the tokenizer
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        
    # tokenization
    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = \
                    convert_data_to_ids(tokenizer, x_train_target, x_train)
    x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len = \
                    convert_data_to_ids(tokenizer, x_val_target, x_val)
    x_test_input_ids, x_test_seg_ids, x_test_atten_masks, x_test_len = \
                    convert_data_to_ids(tokenizer, x_test_target, x_test)

    x_train_all = [x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_len]
    x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len]
    x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_len]
    
    return x_train_all,x_val_all,x_test_all


def data_loader(x_all, batch_size, data_type):
    
    x_input_ids = torch.tensor(x_all[0], dtype=torch.long).cuda()
    x_seg_ids = torch.tensor(x_all[1], dtype=torch.long).cuda()
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    y = torch.tensor(x_all[3], dtype=torch.long).cuda()
    x_len = torch.tensor(x_all[4], dtype=torch.long).cuda()

    tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len)
    if data_type == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

    return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader
    

def sep_test_set(input_data):
    
    # split the combined test set for Trump, Biden and Bernie
    data_list = [input_data[:841], input_data[841:1657], input_data[1657:2374]]
    
    return data_list