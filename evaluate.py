from copyreg import pickle
from transformers import RobertaModel, RobertaTokenizer, EncoderDecoderModel, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import ast
import torch
import re
from torchtext.legacy.data import BucketIterator
from CodeBERTaModel import CodeBERTaEncoderDecoder
from Config import Config
import sys
import dill


def get_topK_metrics(tgt_seqs,topk_sequence,tgt_lens,topk_length):
    k = topk_length.size(1)
    batch_size = tgt_seqs.size(0)
    #get numpy arrays
    tgt_seqs = tgt_seqs.cpu().data.numpy()    
    topk_sequence = topk_sequence.cpu().data.numpy()
    topk_length = topk_length.cpu().data.numpy()
    
    #metrics to compute
    top1_f1 = 0
    top1_acc = 0
    topK_acc = 0
    topK_f1 = 0
    #loop: for each prediction, different pred_len and tgt_len make vectorized computation impossible
    for i in range(batch_size):
        tgt = tgt_seqs[i,1:tgt_lens[i].item()-1]
        best_acc = 0
        best_f1 = 0
        for j in range(k):
            pred = topk_sequence[i,j,1:topk_length[i,j]-1]

            tp = float((np.isin(pred,tgt)*1).sum())
            fp = float((np.isin(pred,tgt,invert=True)*1).sum())
            fn = float((np.isin(tgt,pred,invert=True)*1).sum())

            #Precision
            if (tp + fp != 0.): 
              precision = tp/(tp + fp)
            else: 
              precision = 0
            #Recall
            if (tp + fn != 0.): 
              recall = tp/(tp + fn)
            else: 
              recall = 0
            #Acc
            acc = (fp==0. and fn==0.) * 1.
            #F1
            if precision + recall != 0.:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.
            
            #record top1 value
            if j==0:
                top1_acc += acc
                top1_f1 += f1

            #keep best of K values
            if f1>best_f1:
                best_f1 = f1
            if acc>best_acc:
                best_acc = acc

        #add best values to topK metrics
        topK_acc += best_acc
        topK_f1 += best_f1
            

    #average values
    top1_acc /= batch_size
    top1_f1 /= batch_size
    topK_acc /= batch_size
    topK_f1 /= batch_size
    return top1_acc,top1_f1,topK_acc,topK_f1

def evaluate_full_dataset(val_dataloader,model):
    val_dataloader.create_batches()
    total_top1_acc = 0
    total_top1_f1 = 0
    total_topK_acc = 0
    total_topK_f1 = 0
    nb_eval = len(val_dataloader)
    for batch in tqdm(val_dataloader.batches):
        topk_sequence,topk_length,tgt_seqs,tgt_lens,output_prob,decoded_sequences,_ = model.evaluate(batch)

        top1_acc,top1_f1,topK_acc,topK_f1 = get_topK_metrics(tgt_seqs,topk_sequence,tgt_lens,topk_length)

        total_top1_acc += top1_acc
        total_top1_f1 += top1_f1
        total_topK_acc += topK_acc
        total_topK_f1 += topK_f1

    #avg values
    total_top1_acc /= nb_eval
    total_top1_f1 /= nb_eval
    total_topK_acc /= nb_eval
    total_topK_f1 /= nb_eval

    
    return total_top1_acc,total_top1_f1,total_topK_acc,total_topK_f1

class FunctionNamingDataset(Dataset):
    def __init__(self,data_pairs,inputs_raw):
        self.pairs = data_pairs
        self.inputs_raw = inputs_raw
        self.n_examples = len(self.pairs)
    
    def __len__(self):
        r"""When used `len` return the number of examples.
        """

        return self.n_examples


    def __getitem__(self, item):
        r"""Given an index return a pair of input output
        """
        input,output = self.pairs[item]
        input_raw = self.inputs_raw[item]
        return (input,output,len(input),len(output),input_raw)

def get_func_and_name(data):
    try:
        node = ast.parse(data).body[0]
        function_name = node.name
        function = data
        docstring = ast.get_docstring(node)
        #remove docstring
        if docstring is not None:
            function = re.sub(r'\"\"\"(.*)\"\"\"',"",function,count=1,flags=re.DOTALL)
        #remove function name
        function = re.sub(function_name,"<mask>",function,count=1)
        return function,function_name
    except:
        return None

if __name__ == "__main__":
    device = torch.device("cpu")
    with open("C:/Users/zhang/PycharmProjects/Sourcery_Project/python/python/final/jsonl/train/python_train_0.jsonl") as f:
        jsonl_content = f.readlines()

    train_jsons = [json.loads(json_line) for json_line in jsonl_content]


    with open("C:/Users/zhang/PycharmProjects/Sourcery_Project/python/python/final/jsonl/valid/python_valid_0.jsonl") as f:
        jsonl_content = f.readlines()
    #divide valid size by 10
    jsonl_content = jsonl_content[:int(len(jsonl_content)/10)]
    val_jsons = [json.loads(json_line) for json_line in jsonl_content]

    training_pairs_raw = [get_func_and_name(line["code"]) for line in train_jsons if get_func_and_name(line["code"]) is not None]

    training_inputs_raw = [x for (x,y) in training_pairs_raw]
    training_labels_raw = [y for (x,y) in training_pairs_raw]

    val_pairs_raw = [get_func_and_name(line["code"]) for line in val_jsons if get_func_and_name(line["code"]) is not None]

    val_inputs_raw = [x for (x,y) in val_pairs_raw]
    val_labels_raw = [y for (x,y) in val_pairs_raw]

    tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")

    PAD_token = tokenizer.pad_token_id
    EOS_token = tokenizer.eos_token_id
    BOS_token = tokenizer.bos_token_id

    training_inputs = tokenizer.batch_encode_plus(training_inputs_raw)["input_ids"]
    training_labels = tokenizer.batch_encode_plus(training_labels_raw)["input_ids"]

    val_inputs = tokenizer.batch_encode_plus(val_inputs_raw)["input_ids"]
    val_labels = tokenizer.batch_encode_plus(val_labels_raw)["input_ids"]


    #Remove underscore tokens from inputs and labels, and truncate up to max model length
    underscore_token = tokenizer.get_vocab()["_"]

    training_inputs = [[token for token in input if token != underscore_token][:tokenizer.model_max_length] for input in training_inputs]
    training_labels = [[token for token in input if token != underscore_token] for input in training_labels]

    val_inputs = [[token for token in input if token != underscore_token][:tokenizer.model_max_length] for input in val_inputs]
    val_labels = [[token for token in input if token != underscore_token] for input in val_labels]



    training_pairs = list(zip(training_inputs,training_labels))
    validation_pairs = list(zip(val_inputs,val_labels))

    train_dataset = FunctionNamingDataset(training_pairs,training_inputs_raw)
    val_dataset = FunctionNamingDataset(validation_pairs,val_inputs_raw)


    train_batch_size = 16
    valid_batch_size = 16

    train_dataloader,val_dataloader = BucketIterator.splits(
        
                            # Datasets for iterator to draw data from
                            (train_dataset,val_dataset),

                            # Tuple of train and validation batch sizes.
                            batch_sizes=(train_batch_size,valid_batch_size),

                            # Device to load batches on.
                            device=device, 

                            # Function to use for sorting examples.
                            sort_key=lambda x: x[2],


                            # Repeat the iterator for multiple epochs.
                            repeat=True, 

                            # Sort all examples in data using `sort_key`.
                            sort=False, 

                            # Shuffle data on each epoch run.
                            shuffle=True,

                            # Use `sort_key` to sort examples in each batch.
                            sort_within_batch=True,
                            )
    config = Config()
    model = CodeBERTaEncoderDecoder(config,device)
    model.load_state_dict(torch.load(sys.argv[1], pickle_module=dill, map_location='cpu'), strict=False)
    val_top1_acc,val_top1_f1,val_topK_acc,val_topK_f1= evaluate_full_dataset(val_dataloader,model)
    print('- Val Top-1 Accuracy: {}'.format(val_top1_acc))
    print('- Val Top-1 F1 Score: {}'.format(val_top1_f1))
    print('- Val Top-K Accuracy: {}'.format(val_topK_acc))
    print('- Val Top-K F1 Score: {}'.format(val_topK_f1))