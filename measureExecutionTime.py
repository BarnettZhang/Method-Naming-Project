from distutils.command.config import config
from transformers import RobertaTokenizer
import torch
import dill
import sys
from time import time
import os.path
import zipfile
from CodeBERTaModel import CodeBERTaEncoderDecoder
from Config import Config
import time


def measureExecutionTime(src_path, selected_code):
    # Check if model is unzipped, if not, unzip the model
    t0 = time.time()
    if not os.path.isfile(src_path + '/OriginalFullDatasetState'):
        with zipfile.ZipFile(src_path + '/OriginalFullDatasetState.zip', 'r') as zip_ref:
            zip_ref.extractall(src_path)
        os.remove(src_path + '/OriginalFullDatasetState.zip')
    t1 = time.time()

    # Load the model to CPU
    device = torch.device("cpu")
    config = Config()
    model = CodeBERTaEncoderDecoder(config,device)
    model.load_state_dict(torch.load(src_path + '/LayerDropState', pickle_module=dill, map_location='cpu'), strict=False)

    t2 = time.time()

    # Tokenize the selected code and make the prediction using the model. 
    # Adapted from the final project of Ugo Benassayag, https://github.com/UgoBena/Sourcery_Project
    selected_code_raw = selected_code
    tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
    selected_code = tokenizer.batch_encode_plus([selected_code])["input_ids"]
    underscore_token = tokenizer.get_vocab()["_"]
    selected_code = [[token for token in input if token != underscore_token][:tokenizer.model_max_length] for input in selected_code]
    evaluate_input = [(selected_code[0], [0, 883, 19603, 8588, 2], len(selected_code[0]), 5, selected_code_raw[0])]
    top_sequence,top_length,tgt_seqs,tgt_lens,output_prob,decoded_sequences,inputs_raw = model.evaluate(evaluate_input)

    t3 = time.time()

    outputStr = "Time taken to unzip the model state: " + str(t1 - t0) + 's\n' + "Time taken to load the model: " + str(t2 - t1) + 's\n' + "Time taken to make the prediction: " + str(t3 - t2) + 's\n'
    return outputStr

if __name__ == "__main__" :
    # Print the prediction output, extension.js will collect the stdout and display it on the msg.
    measureTime = measureExecutionTime(sys.argv[1], sys.argv[2])
    print(measureTime)
    sys.stdout.flush()