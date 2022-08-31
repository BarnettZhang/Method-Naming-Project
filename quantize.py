import numpy as np
import dill
import torch
from Config import Config
from evaluate import CodeBERTaEncoderDecoder

if __name__ == "__main__" :
    device = torch.device("cpu")
    config = Config()
    model = CodeBERTaEncoderDecoder(config,device)
    model.load_state_dict(torch.load('./OriginalFullDatasetState', pickle_module=dill, map_location='cpu'), strict=False)
    # print(model)
    model.device = torch.device('cpu')
    # model.model.encoder.encoder.layer[0]
    for i in range(5):
        torch.quantization.quantize_dynamic(model.model.encoder.encoder.layer[i], {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
    print(model)
    torch.save(model.state_dict(), 'QuantizeStateINT8Layer12345', pickle_module=dill)