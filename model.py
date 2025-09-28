# model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from GAT import AttentionLayer  # assumes GAT.py is available

class BinaryClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, n_head, config, seq_len=80):
        super().__init__()
        self.gat_layer = AttentionLayer(in_dim, out_dim, n_head, config)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(seq_len * out_dim, seq_len)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, adjacency_matrix, masks):
        x = self.gat_layer(node_features, adjacency_matrix, masks)
        x = self.flatten(x)
        x = self.fc(x)
        return self.sigmoid(x)


class CustomFineTune(nn.Module):
    def __init__(self, in_dim, out_dim, n_head, config, device):
        super().__init__()
        self.device = device
        self.model_gat = BinaryClassifier(in_dim, out_dim, n_head, config)
        self.tokenizer = AutoTokenizer.from_pretrained("p208p2002/bart-squad-qg-hl")
        self.bart_model = AutoModelForSeq2SeqLM.from_pretrained("p208p2002/bart-squad-qg-hl")

    def forward(self, entity_spans, adjacency_matrix, masks,
                questions, entitys, context, entity_lengths, train=True):

        target_ids = self.tokenizer(
            questions, return_tensors="pt", max_length=1024,
            padding=True, truncation=True
        )["input_ids"].to(self.device)

        gat_out = self.model_gat(entity_spans, adjacency_matrix, masks)
        binary_output = (gat_out > 0.5).int()

        for k in range(masks.size(0)):
            indices = torch.nonzero(binary_output[k][:entity_lengths]).squeeze().tolist()
            if isinstance(indices, int):
                indices = [indices]
            if len(indices) > 0:
                selected_entities = [entitys[k][i] for i in indices if i < len(entitys[k])]
                context[k] += " " + " ".join(selected_entities)

        input_ids = self.tokenizer(
            context, return_tensors="pt", max_length=1024,
            padding=True, truncation=True
        )["input_ids"].to(self.device)

        if train:
            outputs = self.bart_model(input_ids=input_ids, labels=target_ids)
            return outputs
        else:
            gen = self.bart_model.generate(
                input_ids=input_ids, max_length=50, num_beams=4,
                early_stopping=True, num_return_sequences=1
            )
            return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
