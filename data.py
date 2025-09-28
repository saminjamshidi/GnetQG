# data.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import mask_entity

class CustomDataset(Dataset):
    """Dataset wrapper for graph-based QG samples."""
    def __init__(self, data_list, tokenizer, bart_model, device, desired_size=80):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.bart_model = bart_model
        self.device = device
        self.desired_size = desired_size

    def get_embedding(self, texts):
        inputs = [self.tokenizer(t, return_tensors="pt", max_length=1024,
                                 truncation=True)["input_ids"][0].to(self.device)
                  for t in texts]
        max_len = max(t.size(0) for t in inputs)
        padded = [F.pad(t, (0, max_len - t.size(0)), value=0) for t in inputs]
        padded_tensor = torch.stack(padded)
        with torch.no_grad():
            outputs = self.bart_model.model.encoder(padded_tensor)
        return outputs.last_hidden_state.mean(dim=1)

    def get_context_embedding(self, context):
        inputs = self.tokenizer(context, return_tensors="pt", max_length=1024,
                                truncation=True)["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.bart_model.model.encoder(inputs)
        return outputs.last_hidden_state[0]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        document, answer, question = data['document'], data['answer'], data['Question']
        context = f"document : {document} answer : {answer} question entity : "
        adjacency_matrix = data['Adg']
        entity_lengths = data['entity_lenghts']
        sample_id = data['Id']
        mask = data['mask']
        entity_spans = data['entity_spans']

        entity_texts = [e[2] for e in entity_spans]
        entities = self.get_embedding(entity_texts)

        if len(entities) > self.desired_size:
            padded_entities = entities[:self.desired_size]
            mask_q = mask_entity(entity_texts[:self.desired_size], question)
        else:
            pad_needed = self.desired_size - entities.size(0)
            padded_entities = F.pad(entities, (0, 0, 0, pad_needed))
            mask_q = mask_entity(entity_texts, question)

        context_emb = self.get_context_embedding(context)

        return {
            'document': document,
            'answer': answer,
            'context': context,
            'question': question,
            'entity_spans': padded_entities,
            'adjacency_matrix': adjacency_matrix,
            'entity_lengths': entity_lengths,
            'id': sample_id,
            'mask': mask,
            'mask_q': mask_q,
            'entitys': entity_texts,
            'context_emb': context_emb
        }


def custom_collate_fn(batch):
    """Pad and collate a batch of samples."""
    max_len = max(sample['entity_lengths'] for sample in batch)

    return {
        'entity_spans': torch.stack([s['entity_spans'] for s in batch]),
        'document': [s['document'] for s in batch],
        'context': [s['context'] for s in batch],
        'answer': [s['answer'] for s in batch],
        'question': [s['question'] for s in batch],
        'id': [s['id'] for s in batch],
        'adjacency_matrix': torch.stack([s['adjacency_matrix'] for s in batch]),
        'mask': torch.stack([s['mask'] for s in batch]),
        'mask_q': torch.stack([s['mask_q'] for s in batch]),
        'entity_lengths': max_len,
        'entitys': [s['entitys'] for s in batch]
    }
