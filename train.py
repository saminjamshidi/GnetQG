# train.py
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AdamW
import pickle
from types import SimpleNamespace

from data import CustomDataset, custom_collate_fn
from model import CustomFineTune
from utils import freeze_encoder, print_and_flush
from eval import check_score

def main(model_path, train_indices, val_indices, test_indices, full_dataset, device):
    config = SimpleNamespace(gnn_drop=0.5, q_attn=False, hidden_dim=8,
                             n_type=4, q_update=False, input_dim=20)
    in_dim, out_dim, n_head = 1024, 1024, 4

    model = CustomFineTune(in_dim, out_dim, n_head, config, device).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_dataset = CustomDataset(full_dataset, model.tokenizer, model.bart_model, device)
    train_loader = DataLoader(train_dataset, batch_size=16,
                              sampler=SubsetRandomSampler(train_indices),
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(train_dataset, batch_size=16,
                            sampler=SubsetRandomSampler(val_indices),
                            collate_fn=custom_collate_fn)
    test_loader = DataLoader(train_dataset, batch_size=16,
                             sampler=SubsetRandomSampler(test_indices),
                             collate_fn=custom_collate_fn)

    best_rouge = 0.42
    num_epochs = 10
    print_and_flush("Start training...\n")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            for k in ['mask','adjacency_matrix','mask_q','entity_spans']:
                batch[k] = batch[k].to(device)
            outputs = model(batch['entity_spans'], batch['adjacency_matrix'], batch['mask'],
                            batch['question'], batch['entitys'], batch['context'],
                            batch['entity_lengths'], train=True)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        preds, refs = [], []
        with torch.no_grad():
            for batch in val_loader:
                for k in ['mask','adjacency_matrix','mask_q','entity_spans']:
                    batch[k] = batch[k].to(device)
                out = model(batch['entity_spans'], batch['adjacency_matrix'], batch['mask'],
                            batch['question'], batch['entitys'], batch['context'],
                            batch['entity_lengths'], train=False)
                preds.extend(out)
                refs.extend(batch['question'])

        rouge_score = check_score(preds, refs)
        print_and_flush(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss/len(train_loader):.4f} | ROUGE: {rouge_score:.4f}\n")

        if rouge_score > best_rouge:
            best_rouge = rouge_score
            torch.save(model.state_dict(), f"{model_path}/best_model_gatbart.pth")
            print_and_flush("Saved new best model.\n")

    print_and_flush("Training complete.\n")

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "..."  # change to your directory

    with open(f"{model_path}/concate_data_2.pkl","rb") as f:
        full_dataset = pickle.load(f)
    with open(f"{model_path}/train_data_v10.pkl","rb") as f:
        train_indices = pickle.load(f)
    with open(f"{model_path}/validation_data_v10.pkl","rb") as f:
        val_indices = pickle.load(f)
    with open(f"{model_path}/test_data_v10.pkl","rb") as f:
        test_indices = pickle.load(f)

    main(model_path, train_indices, val_indices, test_indices, full_dataset, device)
