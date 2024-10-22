import os
import json
from typing import Tuple, List
from matplotlib import pyplot as plt
import tqdm  # type: ignore
import numpy as np

from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, ConfusionMatrixDisplay, confusion_matrix  # type: ignore
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pytorch_warmup as warmup
import optuna

from utils import seed_everything
from constants import (
    Label, NUM_MLP_EPOCHS, TRAINING_METRICS_DIR_PATH, DATASET_SPLITS_FILE_PATH, AI_EMBEDDINGS_DIR_PATH,
    HUMAN_EMBEDDINGS_DIR_PATH, CLAP_EMBEDDING_SIZE
)


def load_audio_embeddings(audio_embeddings_file_path: str) -> torch.Tensor:
    return torch.tensor(np.load(audio_embeddings_file_path))


class LabeledAudioFilePath(BaseModel):
    audio_file_path: str
    label: int


class AudioDataset(Dataset):
    def get_embedding_file_path(self, labeled_audio_file_path: LabeledAudioFilePath) -> str:
        audio_file_name = os.path.basename(labeled_audio_file_path.audio_file_path)
        audio_file_name_prefix = os.path.splitext(audio_file_name)[0]
        if labeled_audio_file_path.label == Label.AI.value:
            embedding_dir_path = AI_EMBEDDINGS_DIR_PATH
        else:
            embedding_dir_path = HUMAN_EMBEDDINGS_DIR_PATH
        return os.path.join(embedding_dir_path, f'{audio_file_name_prefix}.npy')

    def __init__(self, labeled_audio_file_paths: List[LabeledAudioFilePath], device: torch.device) -> None:
        # Load all audio embeddings, each can have a different number of time steps
        self.audios_embeddings = [load_audio_embeddings(self.get_embedding_file_path(labeled_audio_file_path)).to(device)
                                  for labeled_audio_file_path in labeled_audio_file_paths]
        # Load labels
        self.labels = torch.tensor(
            [labeled_audio_file_path.label for labeled_audio_file_path in labeled_audio_file_paths],
            dtype=torch.float,
            device=device,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return embeddings and labels
        return self.audios_embeddings[idx], self.labels[idx].view(1)

    # Custom collate_fn to handle padding
    def collate_fn(self, batch):
        # Separate the batch into embeddings and labels
        embeddings, labels = zip(*batch)
        
        # Pad the sequences to the length of the longest sequence in the batch
        embeddings_padded = pad_sequence(embeddings, batch_first=True)  # (batch_size, max_seq_len, embedding_dim)
        
        # Stack the labels (no padding needed for labels)
        labels_tensor = torch.stack(labels)
        
        return embeddings_padded, labels_tensor

class PositionalEncoding(nn.Module):
       
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        import math

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerwithMLP(nn.Module):
    def __init__(self, input_size: int, num_heads: int, dim_ffn: int) -> None:
        super(TransformerwithMLP, self).__init__()
        self.nhead = num_heads
        self.ffn_dim = dim_ffn
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=self.nhead, batch_first=True, dim_feedforward=self.ffn_dim ,norm_first=True ,dropout=0.1
        )
        self.linear = nn.Sequential(
            nn.Linear(input_size, 1)
        )
        self.linear2 =nn.Sequential(nn.Linear(input_size, input_size))
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(input_size)
        self.positional_encodings = self.generate_sinusoidal_positional_encodings(100, input_size)

    def generate_sinusoidal_positional_encodings(self, max_len: int, d_model: int):
        """Generates sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        return pe

    def forward(self, batch_embeddings: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # batch_embeddings shape: (Batch, Seq_Length, 512)
        #transformer_output = self.layernorm(self.transformer_layer(batch_embeddings))  # (Batch, Seq_Length, 512)
        #pos_enc = self.positional_encodings[:, :batch_embeddings.shape[1], :].to(batch_embeddings.device)
        #batch_embeddings += pos_enc
        transformer_output = self.layernorm(self.transformer_layer(self.linear2(batch_embeddings))) 
        #print("Transformer output", transformer_output.shape)
        transformer_output_flat = transformer_output.mean(dim=1)  # (Batch, 512)
        #transformer_cls=transformer_output[:,0,:]
        return self.sigmoid(self.linear(transformer_output_flat)), transformer_output_flat



def objective(trial):
        seed_everything(42)

        
        with open(DATASET_SPLITS_FILE_PATH) as f:
            dataset_splits = json.load(f)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        train_dataset = AudioDataset(
            labeled_audio_file_paths=[
                LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
                for labeled_audio_file_path_dict in dataset_splits['train']
            ],
            device=device,
        )
        val_dataset = AudioDataset(
            labeled_audio_file_paths=[
                LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
                for labeled_audio_file_path_dict in dataset_splits['val']
            ],
            device=device,
        )
        test_dataset = AudioDataset(
            labeled_audio_file_paths=[
                LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
                for labeled_audio_file_path_dict in dataset_splits['test']
            ],
            device=device,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=val_dataset.collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn)
        # Suggest values for hyperparameters
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        T_max = trial.suggest_int('T_max', 10, 50)
        eta_min = trial.suggest_loguniform('eta_min', 1e-6, 1e-3)
        warmup_period = trial.suggest_int('warmup_period', 1, 10)
        dim_ffn = int(trial.suggest_loguniform('dim_ffn', 128, 1024))
        #num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        num_heads = 2 ** trial.suggest_int('num_heads', 0, 3)
        
        # Initialize model
        model = TransformerwithMLP(input_size=CLAP_EMBEDDING_SIZE, num_heads=num_heads, dim_ffn=dim_ffn)
        model.to(device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warmup_period)  
        # Learning rate scheduler with cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        
        train_losses = []
        val_losses = []
        train_accuracies = [0]
        val_accuracies = [0]

        # Training loop
        for epoch in range(NUM_MLP_EPOCHS):
                model.train()
                epoch_train_loss = 0
                for X_batch, y_batch in train_dataloader:
                    optimizer.zero_grad()
                    y_pred, _ = model(X_batch)  # torch.Size([32, seq_len, 512])
                    loss = criterion(y_pred, y_batch)  # Adjust loss function for the correct shape
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()

                train_losses.append(epoch_train_loss / len(train_dataloader))

                # Compute train accuracy
                model.eval()
                y_train_list = []
                y_pred_train_list = []
                with torch.no_grad():
                    correct_cnt = 0
                    total_cnt = 0
                    for X_batch, y_batch in train_dataloader:
                        y_pred = torch.round(model(X_batch)[0])
                        y_train_list.extend(y_batch.cpu().numpy())
                        y_pred_train_list.extend(y_pred.cpu().numpy())
                        correct_cnt += (y_pred == y_batch).sum().item()
                        total_cnt += len(y_batch)

                train_accuracy = correct_cnt / total_cnt
                train_accuracies.append(train_accuracy)
                y_train = np.array(y_train_list)
                y_pred_train = np.array(y_pred_train_list)

                # Validation loop
                epoch_val_loss = 0
                y_val_list = []
                y_pred_val_list = []
                with torch.no_grad():
                    correct_cnt = 0
                    total_cnt = 0
                    for X_batch, y_batch in val_dataloader:
                        y_pred = torch.round(model(X_batch)[0])
                        loss = criterion(y_pred, y_batch)
                        epoch_val_loss += loss.item()
                        correct_cnt += (y_pred == y_batch).sum().item()
                        total_cnt += len(y_batch)
                        y_val_list.extend(y_batch.cpu().numpy())
                        y_pred_val_list.extend(y_pred.cpu().numpy())

                y_val = np.array(y_val_list)
                y_pred_val = np.array(y_pred_val_list)
                val_losses.append(epoch_val_loss / len(val_dataloader))
                print(val_losses)
                scheduler.step()

                val_accuracy = correct_cnt / total_cnt
                val_accuracies.append(val_accuracy)

                # Print out performance metrics
                print(f'Epoch: {epoch}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f} '
                    f'Train F1 Score: {f1_score(y_train, y_pred_train):.4f} Train AUC: {roc_auc_score(y_train, y_pred_train):.4f}')

                # Testing
                model.eval()
                y_test_list = []
                y_pred_test_list = []
                with torch.no_grad():
                    correct_cnt = 0
                    total_cnt = 0
                    for X_batch, y_batch in test_dataloader:
                        y_pred = torch.round(model(X_batch)[0])
                        correct_cnt += (y_pred == y_batch).sum().item()
                        total_cnt += len(y_batch)
                        y_test_list.extend(y_batch.cpu().numpy())
                        y_pred_test_list.extend(y_pred.cpu().numpy())

                y_test = np.array(y_test_list)
                y_pred_test = np.array(y_pred_test_list)

                accuracy = correct_cnt / total_cnt
                print(f'\nTesting: Test Accuracy: {accuracy:.4f} Test F1 Score: {f1_score(y_test, y_pred_test):.4f} Test AUC: {roc_auc_score(y_test, y_pred_test):.4f}')



        # The objective to minimize (validation loss)
        return np.min(val_losses)



def main() -> None:
    seed_everything(42)

    
    with open(DATASET_SPLITS_FILE_PATH) as f:
        dataset_splits = json.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = AudioDataset(
        labeled_audio_file_paths=[
            LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
            for labeled_audio_file_path_dict in dataset_splits['train']
        ],
        device=device,
    )
    val_dataset = AudioDataset(
        labeled_audio_file_paths=[
            LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
            for labeled_audio_file_path_dict in dataset_splits['val']
        ],
        device=device,
    )
    test_dataset = AudioDataset(
        labeled_audio_file_paths=[
            LabeledAudioFilePath.model_validate(labeled_audio_file_path_dict)
            for labeled_audio_file_path_dict in dataset_splits['test']
        ],
        device=device,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=val_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn)

    #model = TransformerwithMLP(input_size=CLAP_EMBEDDING_SIZE, num_heads=8, dim_ffn=512) # MINE
    model = TransformerwithMLP(input_size=CLAP_EMBEDDING_SIZE, num_heads=8, dim_ffn=408)
    #model.fit_scaler(train_dataset.audios_embeddings)
    model.to(device)

    criterion = nn.BCELoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01) # MINE
    optimizer = torch.optim.AdamW(model.parameters(), lr= 0.004255481529466004, weight_decay=1.9677602668675473e-06)
    #warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=5) # MINE
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=2)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6) # MINE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=31, eta_min=0.00017337568045500368)


    train_losses = []
    val_losses = []
    train_accuracies = [0]
    val_accuracies = [0]


    for epoch in range(NUM_MLP_EPOCHS):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred, _ = model(X_batch)  # torch.Size([32, seq_len, 512])
            loss = criterion(y_pred, y_batch)  # Adjust loss function for the correct shape
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_dataloader))

        # Compute train accuracy
        model.eval()
        y_train_list = []
        y_pred_train_list = []
        with torch.no_grad():
            correct_cnt = 0
            total_cnt = 0
            for X_batch, y_batch in train_dataloader:
                y_pred = torch.round(model(X_batch)[0])
                y_train_list.extend(y_batch.cpu().numpy())
                y_pred_train_list.extend(y_pred.cpu().numpy())
                correct_cnt += (y_pred == y_batch).sum().item()
                total_cnt += len(y_batch)

        train_accuracy = correct_cnt / total_cnt
        train_accuracies.append(train_accuracy)
        y_train = np.array(y_train_list)
        y_pred_train = np.array(y_pred_train_list)

        # Validation loop
        epoch_val_loss = 0
        y_val_list = []
        y_pred_val_list = []
        with torch.no_grad():
            correct_cnt = 0
            total_cnt = 0
            for X_batch, y_batch in val_dataloader:
                y_pred = torch.round(model(X_batch)[0])
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item()
                correct_cnt += (y_pred == y_batch).sum().item()
                total_cnt += len(y_batch)
                y_val_list.extend(y_batch.cpu().numpy())
                y_pred_val_list.extend(y_pred.cpu().numpy())

        y_val = np.array(y_val_list)
        y_pred_val = np.array(y_pred_val_list)
        val_losses.append(epoch_val_loss / len(val_dataloader))
        print(val_losses)
        scheduler.step()

        val_accuracy = correct_cnt / total_cnt
        val_accuracies.append(val_accuracy)

        # Print out performance metrics
        print(f'Epoch: {epoch}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f} '
            f'Train F1 Score: {f1_score(y_train, y_pred_train):.4f} Train AUC: {roc_auc_score(y_train, y_pred_train):.4f}')

        # Testing
        model.eval()
        y_test_list = []
        y_pred_test_list = []
        with torch.no_grad():
            correct_cnt = 0
            total_cnt = 0
            for X_batch, y_batch in test_dataloader:
                y_pred = torch.round(model(X_batch)[0])
                correct_cnt += (y_pred == y_batch).sum().item()
                total_cnt += len(y_batch)
                y_test_list.extend(y_batch.cpu().numpy())
                y_pred_test_list.extend(y_pred.cpu().numpy())

        y_test = np.array(y_test_list)
        y_pred_test = np.array(y_pred_test_list)

        accuracy = correct_cnt / total_cnt
        print(f'\nTesting: Test Accuracy: {accuracy:.4f} Test F1 Score: {f1_score(y_test, y_pred_test):.4f} Test AUC: {roc_auc_score(y_test, y_pred_test):.4f}')
        torch.save(model.state_dict(), 'classifier_checkpoints/transformer_tuned.pt')




    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['AI', 'Human'])
    disp.plot()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'transformer_confusion_matrix.png'))
    plt.close()

    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'mlp_training_losses.png'), dpi=300)
    plt.close()

    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'transformer_training_losses.png'), dpi=300)
    plt.close()


    
    plt.plot(train_accuracies, label='Train Accuracy', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'transformer_training_accuracies.png'), dpi=300)
    plt.close()


    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)
    fpr_val, tpr_val, _ = roc_curve(y_val, y_pred_val)
    plt.plot(fpr_train, tpr_train, label='Train ROC curve')
    plt.plot(fpr_val, tpr_val, label='Validation ROC curve')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_METRICS_DIR_PATH, 'transformer_train_val_roc_curve.png'), dpi=300)
    plt.close()



    # Save embeddings and performance metrics
    embed = []
    for X_batch, y_batch in test_dataloader:
        y_pred, embeddings = model(X_batch)
        embed.append(embeddings.detach().cpu().numpy())

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'transformer_embeddings.npy'), np.vstack(embed))

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_losses.npy'), np.array(val_losses))

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'train_accuracies.npy'), np.array(train_accuracies))
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'val_accuracies.npy'), np.array(val_accuracies))

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_train.npy'), y_train)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_train.npy'), y_pred_train)

    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_val.npy'), y_val)
    np.save(os.path.join(TRAINING_METRICS_DIR_PATH, 'y_pred_val.npy'), y_pred_val)


    

if __name__ == '__main__':

    main()
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=100)
    
    # print(f'Best trial: {study.best_trial.params}')
    # print(f'Best validation loss: {study.best_value}')
    # study.trials_dataframe().to_csv(os.path.join(TRAINING_METRICS_DIR_PATH, 'transformer_trials.csv'), index=False)
