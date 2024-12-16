import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from gensim.models import Word2Vec
import numpy as np
from model import MLPModel


class IMDBDataset(Dataset):
    

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


def create_word_embeddings(texts, embedding_dim=100, window=5, min_count=1):
    
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, window=window, min_count=min_count)
    return model


def text_to_embedding(text, w2v_model, max_len=100):
    
    embedding_dim = w2v_model.vector_size
    embeddings = []

    for word in text.split():
        if word in w2v_model.wv:
            embeddings.append(w2v_model.wv[word])
        else:
            embeddings.append(np.zeros(embedding_dim))

    # Padding ou troncature pour atteindre la longueur max_len
    if len(embeddings) < max_len:
        embeddings += [np.zeros(embedding_dim)] * (max_len - len(embeddings))
    else:
        embeddings = embeddings[:max_len]

    return np.array(embeddings).flatten()


def train_model(model, dataloader, criterion, optimizer, device):
    
    model.train()
    total_loss = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device):
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            predicted = (outputs >= 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    
    # Charger les données IMDB
    data = load_files("aclImdb/train/", categories=["pos", "neg"], encoding="utf-8", shuffle=True)
    texts, labels = data.data, data.target

    # Entraîner Word2Vec pour générer les embeddings
    print("Entraînement du modèle Word2Vec...")
    w2v_model = create_word_embeddings(texts)

    # Convertir les textes en matrices d'embeddings
    print("Conversion des textes en embeddings...")
    embeddings = [text_to_embedding(text, w2v_model) for text in texts]

    # Diviser les données en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    # Créer les datasets et les dataloaders
    train_dataset = IMDBDataset(X_train, y_train)
    val_dataset = IMDBDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialiser le modèle
    input_dim = len(X_train[0])
    hidden_dim = 128
    model = MLPModel(input_dim=input_dim, hidden_dim=hidden_dim)

    # Configurer l'appareil (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Définir la perte et l'optimiseur
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraîner et évaluer le modèle
    epochs = 5
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_accuracy = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    

if __name__ == "__main__":
    main()
