import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

#DANE (HUGGINGFACE)
print("Loading IMDB...")
dataset = load_dataset("imdb")

X_train = dataset["train"]["text"]
y_train = dataset["train"]["label"]

X_test = dataset["test"]["text"]
y_test = dataset["test"]["label"]

print("Data loaded!")

# (opcjonalnie przyspieszenie)
X_train = X_train[:10000]
y_train = y_train[:10000]
X_test = X_test[:5000]
y_test = y_test[:5000]

# zmiana twkstu na liczby
print("TF-IDF...")

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

print("TF-IDF done!")

# robienie tnesorow
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# dataloader
batch_size = 64

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

model = NeuralNetwork().to(device)
print(model)

#optimizing ADAM
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#train
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            print(f"loss: {loss.item():.4f} [{batch * len(X)}/{size}]")

#test
def test(dataloader, model, loss_fn):
    model.eval()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    print(f"Accuracy: {(100 * correct / total):.2f}% | Loss: {test_loss/len(dataloader):.4f}")

#petla do treningu
epochs = 3

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("Done!")
