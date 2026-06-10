import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from collections import Counter
import re


#DANE
data = fetch_20newsgroups(subset='all')

X = data.data
y = list(data.target)

# AUGMENTACJA

augmented_texts = []
augmented_labels = []

def add(label, texts):
    augmented_texts.extend(texts)
    augmented_labels.extend([label] * len(texts))

add(1, [
"GPU drivers improved rendering performance significantly.",
"3D graphics are smoother after hardware upgrade.",
"OpenGL applications run more stable after update.",
"Texture rendering is faster with new drivers.",
"Graphics pipeline performance increased.",
"Video games show higher frame rates.",
"Rendering complex scenes is optimized now.",
"Shader execution is improved.",
"3D animation is more efficient.",
"GPU acceleration reduced processing time.",
"Image rendering quality improved.",
"Graphics API performance increased.",
"Real-time rendering is smoother.",
"Frame drops were reduced.",
"Video editing is faster now.",
"Graphics hardware works more efficiently.",
"Display latency decreased.",
"Modern games run better.",
"Rendering engine optimized performance.",
"GPU utilization improved."
])

add(2, [
"Operating system update improved stability.",
"System boots faster after optimization.",
"Windows update fixed critical bugs.",
"Linux kernel upgrade improved performance.",
"System crashes stopped occurring.",
"Memory usage is optimized.",
"Background processes are reduced.",
"System security improved.",
"File system errors were fixed.",
"OS is more responsive.",
"Startup time decreased.",
"System handles multitasking better.",
"Driver issues resolved.",
"System performance increased.",
"Software compatibility improved.",
"System updates improved reliability.",
"Resource usage optimized.",
"System lag reduced.",
"Operating system is stable.",
"System runs more efficiently."
])

add(14, [
"Astronomers discovered a new exoplanet.",
"Space telescope captured galaxy images.",
"NASA launched Mars mission.",
"Satellite data analyzed climate changes.",
"Rocket launch was successful.",
"Black hole research continues.",
"Space station conducted experiments.",
"Deep space probe sent data.",
"Spacecraft technology improved.",
"Cosmic radiation studied.",
"Exoplanet detection improved.",
"Space mission collected data.",
"Astronomical observations revealed stars.",
"Space agency announced mission.",
"Space research expanded knowledge.",
"Telescopes detected signals.",
"Rocket technology advanced.",
"Planet exploration continues.",
"Space science discoveries increased.",
"Orbital data analyzed."
])

add(9, [
"Baseball team won the match.",
"Strong pitching secured victory.",
"Home run decided the game.",
"Team improved ranking.",
"Fans celebrated win.",
"Match ended narrowly.",
"Batting performance was strong.",
"Coaching led to success.",
"Team defense was solid.",
"Pitcher dominated game.",
"Match was intense.",
"Players showed teamwork.",
"Team scored runs.",
"Victory in final inning.",
"Offensive play worked well.",
"Winning streak continued.",
"Game decided late.",
"Key home run scored.",
"Strong defense secured win.",
"Team outplayed opponent."
])

add(10, [
"Hockey team scored goals.",
"Goalkeeper made saves.",
"Match went into overtime.",
"Team showed strong teamwork.",
"Club signed new player.",
"Championship was intense.",
"Power play secured win.",
"Puck moved quickly.",
"Defense worked well.",
"Team reached playoffs.",
"Fans cheered loudly.",
"Winning goal scored.",
"Match was physical.",
"Coordination was strong.",
"Penalty affected result.",
"Game ended in victory.",
"Players showed endurance.",
"Coach changed strategy.",
"Team dominated match.",
"Playoff qualification secured."
])

add(13, [
"Medical research shows promising results.",
"Doctors recommend regular exercise.",
"Clinical trials improved health outcomes.",
"Health studies show progress.",
"New treatment is effective.",
"Researchers published findings.",
"Science improves healthcare.",
"Economic discussions continue.",
"Policy debates are ongoing.",
"Technology improves communication.",
"Software update increased efficiency.",
"Machine learning improves accuracy.",
"AI systems evolve quickly.",
"Data analysis shows patterns.",
"Experiments produced results.",
"Cybersecurity improved.",
"Cloud systems are efficient.",
"Networks became faster.",
"Databases optimized performance.",
"Algorithms became more accurate."
])

#TOKENIZACJA

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


#SŁOWNIK


counter = Counter()

all_texts = X + augmented_texts

for text in all_texts:
    counter.update(tokenize(text))

vocab_size = 10000

vocab = {
    word: idx + 1
    for idx, (word, _) in enumerate(counter.most_common(vocab_size))
}


#ENCODING

max_len = 200

def encode(text):
    tokens = tokenize(text)
    ids = [vocab.get(t, 0) for t in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))

X_encoded = [encode(t) for t in all_texts]
y_all = y + augmented_labels

# TRAIN / TEST
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_all,
    test_size=0.2,
    random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

#DATALOADER
batch_size = 64

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=batch_size
)

# DEVICE

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# MODEL(EMBEDDING)
class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size + 1, 128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 20)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.mean(dim=1)
        return self.classifier(x)

model = TextClassifier().to(device)

#optimizing
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#testtowanie
def train(loader):
    model.train()

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#test

def test(loader):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

    print("Accuracy:", correct / total)


for epoch in range(5):
    print("\nEpoch", epoch + 1)

    train(train_loader)
    test(test_loader)
