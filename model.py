import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Gerät für Training (GPU oder CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    """
    QNetwork-Modell für DQN-Agent.
    Besteht aus:
    - 17 Eingabeneuronen (State-Repräsentation)
    - 2 versteckten Schichten mit je 64 Neuronen
    - 8 Ausgabeneuronen (eine pro möglicher Aktion)
    """

    def __init__(self):
        super(QNetwork, self).__init__()

        # Netzwerkarchitektur definieren
        self.fc1 = nn.Linear(17, 64)    # Input Layer → 1. Hidden Layer
        self.fc2 = nn.Linear(64, 64)    # 1. Hidden Layer → 2. Hidden Layer
        self.fc3 = nn.Linear(64, 8)     # 2. Hidden Layer → Output Layer (Q-Werte für 8 Aktionen)

    def forward(self, state):
        """
        Forward-Pass: Berechne Q-Werte für gegebenen Zustand.
        Eingabe: Zustand (Tensor)
        Ausgabe: Q-Werte für alle 8 Aktionen
        """
        x = F.relu(self.fc1(state))     # Aktivierung nach erster Schicht
        x = F.relu(self.fc2(x))         # Aktivierung nach zweiter Schicht
        return self.fc3(x)              # Keine Aktivierung in der Ausgabeschicht (rohe Q-Werte)

    def save_checkpoint(self, path, index=None):
        """
        Speichert das Modell als .pth-Datei.
        """
        filename = f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth'
        torch.save(self.state_dict(), os.path.join(path, filename))

    def load_checkpoint(self, path, index=None):
        """
        Lädt ein gespeichertes Modell aus .pth-Datei.
        """
        filename = f'qnetwork_torch_madqn_{index}.pth' if index is not None else 'qnetwork_torch_dqn.pth'
        self.load_state_dict(torch.load(os.path.join(path, filename), map_location=device))
