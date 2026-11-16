import os
import json
import torch
import matplotlib.pyplot as plt
from gradcam.data import CocoVal
from gradcam.viz import scale_for_display, gray_rgb

DISPLAY_HIGH = 0.99
DISPLAY_GAMMA = 0.5

def setup(model_name="vgg16"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    
    # Valida se os dados/pesos existem
    weights_path = f"results/models/{model_name}.pth" # ou torchvision standard
    print("Sessão inicializada com sucesso.")
    return {"device": device, "model_name": model_name}

def show(fig):
    fig.tight_layout()
    plt.show()

def grid(rows, cols, **kwargs):
    from gradcam.viz import figure_grid
    return figure_grid(rows, cols, **kwargs)

def result(name):
    path = os.path.join("results", "metrics", f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Métrica não encontrada em {path}. Execute o script correspondente primeiro.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fonte(paper, nosso):
    print(f"[Fonte] Paper: {paper} | Nosso repositório: {nosso}")
