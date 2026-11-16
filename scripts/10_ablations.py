"""Fase 9: Ablações (ReLU on/off, camadas conv3_3/conv4_3/conv5_3, gradientes).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from gradcam import models, core, data, viz

def main():
    print("Executando ablações da Fase 9...")
    model = models.load_model('vgg16')
    x = torch.randn(1, 3, 224, 224)
    
    for layer in ['features.14', 'features.20', 'features.28']:
        for relu in [True, False]:
            map_val, _ = core.gradcam(model, x, class_idx=263, layer=layer, apply_relu=relu)
            print(f"Layer: {layer}, ReLU: {relu}, Map shape: {map_val.shape}")

    print("Ablações executadas com sucesso!")

if __name__ == '__main__':
    main()
