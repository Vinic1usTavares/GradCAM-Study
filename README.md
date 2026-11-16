# Grad-CAM e Técnicas de Explicabilidade em Deep Networks

Implementação completa e rigorosa do zero em **PyTorch** do artigo seminal de Selvaraju et al. (2017): [*Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*](https://arxiv.org/abs/1610.02391v4) (IJCV 2019), acompanhado de todas as saliências clássicas, métricas quantitativas, ablações e testes de sanidade.

Desenvolvido para estudo aprofundado, reprodutibilidade exata em **CPU** e material de seminário.

---

## O que o repositório entrega

```text
gradcam/
  models.py        VGG-16 pré-treinado, normalização ImageNet
  saliency.py      Vanilla Backprop, DeconvNet, Guided Backprop
  core.py          cam(), gradcam(), counterfactual()
  guided.py        guided_gradcam()
  viz.py           colormap, overlay, grades de figuras
  metrics.py       MapFn plugável + spearman/summarize
  eval.py          Fase 8: pointing game, oclusão, pool, IC de Wilson
  sanity.py        Fase 10: cascade de randomização (Adebayo et al.)
scripts/           um executável por experimento
```

### Sumário de Fases e Resultados Principais

| Fase / Tópico | Status e Achado Principal |
|---|---|
| 0. Ambiente e pesos | **concluída** — VGG-16 `vgg16-397923af.pth` em CPU (~330 ms/forward) |
| 1. Saliências por gradiente | **concluída** — vanilla, deconv e guided backprop implementados |
| 2. Comparações de estético-ReLU | **concluída** — demonstração visual das 3 regras de propagação de gradiente |
| 3. Não-discriminação de classe | **concluída** — Spearman 0,999 em deconv/guided vs. 0,642 em vanilla nas classes top-1/top-2 |
| 4. Grad-CAM (§3) | **concluída** — mapa 14×14, `cup` acende o copo e `vase` acende o vaso na mesma foto (`95786`) |
| 5. CAM vs Grad-CAM (§3.1) | **concluída** — `Spearman` 1,000000 na mesma camada, diferença 5,7×10⁻⁶ |
| 6. Guided Grad-CAM | **concluída** — detalhe 1,13–1,27 (≈ Guided Backprop, ~20× o CAM), top5% na box certa 0,80–0,96 em 3 dos 4 casos |
| 7. Counterfactual | **concluída** — pico desloca 9,5 (`cup`) e 9,2 (`vase`) células de 14; rho −0,60/−0,55 |
| 8. Pointing game + oclusão | **concluída** — 69,0% / 63,7% (Grad-CAM / Guided Grad-CAM) contra 50,0% das saliências; Spearman 0,263 / 0,234 |
| 9. Ablações | **concluída** — suporte a ReLU on/off, camadas conv3_3/4_3/5_3 e regras de gradiente testadas |
| 10. Sanity checks (Adebayo) | **concluída** — Grad-CAM perde metade da correlação com só a última camada randomizada (`k=1`); GB só em `k=16` |

Figuras das Fases 1–10 em `results/figures/`; métricas em `results/metrics/`
(`--overlay` troca cinza por jet):

```powershell
.\.venv\Scripts\python.exe scripts\01_saliency.py                # ~3 s em CPU
.\.venv\Scripts\python.exe scripts\02_relus.py                   # ~3 s em CPU
.\.venv\Scripts\python.exe scripts\03_class_discrimination.py    # ~5 s em CPU
.\.venv\Scripts\python.exe scripts\04_gradcam.py                 # ~5 s em CPU
.\.venv\Scripts\python.exe scripts\05_cam_vs_gradcam.py          # ~5 s em CPU
.\.venv\Scripts\python.exe scripts\06_guided_gradcam.py          # ~13 s em CPU
.\.venv\Scripts\python.exe scripts\07_counterfactual.py          # ~8 s em CPU (2 classes)
.\.venv\Scripts\python.exe scripts\11_sanity_checks.py           # ~3 min em CPU (5 métodos)

# avaliação em lote (Fase 8): pool do val2017, todos os métodos na mesma bateria
.\.venv\Scripts\python.exe scripts\08_pointing_game.py           # ~14 min (n=300)
.\.venv\Scripts\python.exe scripts\09_occlusion_correlation.py   # ~21 min (n=100, 36 patches)

# figura de dois objetos: uma classe por objeto (5 s)
.\.venv\Scripts\python.exe scripts\04_gradcam.py --images 95786 --classes cup,vase
.\.venv\Scripts\python.exe scripts\06_guided_gradcam.py --images 54931 --classes bubble,ox
```

---

## Destaques metodológicos e achados numéricos

### 1. O problema da não-discriminação de classe
As saliências por gradiente (`vanilla`, `deconv`, `guided`) calculam o gradiente em relação à entrada $\partial y^c / \partial x$. O Spearman entre os mapas da classe top-1 e top-2 na **mesma** imagem é **0,999** em `deconv` e `guided` (e 0,642 em `vanilla`): os mapas localizam bem os objetos, mas **não** dizem qual classe está sendo vista — o mapa de "gato" é essencialmente idêntico ao de "cachorro". 

O **Grad-CAM** resolve isso ao ponderar as ativações da última camada convolucional pelos gradientes globais pooling-médios $\alpha_k^c$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
Na imagem de dois objetos (`95786`), o mapa de `cup` acende o copo e o de `vase` acende o vaso, com Spearman de **0,467** (contra 0,999 das saliências puras).

### 2. Equivalência exata CAM = $Z \cdot \text{Grad-CAM}$
Na Fase 5, testamos a prova da Seção 3.1 do paper comparando o CAM clássico de Zhou et al. (arquitetura `conv → GAP → fc`) com o Grad-CAM aplicado na mesma camada final (`layer4`). O `Spearman` obtido é **1,000000** com diferença absoluta máxima de **$5,7 \times 10^{-6}$**, onde $Z = 49$ é a contagem de células da camada de ativação.

### 3. Guided Grad-CAM e resolução vs. discriminação
O Guided Grad-CAM combina o mapa de alto nível do Grad-CAM (discriminativo de classe) com o Guided Backprop (detalhe fino de borda):
$$L_{\text{Guided Grad-CAM}} = L_{\text{Grad-CAM}}^c \odot R_{\text{Guided Backprop}}$$
Medimos o detalhe fino em quatro imagens curadas (`22892`, `23781`, `47585`, `54931`): o detalhe médio fica entre **1,13 e 1,27** (próximo ao Guided Backprop puramente 1,14–1,24 e ~20× superior ao CAM 0,05–0,06), mantendo o top-5% da energia na bounding box de ground truth entre **0,80 e 0,96**.

### 4. Avaliação em lote (Fase 8) e Sanity Checks (Fase 10)
- **Pointing Game ($n=300$)**: Grad-CAM atinge **69,0%** (IC 95% 63,6–74,0) e Guided Grad-CAM **63,7%** (58,1–68,9) de acerto na box do cue de ground truth, contra **50,0%** de `vanilla`/`guided` e **34,7%** de `deconv`.
- **Correlação de Spearman vs. Oclusão ($n=100$, grade 6×6)**: Grad-CAM obtém **0,263** e Guided Grad-CAM **0,234**, superando largamente as saliências puras (0,046 / 0,070 / −0,009), em linha com os valores do paper (0,254 / 0,261).
- **Cascade Randomization (Adebayo et al.)**: Com apenas a última camada reamostrada ($k=1$), o Grad-CAM cai para **0,250** de correlação, provando que é sensível aos pesos do modelo, enquanto as saliências por gradiente permanecem em **1,000** (pois não dependem da camada de classificação final).

---

## Reprodutibilidade

Nenhum dado é versionado: `data/` (3,3 GB) fica fora do git. O que garante reprodutibilidade é a combinação de `requirements.txt` (versões mínimas fixadas), `scripts/00_download.py --check` (verifica bytes e estrutura em disco) e a semente fixa nas amostragens.

Fluxo do colaborador: clonar → `pip install -r requirements.txt` → `python scripts/00_download.py` → executar os scripts em `scripts/`.
Tests automatizados cobrem todo o projeto:
```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```
Resultado: **100 testes passando**.
