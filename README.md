# GradCAM-Study

Reimplementação do **Grad-CAM** do zero — gradientes por pixel, Deconvolution,
Guided Backpropagation, CAM, Grad-CAM, Guided Grad-CAM e explicacões
contrafactuais — com as avaliações do artigo reproduzidas na escala que o
ambiente permite.

- Artigo: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization*, `arXiv:1610.02391v4` (IJCV 2019).
  PDF local: `GradCAM-paper.pdf`.
- Plano completo, decisões técnicas e critérios de aceite: [`plans/general_plan.md`](plans/general_plan.md).
- Escopo: classificação de imagens. Captioning e VQA (Seç. 8) ficam de fora.

Nenhuma dependência de `pytorch-grad-cam`: tudo implementado em `gradcam/`.

## Setup

Os dados **não** vão para o repositório: `data/` está no `.gitignore` (3,3 GB).
Cada colaborador roda o script uma vez.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\00_download.py          # baixa o que faltar
.\.venv\Scripts\python.exe scripts\00_download.py --check  # só verifica, sem rede
.\.venv\Scripts\python.exe scripts\01_forward_sanity.py
```

`00_download.py` é idempotente e verifica integridade antes de baixar qualquer
coisa:

| Verificação | Como |
|---|---|
| Pesos presentes e completos | tamanho em bytes conferido contra a tabela `EXPECTED_SIZES` |
| Zips presentes e completos | idem (detecta download truncado) |
| Extração do COCO | 5.000 `.jpg` em `data/coco/val2017/` |
| Anotações legíveis | `instances_val2017.json` abre e tem 5.000 imagens |
| Extração parcial | toda imagem anotada existe em disco |

Saída: `0` se tudo ok, `1` se falta algo — serve para CI ou hook de pre-commit.
`--check` nunca toca na rede. `--clean` apaga os `.zip` após extrair (economiza
1 GB).

Requisitos: Python 3.14 (wheel `cp314` do torch), ~3,3 GB de disco com os zips,
~2,2 GB sem eles. Roda em CPU; GPU de 2 GB é suficiente porque não há
treinamento.

## Estrutura

```
gradcam/
  models.py        carregamento dos modelos + correção de ReLU in-place
  data.py          COCO val2017: imagens e boxes, sem pycocotools
  relu_variants.py vanilla | deconv | guided  (Fases 1-2)
  saliency.py      gradiente w.r.t. a entrada
  core.py          cam(), gradcam(), counterfactual()
  guided.py        guided_gradcam()
  viz.py           colormap, overlay, grades de figuras
  metrics.py       MapFn plugável: pointing game, Spearman vs oclusão
scripts/           um executável por experimento
```

## Status

| Etapa | Estado |
|---|---|
| 0. Ambiente, downloads, sanity check | **concluída** — top-1 63%, pytest 7/7 |
| 1. Saliency de gradiente | **concluída** — `02_saliency.py`, 4 imagens × 2 classes |
| 2. Deconv / Guided Backprop | **concluída** — 3 mapas lado a lado |
| 3. Teste de não-discriminação de classe | **concluída** — 0,999 em deconv/guided vs 0,642 em vanilla |
| 4. Grad-CAM | **concluída** — mapa 14×14, Spearman entre classes 0,760 |
| 5. CAM vs Grad-CAM (§3.1) | pendente |
| 6. Guided Grad-CAM | pendente |
| 7. Counterfactual | pendente |
| 8. Pointing game + oclusão | pendente |
| 9. Ablações | pendente |
| 10. Sanity checks (Adebayo) | pendente |

Figuras das Fases 1–4 em `results/figures/`; métricas em `results/metrics/`
(`--overlay` troca cinza por jet):

```powershell
.\.venv\Scripts\python.exe scripts\02_saliency.py              # ~30 s em CPU
.\.venv\Scripts\python.exe scripts\03_class_discrimination.py  # ~75 s em CPU
.\.venv\Scripts\python.exe scripts\04_gradcam.py               # ~18 s em CPU

# figura de dois objetos: uma classe por objeto (5 s)
.\.venv\Scripts\python.exe scripts\04_gradcam.py --images 95786 --classes cup,vase
```

O resultado é o do artigo: as três regras produzem estéticas distintas na mesma
imagem e classe — `vanilla` é ruído fino com um blob difuso sobre o objeto,
`deconv` é ruído com estrutura de grade, `guided` desenha o contorno do objeto
(vaso, laptop, parquímetros). Os picos de `|∂y/∂x|` diferem por ~20× (~0,3 em
`vanilla`, ~6 em `deconv`, ~2 em `guided`), e a mediana fica em ~4% do pico —
por isso as figuras normalizam cada mapa por si, com corte no quantil 99% e
gamma (escala de exibição, nunca de métrica).

E o defeito que o Grad-CAM conserta: `Spearman` entre o mapa da classe top-1 e o
da top-2 na **mesma** imagem é 0,999 em `deconv`/`guided` — os mapas localizam o
objeto, mas não dizem qual classe está sendo vista. Em `vanilla` cai para 0,642,
ainda longe de discriminar. Controle: entre imagens diferentes a correlação vai
a ~0, então a métrica não é degenerada.

O Grad-CAM (Eqs. 1–2) entra na Fase 4 e leva esse número a 0,760 — abaixo do
0,999, mas **acima** do 0,642 do vanilla. A comparação honesta é contra
`deconv`/`guided`, que operam na mesma resolução de pixel: 14×14 contra 224×224
muda a ordenação antes de qualquer diferença de método. O que dá para afirmar
com os dados: os 8 mapas têm o pico dentro da box de ground truth, e o pico só se
desloca entre as duas classes em 2 das 4 imagens. Discriminação de classe em si
é a Fase 8 (pointing game *cued*), onde a resolução não entra na conta.

O critério do plano — "mapa de gato no gato e de cachorro no cachorro, na mesma
foto" — só se cumpre com **dois objetos**, e o top-2 do modelo não serve para
isso: em 23 das 26 imagens do val2017 com dois objetos grandes e separados,
top-1 e top-2 são sinônimos do objeto dominante. Com as classes fixadas uma por
objeto (`04_gradcam_vgg16_95786_cup-vase.png`), o mapa de `cup` acende o copo e
o de `vase` acende o vaso, com `Spearman` 0,467 entre os dois — a discriminação
que o `deconv`/`guided` não tem (0,999) e que o Grad-CAM passa a ter.


## Reprodutibilidade

Nenhum dado é versionado: `data/` (3,3 GB) fica fora do git. O que garante
reprodutibilidade é a combinação de `requirements.txt` (versões mínimas
fixadas), `scripts/00_download.py --check` (verifica bytes e estrutura em disco)
e a semente fixa nas amostragens (`subset`, `stratified_subset`).

Fluxo do colaborador: clonar → `pip install -r requirements.txt` →
`00_download.py` → `00_download.py --check` (espera-se exit 0) →
`01_forward_sanity.py` (espera-se `ambiente OK`).


## Medições

Registradas no `plans/general_plan.md`, seção 0. Resumo: VGG-16 em CPU
(~330 ms/forward, ~1 s/backward), top-1 de 63% em 27 imagens do COCO val2017
restritas às 18 categorias que são classes ImageNet verbatim; não-discriminação
de classe de 0,999 (deconv/guided) e 0,642 (vanilla) em 16 imagens.

Decisão de ambiente: **CPU**, não GPU. Não existe wheel CUDA do torch 2.15 para
Python 3.14 (os índices param em 2.14.1+cu126), e ~330 ms/forward é suficiente
para o volume de avaliação previsto.


## Nota sobre números

O README vai comparar a **ordenação relativa** dos métodos com a do artigo.
Os valores absolutos da Tabela 1/3 dependem do setup Caffe original e de
avaliação 10-crop, então não são reproduzíveis aqui.
