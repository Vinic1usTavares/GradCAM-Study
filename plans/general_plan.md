# GradCAM-Study — Plano Geral

Reimplementação do Grad-CAM **do zero** (sem `pytorch-grad-cam`), reproduzindo as
avaliações do artigo com os meios disponíveis, seguida de seminário.

- **Paper**: Selvaraju, Cogswell, Das, Vedantam, Parikh & Batra. *Grad-CAM: Visual
  Explanations from Deep Networks via Gradient-based Localization*. `arXiv:1610.02391v4`
  (IJCV 2019). Arquivo local: `GradCAM-paper.pdf`.
- **Vídeo de referência**: "Where Does AI Look? Grad-CAM, Built From Scratch" (mesanu,
  32min38s). Cobre o artigo e acrescenta material que **não** está nele: sanity checks
  (Adebayo et al. 2018), Grad-CAM++ e Score-CAM.
- **Escopo**: classificação de imagens apenas. Captioning (§8.1) e VQA (§8.2) fora.

---

## 0. Restrições do ambiente

| Item | Situação |
|---|---|
| Python | 3.14.3 (`./.venv`) |
| PyTorch | **2.15.0+cpu** — o PyPI no Windows entrega build sem CUDA |
| GPU | NVIDIA MX550, **2 GB** — **não usada** (ver decisão abaixo) |
| Datasets | COCO val2017 (816 MB) + anotações (253 MB) — baixados |
| Pesos | `vgg16-397923af.pth` (553 MB) — em `data/weights/` |

### Decisão: CPU em vez de GPU

Para Python 3.14, os índices CUDA do PyTorch vão só até `2.14.1+cu126`; não existe
build CUDA do 2.15. Trocar exigiria downgrade de torch/torchvision + ~3 GB de
download. Medido em CPU: **~330 ms por forward** do VGG-16 a 224×224.

Custo estimado do pior caso (oclusão, 36 patches × 100 imagens): ~20 min. Não
justifica o downgrade. Se a Fase 8/10 crescer, o comando é:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Medições da etapa 0

| Métrica | Valor |
|---|---|
| VGG-16 forward, CPU | 316–414 ms |
| Top-1 (27 imagens, 18 categorias verbatim, `stratified_subset`) | **63%** |
| Top-5, mesmo conjunto | 74% |
| `pytest` | 7 passed |

`Resize(256) + CenterCrop(224)` não trouxe ganho medível sobre `Resize((224,224))`;
ficou o resize direto porque preserva a geometria das boxes do COCO (afim
alinhado aos eixos, nenhuma box cortada). Ver `gradcam/models.py:preprocess`.

### Medições das etapas 1–2

`scripts/02_saliency.py`, VGG-16 em CPU, 4 imagens × 2 classes × 3 regras
(24 passos de backward; ~30 s no total, incluindo os forwards de seleção):

| Métrica | vanilla | deconv | guided |
|---|---|---|---|
| Pico do gradiente | 0,23 – 2,65 | 4,2 – 8,3 | 0,9 – 2,3 |
| Mediana / pico | 3,5% – 5,3% | 4,8% – 7,2% | 0,5% – 1,5% |
| Quantil 99% / pico | 0,24 – 0,35 | 0,27 – 0,37 | 0,13 – 0,16 |

`pytest`: 84 passed (Fases 0–7).

Consequências registradas no código:

1. **Escala de exibição ≠ escala de métrica.** A mediana fica em 4% do pico: uma
   imagem exibida com escala linear até o máximo sai quase toda preta, com a
   estrutura concentrada numa cauda curta de pixels brilhantes. As figuras usam
   `viz.scale_for_display(high=0.99, gamma=0.5)` — branco no quantil 99%, mais
   gamma. Nenhuma métrica usa essa escala (e Spearman nem se abala: é invariante
   a transformação monótona).
2. **Cada mapa é normalizado por si.** Os picos diferem por ~20× entre vanilla
   (~0,3) e deconv (~6); uma escala compartilhada apagaria dois dos três painéis.

> Correção de uma medição anterior: até a Fase 3, `vanilla` estava implementado
> como "gradiente passa sempre pelo ReLU". Isso não é o gradiente da rede e
> inflava o pico para 10⁵–10⁶, com mapas densos e sem contorno. Depois da
> correção (`x > 0`), o pico fica em 0,23–2,65 e o mapa mostra o blob suave
> sobre o objeto que caracteriza a saliência de Simonyan et al.

Achado qualitativo (o mesmo do artigo, Fig. 1): vanilla é ruído fino com um
blob difuso sobre o objeto; deconv é ruído com estrutura de grade; guided
desenha o contorno do objeto. Nos quatro casos (broccoli, parquímetro, laptop,
vaso) o mapa da classe top-2 — `mashed potato`, `cab`, `pencil box`, `soap
dispenser` — é visualmente parecido com o da top-1, sobretudo em guided. É a
impressão que a Fase 3 quantifica logo abaixo.

### Medições da etapa 3

`scripts/03_class_discrimination.py`, 16 imagens (uma por categoria verbatim;
2 das 18 não têm imagem de objeto único com lado ≥ 80), pareamento top-1 vs
top-2, semente 0. Métrica: `spearman(map_A, map_B)`, com o mapa
`max|∂y^c/∂x|` por pixel.

| Método | média | mediana | desvio | n |
|---|---|---|---|---|
| vanilla | **0,642** | 0,650 | 0,090 | 16 |
| deconv | **0,999** | 0,999 | 0,000 | 16 |
| guided | **0,999** | 0,999 | 0,001 | 16 |

Controles rodados à parte, para o número não ser artefato de métrica degenerada:

| Par | vanilla | deconv | guided |
|---|---|---|---|
| mesma imagem, classes top-1 vs top-2 | 0,52 | 0,999 | 0,999 |
| imagens diferentes, mesma classe predita | +0,05 | +0,22 | −0,01 |

Ou seja: a métrica **discrimina** pares de imagens (rho ≈ 0) e colapsa em pares
de classe (rho ≈ 1). Não é o caso de "todo par dá 1".

Leitura correta do 0,999: em deconv/guided a *ordenação* dos pixels é quase a
mesma entre as duas classes — o mapa localiza o objeto, não a classe. Não quer
dizer que os mapas sejam idênticos: 100% dos pixels diferem em valor; o que
coincide é o suporte/ordem. Vanilla, por usar o gradiente comum (com a máscara
`x > 0`), ainda diferencia mais as classes — mas 0,64 continua longe de
discriminar.

O resultado é o do paper (Fig. 2: "gato" e "cachorro" quase iguais em GB), e é
o motivador direto do Grad-CAM. O JSON (`results/metrics/class_discrimination.json`)
é o contrato consumido por `notebooks/02_relus.ipynb`.


### Medições da etapa 4

`scripts/04_gradcam.py`, 4 imagens (as mesmas das etapas 1–2), top-1 vs top-2,
camada = camada de saída do último bloco (ReLU em `features.29`, o `conv5_3` do
artigo), semente 0. Mapa 14×14 — a resolução de `conv5_3` no VGG-16 com entrada
224×224. ~18 s em CPU (8 backward + 28 forward: ~330 ms cada).

| Métrica | Valor |
|---|---|
| Spearman(A, B) na mesma imagem | 0,760 (0,649 – 0,874) |
| Controle: imagens diferentes, mesma classe | +0,009 – +0,211 (4 pares) |
| Resolução do mapa | 14 × 14 |

O Grad-CAM **não** é mais discriminativo que o vanilla backprop no Spearman
(0,76 vs 0,64). Isso é esperado e não invalida o método: a métrica compara a
*ordenação* de pixels de mapas com frequências espaciais muito diferentes — 14×14
suavizado contra 224×224 com ruído por pixel. Num mapa grosseiro, dois blobs
diferentes ainda compartilham a ordem grosseira (topo/base, centro/borda). A
comparação com deconv/guided (0,999, mapas também finos) continua sendo a
relevante; contra elas o Grad-CAM cai de 0,999 para 0,76.

A avaliação correta de discriminação de classe é a **Fase 8** (pointing game
*cued*, tabela comparável), onde a resolução não contamina a métrica. O critério
desta fase é qualitativo, e ele passa — mas com ressalvas medidas. Comparando o
pixel de máximo de cada mapa com a box de ground truth (mapeada para a grade de
224×224; `preprocess` redimensiona sem crop), os 8 mapas têm o pico **dentro** da
box. Entre as duas classes, porém, o pico só se desloca em 2 das 4 imagens:

| Imagem | classe A → pico | classe B → pico | deslocamento |
|---|---|---|---|
| broccoli | (152, 88) | (152, 88) | 0 px |
| parking meter | (152, 184) | (152, 152) | 32 px |
| laptop | (152, 56) | (152, 56) | 0 px |
| vase | (56, 216) | (120, 216) | 64 px |

Em `broccoli` e `laptop` o mapa é o mesmo para as duas classes — 14×14 não separa
duas texturas parecidas, e o rho de 0,874/0,649 diz exatamente isso. Onde separa
(`vase`: desloca para a direita, na borda da box; `parking meter`: sobe 2 células),
a mudança é grosseira. É o mesmo defeito que o §4 do paper admite ao comparar com
localização fraca, e o que a Fase 8 vai medir de forma comparável.

#### A figura que cumpre o critério: dois objetos, duas classes

O pareamento top-1/top-2 acima não testa discriminação de verdade: numa foto de
**um** objeto, o top-2 é quase sempre sinônimo do objeto dominante. Medido em 26
imagens do val2017 com exatamente dois objetos (lado ≥ 150 px em ambos, centros a
≥ 35% da diagonal): em **23 das 26** os dois mapas acendem o mesmo objeto
(`Boston bull`/`French bulldog`, `notebook`/`laptop`, `washbasin`/`toilet seat`).
É o comportamento correto do top-2 — e a razão de a figura precisar de classes
explícitas.

`scripts/04_gradcam.py --images 95786 --classes cup,vase` fixa uma classe por
objeto (as duas são classes ImageNet verbatim, `cup` 968 e `vase` 883):

| Classe | prob | pico | mapa |
|---|---|---|---|
| `cup` | 21,8% | caixa do copo | copo (canto inferior direito) |
| `vase` | 1,3% | caixa do vaso | vaso (cilindro central) |
| — | — | `Spearman` | **0,467** |

Os picos caem em objetos diferentes, e o `Spearman` entre os dois mapas fica em
0,467 — contra 0,999 de `deconv`/`guided` e 0,760 do pareamento top-1/top-2 da
mesma figura. É a evidência do critério de aceite. O número 0,467 é de
pareamento customizado (n=1) e não entra em nenhuma média do repositório.

Dois casos descartados na busca, que valem registro: `176446` (livro + laptop)
acerta o objeto de cada classe, mas com probabilidades 0,8% e 0,3% os dois mapas
se sobrepõem; `431140` (pia + vaso sanitário) tem rótulos bons (38,6% / 31,7%),
mas os dois mapas caem na pia (rho 0,78).

Controles de implementação cobertos por teste (`tests/test_core.py`, 8 testes):

1. **Hook removido**: `test_gradcam_removes_its_hook` garante que a captura de
   ativação não vaza para chamadas seguintes (o erro clássico: `forward_hook`
   acumulando 8 ativações ao longo do script).
2. **Classe explícita vs inferida**: `class_idx=None` usa o `argmax`; a bateria
   sempre passa a classe, para o mapa e o rótulo não saírem de classes diferentes.
3. **`apply_relu=False`**: existe na API para a ablacao da Fase 9, mas não virou
   figura — a escala de exibição corta os negativos e as duas imagens saem quase
   idênticas, o que sugeriria (falsamente) que o ReLU não importa.

Um detalhe de implementação que vale registrar: o Grad-CAM precisa do score
**antes** do softmax e de um backward com one-hot na classe `c`. Usar a
probabilidade depois do softmax muda o mapa (as outras 999 classes entram no
gradiente com peso não nulo), e é uma diferença silenciosa — o mapa continua
plausível.


### Medições da etapa 5

`scripts/05_cam_vs_gradcam.py`, ResNet18, caso `61333` (classe `bookcase`,
camada `layer4`, 7×7):

| Métrica | Valor |
|---|---|
| Spearman(CAM, Grad-CAM) | **1,000000** |
| max abs(Z · Grad-CAM − CAM) | 5,7 × 10⁻⁶ |
| Z = H·W da ativação | 49 |

A equivalência não é "quase": no mesmo tensor, `Z · Grad-CAM = CAM` até o ponto
flutuante (a diferença é 2 × 10⁻⁷ da escala do CAM). Duas armadilhas apareceram no
caminho, e a primeira é a que interessa:

1. **`layer=None` não é a mesma camada nos dois métodos.** `cam` usa a saída do
   bloco (`layer4`); `resolve_layer(model, None)` devolve a última `Conv2d`
   (`layer4[-1].conv2`), que vem **antes** do BatchNorm e da soma residual. Comparar
   os dois dá rho 0,987 e diferença máxima de 6,1 numa escala de 30 — plausível, e
   errado. O script agora resolve a camada uma vez (`resolve_cam_layers`) e passa a
   mesma para os dois; o teste (`tests/test_cam.py`) já fazia isso com
   `layer="layer4"`, e foi o script que se afastou dele.
2. **`Z` é a contagem de células da ativação, não de pixels da entrada**: 7×7 = 49
   no `layer4`, 14×14 = 196 no VGG-16. Usar 224² multiplicaria o mapa por 1024.

Detalhe que só aparece depois da correção: o erro anterior era **silencioso** — o
mapa continuava parecendo um Grad-CAM, e o rho de 0,987 passava por "equivalência
numérica" sem que ninguém perguntasse de onde vinham os 6,1. É o mesmo tipo de falha
que `relu_mode` resolveu na Fase 4: dois métodos, duas capturas, nenhum aviso.

### Medições da etapa 6

`scripts/06_guided_gradcam.py`, VGG-16, camada = saída do último bloco (14×14),
4 casos curados × 2 classes, ~13 s em CPU. Duas medidas, uma para cada metade do
critério de aceite:

- **detalhe** — índice de variação total, `(mean|dx| + mean|dy|) / mean|m|`,
  invariante a escala. Sem ele, comparar o CAM (~0,1) com o Guided Backprop (~2)
  seria comparar magnitudes.
- **localização** — fração dos 5% pixels mais fortes que cai dentro da box de ground
  truth do objeto **certo**, com a box do outro objeto como controle (o argmax
  isolado é frágil: o produto pode deslocar um pixel de borda).

| imagem | classe (prob) | top5% GGC na box certa (na do outro) | top5% CAM na box certa | detalhe CAM / GB / GGC | rho CAM | rho GGC |
|---|---|---|---|---|---|---|
| 22892 | Egyptian cat (0,05%) | 0,94 (0,13) | 1,00 | 0,06 / 1,18 / 1,27 | +0,062 | +0,343 |
| 22892 | beagle (0,08%) | 0,80 (0,20) | 1,00 | 0,06 / 1,18 / 1,13 | | |
| 23781 | broccoli (0,23%) | 0,63 (—) | 0,94 | 0,05 / 1,14 / 1,17 | +0,595 | +0,799 |
| 23781 | cucumber (23,5%) | — (0,20) | — | 0,06 / 1,14 / 1,15 | | |
| 47585 | groom (91,2%) | 0,88 (0,24) | 0,79 | 0,06 / 1,20 / 1,20 | +0,016 | +0,326 |
| 47585 | umbrella (3,8%) | 0,93 (0,22) | 1,00 | 0,06 / 1,20 / 1,20 | | |
| 54931 | bubble (12,8%) | 0,93 (0,00) | 0,99 | 0,06 / 1,24 / 1,24 | −0,058 | −0,009 |
| 54931 | ox (5,7%) | 0,96 (0,01) | 1,00 | 0,06 / 1,24 / 1,27 | | |

**A metade "resolução" do critério está cumprida, e com folga.** O índice de detalhe
do Guided Grad-CAM (1,13 – 1,27) fica em cima do Guided Backprop (1,14 – 1,24) e
~20× acima do CAM interpolado (0,05 – 0,06): o produto de fato devolve a textura, e
o efeito é o mesmo nos quatro casos. Os zooms (`06_..._23781_zoom.png`,
`06_..._47585_zoom.png`) mostram isso no recorte: o CAM é uma grade de 14×14
interpolada, o produto desenha as hastes do brócolis e os setores do guarda-chuva.

**A metade "discriminação" se sustenta na box, não na ordenação de pixels.** O
top5% na box do objeto certo fica entre 0,80 e 0,96 em 3 dos 4 casos, com a box do
outro objeto em ≤ 0,24 (`54931` é o caso limpo: `bubble` → 0,93 na pessoa e 0,00 no
cavalo; `ox` → 0,96 no cavalo e 0,01 na pessoa). Três ressalvas medidas:

1. **O produto não melhora a localização do CAM** — em 6 dos 7 casos em que as
   duas medidas existem, o top5% na box certa cai um pouco (1,00 → 0,94; 1,00 →
   0,80; 0,94 → 0,63); a exceção é `groom` (0,79 → 0,88). Multiplicar por um campo
   não-negativo só move massa para onde o Guided Backprop é claro, e ele tem suporte
   de borda próprio. Quem localiza continua sendo a máscara do CAM.
2. **O rho entre as duas classes sobe em todos os quatro casos** (+0,28, +0,20,
   +0,31, +0,05). É o esperado: o termo de detalhe é quase invariante à classe
   (0,999 na Fase 3), então somá-lo puxa a ordenação para um campo de textura
   comum. Onde as máscaras do CAM realmente diferem (`54931`, rho −0,058) a
   diferença sobrevive (−0,009); onde elas já se sobrepunham (`22892` +0,062,
   `47585` +0,016) o produto apenas se parece mais com ele mesmo. Ou seja: a frase
   "detalhe fino **sem perder** discriminação" vale para *onde* (box), não para a
   *ordenação fina* dos pixels — e esta é a leitura honesta.
3. **O pior caso é o mais parecido**: `23781` (brócolis × pepino) cai de 0,94
   (CAM) para 0,63, e o pepino nem tem box anotada — os 37% que saem da box do
   brócolis não podem ser atribuídos a nada. É a limitação do caso, não uma medida
   de erro do método.

Casos medidos e **descartados**, que valem registro porque são o limite do método:

| imagem | cue | prob | top5% GGC na box certa | na box do gato |
|---|---|---|---|---|
| 61333 | `bookcase` (top-1) | 44,7% | 0,27 | 0,69 |
| 61333 | `bookshop` | 10,1% | 0,31 | 0,61 |
| 61333 | `quilt` / `pillow` / `studio couch` | 0,7 – 6,7% | 0,09 – 0,12 | 0,89 – 0,92 |
| 106389 | `rocking chair` | 0,6% | 0,57 | 0,27 |
| 95786 | `cup` / `vase` | 21,8% / 1,3% | 0,79 / 0,75 | — |

Em `61333` o gato domina o gradiente qualquer que seja o cue: **nenhuma** classe de
estante ou colcha localiza a estante, nem antes nem depois do produto. `95786`
funciona (0,79 / 0,75), mas foi descartado por curadoria — é a mesma imagem da Fase 4
e repetiria a figura; `106389`, que seria o par "gato × cadeira", também falha. O que
sobra é a regra prática: o Guided Grad-CAM herda a localização do CAM, então escolher
imagem por medida — e não por intuição — é parte do método. A triagem rodou sobre o
val2017 com 2–3 objetos grandes e separados (lado ≥ 120 px, área ≥ 3% da imagem,
centros a ≥ 30% da diagonal — 326 imagens) e ficou com 4 casos.

### Medições da etapa 7

`scripts/07_counterfactual.py`, VGG-16, imagem `95786` (copo + vaso), camada
`features.29`, ~8 s. Duas colunas de mapa por classe: Grad-CAM (a favor) e o
mesmo mapa com `alpha` negado em (1) antes do ReLU de (2) (contra).

| Classe | prob | pico a favor | pico contra | deslocamento | Spearman(a favor, contra) |
|---|---|---|---|---|---|
| cup | 21,8% | (10,9) | (13,0) | 9,5 células (152 px) | −0,605 |
| vase | 1,3% | (2,9) | (11,7) | 9,2 células (147 px) | −0,546 |

O Spearman negativo é estrutural — negar `alpha` inverte o sinal da combinação
—, então ele **não** é a métrica de aceite: o critério é o deslocamento. Nos
dois casos o pico sai do objeto da classe e cai a ~2/3 do lado do mapa
(9,5 e 9,2 células de 14).

Leitura honesta do que "evidência contrária" significa: o contrafactual não é o
mapa da "outra classe" e não precisa acender o vaso para `cup`. Ele acende o que
**reduz** `y^c` — no caso do `cup`, bordas e o fundo da cena; no do `vase`, a
região do segundo copo. O critério do plano ("o mapa desloca para a evidência
contrária") é cumprido no deslocamento; a identificação do objeto vizinho é
qualitativa e vale para `vase`, não para `cup`, e isso fica registrado.

Controles de implementação cobertos por teste (`tests/test_counterfactual.py`,
4 testes): sem ReLU, o contrafactual é **exatamente** o negativo do Grad-CAM
(`cf == -gc`), o que fixa a definição; o hook de ativação é removido entre
chamadas; e `class_idx=None` continua usando o `argmax` nos dois métodos. A
derivação foi fatorada em `core._class_weights`, então `gradcam` e
`counterfactual` não podem divergir na conta de (1).


### Restrição descoberta: COCO × ImageNet

Apenas **18** das 80 categorias do COCO são classes ImageNet verbatim
(`backpack`, `banana`, `broccoli`, `cup`, `dining table`, `kite`, `laptop`,
`microwave`, `mouse`, `orange`, `parking meter`, `pizza`, `refrigerator`,
`toaster`, `traffic light`, `umbrella`, `vase`, `zebra`).

Isso afeta a Fase 8: no pointing game do paper a visualização é *cued* com o
rótulo de ground truth, não com a predição. Protocolo adotado: para cada
categoria COCO, definir um conjunto de synsets ImageNet correspondentes e usar o
`argmax` do modelo **restrito a esse conjunto** como classe de cue. Para as 18
verbatim, o conjunto é unitário.


---

## 1. Estrutura do repositório

```
GradCAM-Study/
├── gradcam/
│   ├── __init__.py
│   ├── models.py          # carregamento dos modelos pré-treinados + correção de ReLU in-place
│   ├── relu_variants.py   # vanilla | deconv | guided  (autograd.Function)
│   ├── saliency.py        # Fases 1–2: gradiente w.r.t. a entrada
│   ├── core.py            # cam(), gradcam(), counterfactual()
│   ├── guided.py          # guided_gradcam = gradcam ⊙ guided
│   ├── viz.py             # colormap, overlay, grade de figuras
│   ├── metrics.py         # MapFn plugável: pointing game, Spearman vs oclusão
│   └── data.py            # COCO val2017: leitura de imagens e boxes
├── scripts/
│   ├── 01_forward_sanity.py
│   ├── 02_saliency.py
│   ├── 03_class_discrimination.py
│   ├── 04_gradcam.py
│   ├── 05_cam_vs_gradcam.py
│   ├── 06_guided_gradcam.py
│   ├── 07_counterfactual.py
│   ├── 08_pointing_game.py
│   ├── 09_occlusion_correlation.py
│   ├── 10_ablations.py
│   └── 11_sanity_checks.py
├── tests/
├── data/                  # gitignored (COCO + pesos)
├── results/figures/
├── plans/
├── requirements.txt
├── .gitignore
└── README.md
```

**Regra**: nada de `pytorch-grad-cam` no código de produção. No máximo como oráculo
num teste de `pytest`.

---

## 2. Decisões técnicas

### 2.1 ReLU in-place

O `torchvision` usa `nn.ReLU(inplace=True)`, o que quebra hooks e versões de tensor.
A correção é feita uma única vez em `models.py`:

```python
for m in model.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False
```

### 2.2 As três regras de backward (Fases 1–2)

Os três métodos de saliência diferem **apenas** na máscara aplicada ao gradiente na
volta pelo ReLU:

| Método | `grad_out` passa se… |
|---|---|
| Vanilla (Simonyan 2013) | `x > 0` — é o gradiente comum do ReLU |
| Deconvolution (Zeiler 2014) | `grad_out > 0` |
| Guided Backprop (Springenberg 2014) | `grad_out > 0` **e** `x > 0` |

`vanilla` **não** é "deixa passar sempre". Deixar passar tudo produz o gradiente
de uma rede com os ReLUs linearizados na volta: outro objeto, com mapas densos e
picos ~10⁶ em vez de ~10⁰. Essa era a definição da primeira versão; o teste
`test_vanilla_matches_autograd` (compara com `torch.autograd` puro) fixa a
correta e pegaria a regressão.

Implementado via `torch.autograd.Function` (e não `register_full_backward_hook`, que
tem restrições documentadas e é mais lento):

```python
class _ReluVariant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mode):
        ctx.save_for_backward(x); ctx.mode = mode
        return x.clamp(min=0)
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        if ctx.mode == "deconv":
            g = g * (g > 0)
        elif ctx.mode == "guided":
            g = g * (g > 0) * (x > 0)
        return g, None
```

Nota para o seminário: Guided Backprop **não é o gradiente de nenhuma função** (não é
conservativo) — é heurística de visualização. Conecta com a crítica de Adebayo et al.

### 2.3 Grad-CAM

```python
α_k^c = (1/Z) Σ_i Σ_j ∂y^c/∂A^k_ij        (1)
L^c   = ReLU( Σ_k α_k^c · A^k )           (2)
```

- `y^c` é o score **antes** do softmax (o paper é explícito, §3).
- Semente do backward: uma-hot na classe `c` ⇒ gradientes zero para as demais (§Fig. 2).
- ReLU em (2) não é opcional: sem ele o erro de localização sobe 15,3% (Tabela 3).
- Captura: `forward hook` guarda `A`; `A.retain_grad()`; `torch.autograd.grad(y_c, A)`.

### 2.4 Interface plugável de mapas

Toda avaliação passa por uma assinatura única, senão cada método duplica seu script:

```python
MapFn = Callable[[nn.Module, torch.Tensor, int], torch.Tensor]   # (modelo, entrada, classe) -> HxW
```

### 2.5 Tratamento de imagens

- Normalização ImageNet **antes** do forward; desnormalização **só** para exibir.
- Escala de exibição: `viz.scale_for_display(high=0.99, gamma=0.5)` (branco no
  quantil 99% + gamma). Nunca em métrica — ver as medições das etapas 1–2 na
  seção 0.
- `model.eval()` sempre: BatchNorm em modo `train` produz mapas inúteis.
- Saliency: `torch.enable_grad()` obrigatório (não pode ser `inference_mode`).

---

## 3. Fases

| # | Entrega | Critério de aceite |
|---|---|---|
| **0** | `requirements.txt`, env instalado, VGG-16 + COCO val2017 baixados, `data.py` lendo boxes | `01_forward_sanity.py` acerta top-1 em 3 imagens |
| **1** | Saliency vanilla `∂y^c/∂x` no VGG-16 | mapas com aspecto de contorno ruidoso; 4 imagens × 2 classes |
| **2** | Deconv e Guided Backprop (mesma conta, outra máscara) | 3 mapas lado a lado da mesma imagem/classe |
| **3** | **Não-discriminação de classe**: `Spearman(map_cat, map_dog)` na mesma imagem | GB alto, mostrando quantitativamente o problema que o paper resolve — **medido: 0,999 em deconv/guided, 0,642 em vanilla** |
| **4** | Grad-CAM (Eqs. 1–2) | mapa de "cat" no gato e de "dog" no cachorro, na mesma foto — **cumprido em `95786` com `--classes cup,vase`: `cup` acende o copo, `vase` acende o vaso, Spearman 0,467**. Proxy no pareamento top-1/top-2: 0,760 contra 0,999 de deconv/guided; controle entre imagens +0,009–0,211 |
| **5** | CAM em ResNet18 (`conv → GAP → fc`) e equivalência numérica com Grad-CAM | Spearman > 0,9 na mesma camada; valida a prova da §3.1 — **medido: Spearman 1,000000 e max abs(Z·Grad-CAM − CAM) 5,7 × 10⁻⁶**. Sem fixar a camada, os dois métodos capturam tensores diferentes e o número cai para 0,987 (ver seção 0) |
| **6** | Guided Grad-CAM = Fase 4 ⊙ Fase 2 | detalhe fino (listras) sem perder discriminação de classe — **medido: detalhe 1,13–1,27, em cima do Guided Backprop (1,14–1,24) e ~20× o CAM (0,05–0,06); top5% na box certa 0,80–0,96 em 3 dos 4 casos** (`broccoli` em 0,63 é o pior). Ressalva medida: o rho entre classes **sobe** nos quatro casos, porque o termo de detalhe é quase invariante à classe |
| **7** | Counterfactual (negar o gradiente em (1)) | mapa desloca para a evidência contrária — **cumprido no deslocamento: pico sai do objeto em `95786` (`cup` 9,5 e `vase` 9,2 células de 14); acende evidência complementar, não necessariamente o objeto vizinho** |
| **8** | Pointing game + Spearman vs oclusão, **todos os métodos na mesma bateria** | tabela comparável em ordenação à Tabela 1/2 do paper |
| **9** | Ablações (ReLU on/off, GAP vs GMP, camada conv3_3/4_3/5_3, gradiente vanilla/guided/deconv) | reproduzir a **direção** da Tabela 3 |
| **10** | Sanity checks (cascade randomization, Adebayo et al.) | queda monotônica da correlação ao randomizar camadas de cima para baixo |
| **11** | README com tabela "paper vs. nosso número" + material do seminário | — |

---

## 4. Protocolo de avaliação

O que **não** é replicável, e por quê:

| Paper | Motivo | Substituto |
|---|---|---|
| Erro de localização ILSVRC-15 | exige ImageNet val + boxes (~7 GB) | **pointing game** em COCO val2017 |
| Estudo humano (AMT) | 43–54 trabalhadores | análise qualitativa curada |
| IoU de segmentação VOC 2012 | +2 GB e outro modelo | opcional, se houver tempo |

**Conjunto de avaliação**: COCO val2017, subset de ~300–500 imagens com anotações de
instância. Serve para pointing game (ponto de máximo dentro da box GT da classe predita)
e para a correlação com oclusão (referência de fidelidade local).

**Cue do pointing game** (ver restrição COCO × ImageNet na seção 0): o paper usa o
rótulo de ground truth, não a predição. Para cada categoria COCO, mapear para um
conjunto de synsets ImageNet e usar o `argmax` do modelo restrito a esse conjunto.
Sem isso, usar a predição do modelo como cue gera muitos casos insolúveis, porque o
VGG-16 pré-treinado no ImageNet acerta ~63% nesse domínio.

**Ablações**: reproduzimos a *direção* dos efeitos, não os números absolutos — aqueles
dependem do setup Caffe original e de avaliação 10-crop. Isso fica explícito no README.

---

## 5. Números de referência do paper

| Avaliação | Valor |
|---|---|
| Localização fraca ILSVRC-15, VGG-16 top-1 | 56,51 (CAM 57,20; Backprop 61,12; c-MWP 70,92) |
| AlexNet top-1 | 68,3 (c-MWP 92,6) |
| Segmentação VOC 2012 (IoU) | 49,6 (CAM 44,6) |
| Pointing game | 70,58% (c-MWP 60,30%) |
| Estudo humano, discriminação de classe | GB 44,44% → Guided Grad-CAM 61,23% |
| Confiabilidade relativa | GB +1,00 → Guided Grad-CAM +1,27 |
| Spearman vs oclusão | Grad-CAM 0,254 / Guided Grad-CAM 0,261 (GB 0,168) |
| Ablação: sem ReLU em (2) | erro 59,65 → 74,98 |
| Ablação: Deconv-ReLU | erro 83,95 |

---

## 6. Roteiro do seminário

1. O gancho: médicos vs. enfermeiras, o modelo olha o rosto (Fig. 8) — 2 min
2. Backprop como mapa de saliência: `∂y^c/∂x` — 3 min
3. Três regras de ReLU, três estéticas — mas o mapa do gato é igual ao do cachorro (nosso número) — 4 min
4. CAM: exige trocar a arquitetura, perde 2,98% top-1 — 2 min
5. Grad-CAM: a derivação (Eqs. 1–2) — 4 min
6. Generaliza CAM: `w_k = Z·α_k`, validado numericamente — 3 min
7. Guided Grad-CAM como composição: resolução × discriminação — 2 min
8. Nossos números vs. paper — 3 min
9. Sanity checks (Adebayo): o contra-argumento — 3 min
10. Limitações: 14×14, degrada em camadas rasas, não é pixel-accuracy; Grad-CAM++/Score-CAM — 2 min

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| Download de COCO falhar | plano B: ~100 imagens locais, avaliação só qualitativa |
| VGG-16 estourar memória | batch 1; `inference_mode` onde possível |
| `pycocotools` difícil de compilar | não usar: ler `instances_val2017.json` direto com `json` |
| Expectativa de números idênticos ao paper | README compara ordenação relativa, com justificativa |

---

## 8. Dados: nada versionado, tudo verificável

O repositório **não** contém dados. `data/` está no `.gitignore` (3,3 GB com os
`.zip`, 2,2 GB sem eles) e só `plans/`, `gradcam/`, `scripts/`, `tests/` e
`requirements.txt` vão para o GitHub.

Colaborador novo:

```powershell
git clone ...
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\00_download.py          # baixa o que faltar
.\.venv\Scripts\python.exe scripts\00_download.py --check  # exit 0 = pronto
```

`--check` é offline e valida, em ordem:

| Item | Critério |
|---|---|
| pesos (`vgg16`, `alexnet`, `resnet18`) | tamanho em bytes == `EXPECTED_SIZES` |
| zips do COCO | tamanho em bytes == `EXPECTED_SIZES` |
| `data/coco/val2017/` | 5000 arquivos `.jpg` |
| `instances_val2017.json` | abre, 5000 imagens, > 0 anotações |
| extração parcial | toda imagem anotada existe em disco |

O download em si também é idempotente: grava em `.part` e só renomeia no fim
(nunca deixa arquivo truncado com nome válido), pula o que já tem tamanho
correto, e usa um marcador de extração **por arquivo** (`.extracted/<zip>.done`)
— marcador compartilhado faria o segundo zip ser pulado.

