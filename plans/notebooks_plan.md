# Notebooks do seminário — plano

Material de apoio da apresentação. Cada beat do roteiro (`general_plan.md`, §6) vira
um notebook executável: projeta-se o notebook, roda-se a célula, e o número que
aparece na tela é o número que o projeto mediu.

- Roteiro de origem: `plans/general_plan.md` §6 (10 beats, 28 min).
- Fases de código: `plans/general_plan.md` §3 (Fases 0–11).
- Nada aqui substitui `gradcam/`: notebook **não contém lógica de método**.
- Estado na revisão atual: **Fases 0–2 concluídas**. `viz.figure_grid`,
  `viz.scale_for_display` e `viz.gray_rgb` já existem — o notebook **reusa**,
  não reimplementa. Ver §11 (histórico).

---

## 0. Decisões fechadas

| # | Decisão | Escolha |
|---|---|---|
| 1 | Granularidade | **11 notebooks**, um por beat, + `07` como bônus (contrafactual) |
| 2 | Fonte dos números | **cache de `results/metrics/`** + demos ao vivo curtas |
| 3 | Ativos fora do COCO | **uma** foto gato+cachorro (CC-BY/CC0, Wikimedia Commons, ~200 KB) |
| 4 | Projeção | notebook rolável no Jupyter + export **HTML e PDF (`webpdf`)** |
| 5 | Figuras do paper | **não** recortar. Beat 1 é narração da Fig. 8 + teaser ao vivo |
| 6 | Outputs commitados | **sim** — o material precisa abrir sem os 3,3 GB de `data/` |
| 7 | Escala de exibição | `viz.scale_for_display(high=0.99, gamma=0.5)` em **toda** figura de gradiente (decisão de `general_plan.md` §2.5), nunca em métrica |
| 8 | Grades de figura | delegar a `viz.figure_grid`; `deck.py` só adiciona o que falta (bootstrap, cache, zoom) |

A decisão 7 é um slide por si só, e entra no `01_saliency`: o pico do gradiente
vanilla é ~10⁵–10⁶ e a mediana ~5% disso. Normalizar pelo máximo deixa o mapa
quase preto; o corte no quantil 99% mais gamma 0,5 é o que faz a textura aparecer.
Vale dizer em voz alta que **escala de exibição ≠ escala de métrica**.

### Por que outputs commitados

O handout e o notebook precisam ser legíveis em máquina sem `data/` nem pesos.
O que garante que o commitado não mente é `scripts/12_run_notebooks.py --check`,
que reexecuta todos em kernel limpo e falha no primeiro erro.

### Por que `webpdf` e não LaTeX

`--to pdf` exige TeX Live (~1 GB) e quebra em fórmulas específicas. `--to webpdf`
usa Playwright + Chromium (~150 MB, download único) e renderiza o HTML fielmente.
Fica fora do `requirements.txt` de produção — está no `requirements-notebooks.txt`.

---

## 1. Estrutura

```
notebooks/
  deck.py                     # bootstrap, grid(), zoom(), result(), fonte()
  cases.py                    # seleção determinística de imagens/pares de classe
  00_indice.ipynb             # beat 1  — gancho + índice + check de ambiente
  01_saliency.ipynb           # beat 2
  02_relus.ipynb              # beat 3
  03_cam.ipynb                # beat 4
  04_gradcam.ipynb            # beat 5
  05_equivalencia.ipynb       # beat 6
  06_guided.ipynb             # beat 7
  07_counterfactual.ipynb     # bônus   — fora dos 10 beats, primeiro a cortar
  08_numeros.ipynb            # beat 8
  09_sanity.ipynb             # beat 9
  10_limitacoes.ipynb         # beat 10
  README.md                   # ordem, tempos, lista de corte
  assets/
    cat_dog.jpg
    CREDITS.md
results/
  figures/                    # PNGs que vão para README/slides
  metrics/                    # JSON produzido pelos scripts (fonte única de verdade)
  seminar/                    # export HTML/PDF — gitignored, regenerável
scripts/
  12_run_notebooks.py         # --check | --export
requirements-notebooks.txt
```

Adições ao `.gitignore`:

```
.ipynb_checkpoints/
results/seminar/
notebooks/.cache/
```

---

## 2. Princípios

1. **Notebook não implementa.** A conta vem de `gradcam/`. A célula é a prova
   visual da derivação, não uma segunda implementação. Se um número só existe no
   notebook, ele está no lugar errado.
2. **Pesado não roda ao vivo.** Pointing game (300–500 imagens) e oclusão (~20 min
   em CPU) ficam nos `scripts/` e gravam `results/metrics/*.json`. O notebook plota.
3. **Todo notebook executa em < 30 s** (o mais caro é `09_sanity`, ~15 s); os 11
   juntos ~50 s fora o load do modelo. Reexecutar tudo antes de apresentar leva < 2 min.
4. **Cabe na projeção.** ≤ 12 células de código e ≤ 6 figuras por notebook, uma
   figura por célula, fonte ≥ 12 pt, dpi 110, `figsize` ≤ (10, 5).

---

## 3. Os notebooks

Tempos conforme §6. Soma 29 min + 3 min de folga = 32 min.

| Notebook | Beat | Pergunta que responde | Células-chave | Runtime | Depende |
|---|---|---|---|---|---|
| `00_indice` | 1 · 2 min | "onde o modelo olha?" | agenda cronometrada com links; `deck.setup()`; teaser: GB do **gato** e do **cachorro** na mesma foto, quase idênticos | 10 s | Fase 0 ✅ |
| `01_saliency` | 2 · 3 min | "e se eu derivar a saída em relação à imagem?" | `grad` cru (3×224×224) → um canal → `max\|·\|`; **escala de exibição** (quantil 99% + gamma); 1 imagem × 2 classes, cinza | 4 s | Fase 1 ✅ |
| `02_relus` | 3 · 4 min | "por que Deconv e Guided parecem melhores?" | 1 célula com as 3 linhas de máscara (o slide **é** o código); 3 mapas lado a lado; picos (`10⁶` / `10⁰` / `10⁰`); **punchline:** Spearman(gato, cachorro) ≈ 0,9 em barras | 5 s | Fase 2 ✅ / Fase 3 |
| `03_cam` | 4 · 2 min | "CAM: e se eu tirar o FC?" | cirurgia no ResNet18 `conv → GAP → fc`; CAM via hook; custo de 2,98% top-1 | 1 s | Fase 5 |
| `04_gradcam` | 5 · 4 min | "como obter CAM sem trocar a arquitetura?" | **derivação em 8 células**: hook em `A` → `y^c` → one-hot → `α=GAP(∂y/∂A)` → `ΣαA` → `ReLU` → upsample → overlay; célula-espelho `gradcam(model, x, c)`; ablação ReLU on/off | 3 s | Fase 4 |
| `05_equivalencia` | 6 · 3 min | "Grad-CAM é mesmo um CAM generalizado?" | `w_k = Z·α_k`; scatter CAM × Grad-CAM na mesma camada; Spearman > 0,9 | 1 s | Fase 5 |
| `06_guided` | 7 · 2 min | "dá para ter resolução **e** discriminação?" | `Grad-CAM ⊙ GB`; zoom nas listras da zebra; comparar resolução (224) com classe | 3 s | Fase 6 |
| `07_counterfactual` | bônus · 1 min | "e a evidência *contra*?" | negar `α` em (1); mapa migra para o objeto vizinho | 3 s | Fase 7 |
| `08_numeros` | 8 · 3 min | "nossos números vs paper" | pointing game + Spearman vs oclusão, todos os métodos; barras da **ordenação** nossa × paper, lidas do JSON | 0 | Fases 8–9 |
| `09_sanity` | 9 · 3 min | "isso é fiel ao modelo?" | cascade randomization: correlação cai monotonicamente; GB "chapado" × Grad-CAM estável | 15 s | Fase 10 |
| `10_limitacoes` | 10 · 2 min | "quais são os limites?" | grade 14×14 × 224; degradação em `conv3_3`; menção a Grad-CAM++/Score-CAM | 5 s | Fases 9–10 |

Os runtimes acima usam a medição de `02_saliency.py`: **~1,1 s por backward** no
VGG-16 em CPU (24 backwards = 26,6 s) e ~0,35 s por forward. Backward é o custo
dominante — um mapa de Grad-CAM custa ~1,5 s, não os 330 ms do forward. Soma
total dos 11 notebooks: **~50 s**.

### Notas por notebook

**`00_indice`** — beat 1 é narração da Fig. 8 (médicos vs. enfermeiras): o claim do
paper é apresentado oralmente, sem recorte da figura. O teaser ao vivo usa o par
gato/cachorro e planta a pergunta que o resto do seminário responde. É também o
índice: tabela beat → notebook → tempo, com links relativos.

**`01_saliency`** — duas coisas precisam ficar explícitas na tela: (a) o gradiente
tem **sinal** e 3 canais, e `max|·|` por pixel é uma escolha de redução; (b) a
escala. Os mapas vão em **cinza** (`viz.gray_rgb`), como em `02_saliency.py`, porque
o objetivo é comparar textura, não cor. A versão completa (4 imagens × 2 classes ×
3 regras, ~30 s) é o `scripts/02_saliency.py`, já no repo: notebook mostra 1 caso,
script gera as figuras de `results/figures/`. Mencionar isso evita "por que não
rodou tudo aqui?".

**`02_relus`** — o "código como slide" (`g * (g > 0) * (x > 0)`) é o momento em que
a plateia entende que os três métodos diferem em uma linha. Vale mostrar os picos
medidos (vanilla `10⁵–10⁶`, deconv `10⁰`, guided `10⁰`) para justificar por que cada
mapa é normalizado por si. Fechar com a medição de Fase 3, não com impressão visual;
enquanto a Fase 3 não existir, a punchline fica como pergunta em aberto.

**`04_gradcam`** — centro do seminário. Oito células pequenas, cada uma com uma
equação do paper no markdown acima. A nona célula é a chamada de biblioteca que
produz exatamente o mesmo mapa — fecha o argumento "isto é implementável".

**`08_numeros`** — nenhum cálculo. Lê `pointing_game.json` e
`occlusion_spearman.json`, plota as duas tabelas e destaca só a **ordem** dos
métodos: é o que o README promete comparar.

---

## 4. Contratos

### 4.1 `notebooks/deck.py`

Sem lógica de método — só bootstrap, cache e o que `viz.py` não cobre.

```python
DISPLAY_HIGH = 0.99                  # reexportado: escala de figura, nunca de métrica
DISPLAY_GAMMA = 0.5

setup(model="vgg16") -> Sessao       # model, device, labels, coco lazy, seed 0
                                     # erro explícito citando scripts/00_download.py
                                     # imprime sha1 dos ids de cases.py (detecta drift)
show(fig)                            # tight_layout + plt.show(); dpi de projeção
grid(rows, ...)                      # thin wrapper de viz.figure_grid (dpi 110)
zoom(img, heatmap, box)              # recorte com overlay — listras da zebra
result(nome) -> dict                 # lê results/metrics/<nome>.json + idade do arquivo
fonte(paper, nosso)                  # rodapé "paper: X / nosso: Y"
```

`deck.grid` existe só para fixar `dpi`/`figsize` de projeção — `viz.figure_grid`
já faz o trabalho e não deve ser duplicado. `gray_rgb`, `overlay`,
`scale_for_display` e `tensor_to_numpy` são chamados direto de `gradcam.viz`.

Regras: `matplotlib` inline, `font.size >= 12`, sem seaborn, sem `print` de tensor.
Sessão mantém o modelo carregado entre células — load de VGG-16 é o custo dominante.

### 4.2 `notebooks/cases.py`

Seleção por **semente + allowlist de categoria**, não ids escritos à mão: `data/`
não está no repositório, então ids cravados agora seriam chute. O primeiro run com
dados presentes congela a lista no próprio arquivo (`MANIFEST`), e `setup()` imprime
o sha1 para o drift aparecer no `--check`.

### 4.3 Anatomia de cada notebook

1. Célula 0 (idêntica em todos): `import deck; sessao = deck.setup()`.
2. Markdown de abertura: pergunta que o notebook responde + tempo + link para o
   beat anterior/próximo.
3. Blocos de 1 figura + 1 frase em **negrito** com o takeaway.
4. Célula final: cita o JSON de origem e a linha da tabela do README que ele
   alimenta.

### 4.4 `results/metrics/`

| Arquivo | Produzido por | Consome |
|---|---|---|
| `class_discrimination.json` | `03_class_discrimination.py` | `02_relus` |
| `cam_vs_gradcam.json` | `05_cam_vs_gradcam.py` | `05_equivalencia` |
| `pointing_game.json` | `08_pointing_game.py` | `08_numeros` |
| `occlusion_spearman.json` | `09_occlusion_correlation.py` | `08_numeros` |
| `ablations.json` | `10_ablations.py` | `08_numeros` |
| `sanity_cascade.json` | `11_sanity_checks.py` | `09_sanity` |

---

## 5. Execução e export

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe scripts\12_run_notebooks.py --check    # exit 0 = tudo roda
.\.venv\Scripts\python.exe scripts\12_run_notebooks.py --export   # HTML + PDF
```

- `--check`: executa cada notebook em kernel novo via `nbclient`, com timeout de
  120 s por notebook; aborta no primeiro erro e imprime a célula culpada.
- `--export`: `nbconvert --to html` e `--to webpdf` em `results/seminar/`, um arquivo
  por notebook + um combinado (`--combined`) para o handout.
- `requirements-notebooks.txt`: `jupyterlab`, `ipykernel`, `nbclient`, `nbformat`,
  `nbconvert[webpdf]`, `playwright`.
- `results/seminar/` **não** é commitado: é regenerável e evita binário grande no
  histórico.

---

## 6. Ativos

`notebooks/assets/cat_dog.jpg` — Wikimedia Commons, CC-BY ou CC0, redimensionada
para ~800 px e JPEG q85 (~200 KB). `CREDITS.md` registra URL, autor, licença e data
de acesso.

É o **único** dado versionado do repositório. O `README.md` ganha uma linha
explicando a exceção (200 KB de imagem de domínio público/acadêmico, necessária para
o exemplo canônico de discriminação de classe) para não contradizer a regra
"nenhum dado no git" da §8 do plano geral.

---

## 7. Ordem de produção

Um notebook nasce junto com a fase que ele demonstra — nunca antes.

| Quando | Entrega |
|---|---|
| **agora** | `deck.py`, `cases.py`, `assets/`, `00_indice.ipynb`, `01_saliency.ipynb`, `02_relus.ipynb` (sem a punchline) |
| Fase 3 | fecha a punchline de `02_relus` com `class_discrimination.json` |
| Fase 4 | `04_gradcam.ipynb` |
| Fase 5 | `03_cam.ipynb`, `05_equivalencia.ipynb` |
| Fases 6–7 | `06_guided.ipynb`, `07_counterfactual.ipynb` |
| Fases 8–9 | `08_numeros.ipynb` |
| Fase 10 | `09_sanity.ipynb`, `10_limitacoes.ipynb` |
| Fase 11 | `--export`, ensaio cronometrado, `notebooks/README.md` |

Com as Fases 1–2 concluídas, `00_indice`, `01_saliency` e `02_relus` (sem a
punchline) não dependem de código novo — só de `data/` presente na máquina.

---

## 8. Critérios de aceite

- [ ] `12_run_notebooks.py --check` retorna 0, com todos os notebooks em < 30 s cada.
- [ ] Cada notebook roda em kernel limpo, sozinho, na ordem em que é apresentado.
- [ ] Nenhum notebook importa nada fora de `requirements*.txt`.
- [ ] Todo número exibido vem de `results/metrics/*.json` ou é calculado em célula
      visível; nenhum valor digitado à mão no markdown.
- [ ] ≤ 12 células de código e ≤ 6 figuras por notebook.
- [ ] Toda figura de gradiente usa `scale_for_display(high=0.99, gamma=0.5)`;
      nenhuma usa `viz.normalize` (escala de máximo) nem reaproveita a escala
      para métrica.
- [ ] Nenhum `.ipynb` passa de 4 MB (`--check` avisa); figuras de projeção em
      dpi 110.
- [ ] `--export` gera HTML e PDF legíveis em `results/seminar/`.
- [ ] Somatório dos tempos ≤ 32 min, conferido no ensaio.

---

## 9. Lista de corte (se atrasar)

Nesta ordem:

1. `07_counterfactual` inteiro (já fora dos 10 beats).
2. `10_limitacoes` — manter só a célula da grade 14×14, cortar Grad-CAM++/Score-CAM.
3. `05_equivalencia` — reduzir a uma frase + o scatter, sem a derivação de `w_k`.
4. `09_sanity` — mostrar só a curva do VGG-16, sem a comparação GB × Grad-CAM.

---

## 10. Riscos

| Risco | Mitigação |
|---|---|
| Demo travando ao vivo | cada notebook < 30 s; `--check` antes de apresentar; outputs já commitados |
| `data/` ausente na máquina da plateia | outputs commitados; `setup()` falha com instrução do `00_download.py` |
| Drift na seleção de imagens | `MANIFEST` congelado em `cases.py` + sha1 impresso no `setup()` |
| Notebook divergir da biblioteca | regra 1 da §2 + célula final amarrando ao JSON/README |
| `.ipynb` gigante no git | dpi 110, ≤ 6 figuras, `--check` alerta > 4 MB. Referência: as grades atuais em `results/figures/` têm 1–2 MB cada com dpi 130 — projeção usa dpi menor |
| `webpdf` sem Chromium | `playwright install chromium` documentado no `notebooks/README.md` |

---

## 11. Histórico de revisões

| Data | Commit base | Mudança |
|---|---|---|
| 2026-11-15 | `b35648d` | versão inicial: 11 notebooks, cache de `results/metrics/`, `webpdf`, asset gato+cachorro |
| 2026-11-15 | `6f3902e` | pós-pull (Fases 1–2): `deck.grid` passa a delegar a `viz.figure_grid`; escala de exibição vira decisão fixa (item 7 da §0); runtimes recalibrados com backward medido em 1,1 s; `01`/`02` promovidos para "agora"; critério de tamanho do `.ipynb` corrigido para 4 MB com dpi 110 |

