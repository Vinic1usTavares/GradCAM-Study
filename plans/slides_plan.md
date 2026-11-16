# Plano de Slides — Seminário Grad-CAM

Este documento descreve o plano estruturado de slides para a apresentação do seminário sobre o artigo **Grad-CAM**, contrastando a teoria, as motivações, a formulação matemática e os resultados originais com os achados da nossa reimplementação *do zero*.

---

## Seção 1: Introdução e Motivação (Abertura)

### Slide 1: Título e Contexto
- **Ideia central**: Apresentar o tema e a grande questão do estudo.
- **Tópicos para a fala**:
  - Título do paper: *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* (Selvaraju et al., IJCV 2019 / arXiv 2016).
  - O problema da "caixa-preta" em Redes Neurais Profundas (CNNs).
  - Objetivo: Explicar *por que* o modelo tomou determinada decisão de forma visual e interpretável.

### Slide 2: O Gancho Clínico (A Figura 8 do Artigo)
- **Ideia central**: Mostrar o perigo de modelos sem interpretabilidade confiável.
- **Tópicos para a fala**:
  - Exemplo clássico do artigo (diagnóstico médico vs. viés de fundo).
  - Um modelo pode acertar a predição olhando para o lugar errado (ex: artefatos no fundo da imagem e não na lesão).
  - A necessidade de mapas de localização que sejam específicos de classe.

---

## Seção 2: Evolução dos Métodos de Saliência

### Slide 3: Saliência por Gradiente Simples (Vanilla Backprop)
- **Ideia central**: O primeiro passo — gradiente da saída em relação à entrada ($\partial y^c / \partial x$).
- **Tópicos para a fala**:
  - Como funciona: retropropagação direta da classe de interesse até os pixels da imagem.
  - O que o mapa revela: contornos ruidosos e dispersos.
  - Limitação principal: alta sensibilidade a ruídos e detalhe excessivo (difícil de discernir o objeto focal).

### Slide 4: As Três Regras de ReLU e suas Estéticas
- **Ideia central**: Comparar Vanilla, Deconvolution (Zeiler & Fergus) e Guided Backpropagation (Springenberg).
- **Tópicos para a fala**:
  - A diferença reside na forma como o gradiente lida com as ativações ReLU na volta.
  - *Vanilla*: blob difuso com ruído fino.
  - *Deconvolution*: estrutura de grade ruidosa.
  - *Guided Backprop*: contornos limpos e nítidos (texturas finas).
  - **O grande defeito quantitativo**: o Spearman entre classes (`map_cat` vs `map_dog`) na mesma imagem é de **0,999** (em deconv/guided). Ou seja, esses mapas localizam *algum* objeto, mas não dizem *qual* classe está sendo vista.

---

## Seção 3: Da Arquitetura CAM ao Grad-CAM

### Slide 5: Class Activation Mapping (CAM)
- **Ideia central**: O precursor direto do Grad-CAM (Zhou et al., 2016).
- **Tópicos para a fala**:
  - Arquitetura restrita: exige Global Average Pooling (GAP) logo antes da camada totalmente conectada (`fc`).
  - Como funciona: os pesos da camada `fc` servem como importância direta para ponderar os mapas de ativação da última camada convolucional.
  - Vantagem: localiza perfeitamente.
  - **Desvantagem**: exige modificação na arquitetura e retreinamento (perda de ~3% no top-1 do VGG, por exemplo).

### Slide 6: Derivação do Grad-CAM (Equações 1 e 2)
- **Ideia central**: Como generalizar o CAM para *qualquer* arquitetura CNN sem retreinar.
- **Tópicos para a fala**:
  - **Eq. 1 (Pesos de importância $\alpha_k^c$)**: Média global dos gradientes do score da classe $c$ ($y^c$) em relação aos mapas de ativação da camada convolucional ($A^k$).
  - **Eq. 2 (Combinação linear com ReLU)**: SOMA ponderada seguida de ReLU.
  - O ReLU não é opcional: evita que características que influenciam negativamente a classe entrem no mapa.
  - Funciona em qualquer rede com camadas convolucionais (classificação, legendagem, VQA).

---

## Seção 4: Composição e Contrafactuais

### Slide 7: Guided Grad-CAM (Resolução × Discriminação)
- **Ideia central**: Unir o melhor dos dois mundos.
- **Tópicos para a fala**:
  - Combinação: $L_{\text{Guided Grad-CAM}} = \text{Upsample}(L_{\text{Grad-CAM}}) \odot \text{Guided Backprop}$.
  - Resultado: Mantém a discriminação de classe e localização grosseira do Grad-CAM, mas recupera a nitidez e o detalhe fino do Guided Backprop.
  - Evidência nas nossas medições: índice de detalhe de ~1,20 (equivalente ao Guided Backprop) e boa concentração dentro da *bounding box* correta.

### Slide 8: Explicabilidade Contrafactual
- **Ideia central**: Perguntar ao modelo o que *não* faz parte da classe.
- **Tópicos para a fala**:
  - Como é feito: inverter o sinal dos pesos $\alpha$ antes do ReLU ($-\alpha$).
  - O que o mapa mostra: o pico se desloca para evidências complementares ou objetos vizinhos na cena (ex: o pico se moveu ~9 células em um grid 14×14 na nossa implementação).

---

## Seção 5: Sanity Checks e Nossos Resultados vs. O Artigo

### Slide 9: Sanity Checks (Adebayo et al., 2018)
- **Ideia central**: O modelo realmente usa a rede aprendida ou o mapa é apenas uma borda fixa da imagem?
- **Tópicos para a fala**:
  - O teste de randomização em cascata de parâmetros (de cima para baixo).
  - Resultado surpreendente:
    - Grad-CAM e Guided Grad-CAM perdem metade da correlação logo no primeiro passo ($k=1$, reamostrando apenas a camada linear final).
    - Guided Backprop resiste até quase o fim ($k=16$), provando que ele depende fortemente das propriedades de borda das primeiras camadas de convolução (quase independente dos pesos aprendidos da rede).

### Slide 10: Nossos Números vs. O Artigo (Realidade da CPU)
- **Ideia central**: Comparar o que o artigo relata com a nossa execução prática do zero em CPU.
- **Tópicos para a fala**:
  - Restrições de ambiente superadas: execução inteiramente em CPU (~330ms/forward no VGG-16).
  - Validação da equivalência matemática: CAM e Grad-CAM coincidem numericamente na ResNet18 ($\text{Spearman} = 1.000000$, erro máximo $< 10^{-5}$) quando lidos na exata mesma camada.
  - Conclusão geral: A reimpressão fiel do zero valida os conceitos teóricos fundamentais do artigo original.

---

## Seção 6: Conclusão e Limitações

### Slide 11: Limitações e Trabalhos Futuros
- **Ideia central**: Olhar criticamente para a ferramenta.
- **Tópicos para a fala**:
  - Resolução limitada (ex: grids $14 \times 14$ interpolados no VGG-16).
  - Degradação do mapa em camadas rasas ou classes altamente ambíguas.
  - Alternativas modernas: Grad-CAM++, Score-CAM.
  - Agradecimentos e abertura para perguntas.
