# Visão Computacional Interativa — Detecção, Segmentação e Classificação

Aplicação didática em Python para **Aula 7 — Aplicações Contemporâneas da Inteligência Artificial**. A interface foi concebida a partir do estilo e da organização do projeto **Classificação Interativa — Aprendizado Supervisionado** fornecido como referência: janela Pygame, painéis numerados, controles visíveis e comentários explicativos.

O software utiliza **YOLO11-seg**, um modelo de visão computacional previamente treinado, para identificar **instâncias de objetos** em arquivos de imagem e vídeos da webcam. Um mesmo modelo prevê **caixas**, **máscaras** e **classes**. As cinco visualizações são *vistas didáticas* das saídas, e não cinco modelos nem cinco execuções consecutivas.

## Funcionalidades

- **Abrir uma imagem** JPG, PNG, BMP ou WEBP do computador.
- **Iniciar webcam**, processar quadros e congelar o resultado selecionando **Capturar quadro**.
- **Cinco vistas explicadas**: imagem original, detecção, segmentação, classificação e resultado integrado.
- **Interação direta**: clicar na imagem processada para destacar uma instância. A lista à direita mostra as classes e a confiança de cada previsão.
- **Parâmetros configuráveis**: variante YOLO11n/s, confiança mínima, limiar de IoU na supressão de caixas, dimensão de entrada, opacidade das máscaras, máximo de objetos e índice da câmera.
- **Métricas por quadro**: dimensões da imagem, número de instâncias e duração aproximada de uma inferência em milissegundos.
- **Exportar PNG** da imagem anotada, sem enviar o conteúdo a serviços externos.
- **Interface adaptativa**: janela redimensionável, preservando proporções da interface original.

## Windows — execução simplificada

1. Extraia o ZIP inteiro, mantendo `tests` e os arquivos `.py` juntos.
2. Instale o Python 3.12 ou 3.13 (64 bits) e verifique a opção **Add python.exe to PATH**. O iniciador tenta usar Python 3.13, depois 3.12 e depois o Python padrão.
3. Clique duas vezes em **`INICIAR_WINDOWS.bat`**.
4. Na primeira execução, o iniciador cria `.venv` e instala dependências. **Internet é necessária nesta etapa**.
5. O aplicativo abrirá sem nenhuma imagem carregada. Clique em **Abrir imagem** e selecione uma imagem do computador para usar **Analisar**, ou clique em **Iniciar webcam** para iniciar a captura. Na primeira inferência, o Ultralytics baixa os pesos `yolo11n-seg.pt`, se ainda não estiverem em cache. Nas execuções posteriores, os pesos podem ser reutilizados localmente.

**Observação:** a instalação pode ser grande porque PyTorch é uma dependência do Ultralytics. A execução padrão usa CPU; uma GPU não é necessária. Em um computador sem GPU, a taxa de quadros pode ser baixa, especialmente usando `YOLO11s-seg` e resolução 768.

### Execução manual no PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe visao_computacional.py
```

Se não tiver Python 3.13, use `py -3.12`. Não é necessário ativar o ambiente virtual para executar esses comandos.

### macOS/Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python visao_computacional.py
```

Em Linux, pode ser necessário instalar suporte Tkinter (diálogo de arquivos), X11/Wayland e drivers de câmera. Pygame e OpenCV requerem ambiente gráfico. Em macOS pode ser necessário conceder permissão para uso da câmera.

## Experiência sugerida em sala (10–15 minutos)

1. **Original**: pressione `1`, identifique o que um ser humano entende da cena e discuta a noção de pixels/canais RGB.
2. **Detecção**: clique em **Analisar** e pressione `2`. Veja as caixas retangulares. Pergunte: *O algoritmo identificou a posição ou o contorno exato?*
3. **Segmentação**: pressione `3`. Veja as máscaras coloridas, que podem distinguir dois objetos de uma mesma classe.
4. **Classificação**: pressione `4`. Associe a classe prevista à confiança, distinguindo previsão de certeza matemática.
5. **Resultado integrado**: pressione `5`. Clique em uma máscara ou caixa e destaque a instância correspondente.
6. **Parâmetros**: aumente a confiança mínima para `0.75` e execute novamente. Depois diminua para `0.15`. Compare quantos resultados passam no filtro.
7. **IoU**: altere o limiar de sobreposição (usado no pós-processamento de caixas), reanalise e observe possíveis diferenças.
8. **Webcam**: clique em **Iniciar webcam** e use objetos reais. Clique em **Capturar quadro** para congelar uma análise antes de explicar.
9. **Fechamento**: discuta erros, vieses de dados, confiança e o que é necessário para reconhecer novas classes ou ambientes muito diferentes do treino.

Atalhos: `1` a `5` para alterar a visualização, `Espaço` para analisar, `S` para exportar PNG, `H` para ajuda, `Esc` para fechar a ajuda ou o programa.

## Como o algoritmo funciona

```text
Imagem RGB ou quadro da webcam
         │
         ▼
Pré-processamento (redimensionamento/letterbox, normalização)
         │
         ▼
Rede neural profunda YOLO11-seg (pesos pré-treinados)
         │
         ├────► caixas delimitadoras (detecção / posição)
         ├────► classes e escores (classificação por objeto)
         └────► máscaras por instância (segmentação)
         │
         ▼
Filtragem por confiança, tratamento de sobreposições / NMS (IoU)
         │
         ▼
Sobreposição de caixas, rótulos e máscaras na imagem original
```

**Importante:** as saídas de detecção, segmentação e classificação são produzidas de maneira conjunta pelo modelo. A ordem em que elas são mostradas nos botões é uma **organização pedagógica**, não uma alegação de que um algoritmo de segmentação é executado depois de um algoritmo de detecção e antes de um classificador independente.

### Configurações

| Parâmetro | Valor inicial | O que altera |
|---|---:|---|
| Modelo | `yolo11n-seg.pt` | Rede compacta e rápida (*nano*). A opção `s` aumenta a capacidade e o custo computacional. |
| Confiança mínima | `0.35` | Filtra previsões de escores abaixo do limiar. Não modifica os pesos da rede. |
| IoU | `0.50` | Limiar de sobreposição de caixas usado na seleção/supressão de previsões redundantes. |
| Tamanho de entrada | `640` | Tamanho solicitado para a entrada da rede. Em geral, maior tamanho aumenta o custo. |
| Opacidade | `46%` | Apenas a transparência visual das máscaras; não afeta o resultado da rede. |
| Máximo de objetos | `30` | Quantidade máxima de previsões por imagem. |
| Câmera | `0` | Índice da câmera usado por OpenCV (0 costuma ser a padrão). |

A confiança de cada objeto é um **escore produzido pelo modelo**; não é equivalente a probabilidade calibrada de a previsão estar correta. A área percentual da máscara pode ser calculada a partir dos pixels que ela ocupa, mas **não representa distância real, área física ou profundidade**.

### Conceitos para explicar

- **Detecção** responde *onde está cada objeto?* e desenha uma caixa de coordenadas `(x1,y1,x2,y2)`.
- **Segmentação de instâncias** responde *quais pixels pertencem a cada objeto individual?* e desenha uma máscara.
- **Classificação por objeto** responde *qual categoria conhecida foi prevista para cada instância?*. Diferentemente da **classificação de imagem inteira**, uma foto pode ter várias classes simultaneamente.
- **IoU (Intersection over Union)** compara duas regiões: `IoU = área da interseção / área da união`. NMS utiliza o parâmetro de IoU para lidar com detecções redundantes; seu efeito depende da implementação e das detecções geradas.
- **Inferência versus treinamento:** aqui não há atualização de pesos nem criação de uma base de treinamento, pois a aula focaliza a aplicação de um modelo já treinado.
- **Cobertura limitada:** o modelo padrão conhece as **80 classes do COCO**. Objetos fora desse repertório podem não ser detectados ou podem ser confundidos com classes disponíveis.
- **Tempo de inferência** é o tempo de uma chamada do modelo no ambiente atual; não equivale necessariamente ao FPS da webcam ou à latência total da interface.

## Organização dos arquivos

```text
Visao_Computacional_Interativa/
├── visao_computacional.py      Interface, fluxos didáticos e controles
├── motor_visao.py             YOLO, parâmetros, classes e extração de resultados
├── renderizacao.py            Caixas, contornos, máscaras, saída final
├── trabalhador.py             Thread para câmera e inferência sem travar UI
├── INICIAR_WINDOWS.bat        Ambiente virtual, instalação, inicialização
├── requirements.txt           Dependências
└── tests/
    ├── test_motor_visao.py
    └── test_trabalhador.py
```

## Testes automáticos (sem rede e sem webcam)

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Testes simulam saídas de YOLO e verificam tradução de classes, máscaras, caixas, vistas da interface de processamento e comportamento assíncrono. **Não** são um teste de acurácia em um conjunto rotulado. O desempenho do modelo exige avaliação separada com dados independentes.

## Observações de qualidade e privacidade

- Por padrão o aplicativo processa localmente as imagens fornecidas. Não envia as fotografias ou os quadros da webcam para um serviço remoto. **Na primeira execução, há download de pesos e dependências** quando ainda não instalados, o que requer rede.
- Não use a saída como sistema de segurança, identificação pessoal, diagnóstico ou decisão de alto risco. Pessoas e objetos podem ser classificados incorretamente.
- O programa não faz reconhecimento de identidade facial nem classificação de emoções, o que exigiria tarefas/modelos distintos e cuidados metodológicos adicionais.
- A biblioteca **Ultralytics** possui licenciamento próprio (**AGPL-3.0** e opções empresariais); consulte as condições antes de redistribuir o pacote em produtos ou sistemas comerciais. Este projeto didático não inclui os pesos binários do modelo no ZIP.

## Referências técnicas

- Documentação oficial de YOLO11: https://docs.ultralytics.com/models/yolo11/
- Segmentação de instâncias: https://docs.ultralytics.com/tasks/segment/
- Instalação Ultralytics: https://docs.ultralytics.com/quickstart/
- PyTorch: https://pytorch.org/get-started/locally/
