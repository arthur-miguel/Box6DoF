# Estimação de pose 6-DoF de caixas para paletização robótica

Sistema de visão computacional para uma célula de paletização com um
manipulador UR10 e uma webcam de baixo custo (640×360) montada no flange, em
configuração *eye-in-hand*. Os quatro vértices da face superior de cada caixa
são detectados na imagem e usados como keypoints para resolver o problema
Perspective-n-Point (PnP); a pose resultante é transportada ao referencial da
base do robô pela cadeia cinemática calibrada.

Três estratégias de detecção de keypoints estão implementadas sobre o mesmo
estimador, o que permite compará-las em igualdade de condições:

1. marcadores fiduciários ArUco (baseline de precisão);
2. segmentação por cor no espaço HSV, com discriminação da face superior por
   limiarização de Otsu no canal V;
3. segmentação de instâncias com um modelo YOLO26 nano ajustado por
   fine-tuning para caixas de papelão.

Código desenvolvido no Iris Lab (UFSC, Campus Joinville) para a disciplina
TVR410001 — Visão Computacional. Autores: Arthur M. P. Gabardo, Bruno Z. R.
da Silva, Vitor F. de Borba e Anelize Z. Salvi.

## Estrutura

```
calibracao/
  0_captura_metadata.py       coleta automatizada de imagens + poses via RTDE
  1_intrinsic_calibration.py  calibração intrínseca (Zhang, duas passagens)
  2_handeye_calibration.py    calibração hand-eye (5 métodos do OpenCV)
pose-estimation/
  estimar_pose.py             detectores de keypoints, PnP e cadeia cinemática
  paletizacao.py              execução em linha de comando (janelas OpenCV)
  gui_app.py                  interface gráfica (CustomTkinter)
  plot_reprojection_errors.py boxplots de erro de reprojeção a partir dos logs
  handeye_calibration.json    exemplo de calibração da célula usada no trabalho
yolo-finetune/
  train_segmentation.py       fine-tuning do yolo26n-seg (treino/inferência/export)
  results.csv                 métricas por época do treinamento realizado
  Dataset/data.yaml           configuração do dataset (imagens baixadas à parte)
```

## Instalação

Requer Python 3.10 ou superior.

```
pip install -r requirements.txt
```

O pacote `ur-rtde` só é necessário para operar o robô físico. Sem ele (ou com
a flag `--no-robot`), os scripts de pose usam um robô simulado que executa a
cinemática direta localmente, o que permite testar todo o pipeline de visão
sem acesso à célula.

## Uso

### 1. Calibração

Coleta de dados com o robô (varredura esférica sobre um tabuleiro de xadrez,
com pré-validação de waypoints por cinemática inversa):

```
cd calibracao
python 0_captura_metadata.py
```

Ajuste `IP_DO_ROBO` e `CAMERA_INDEX` no topo do script antes de rodar. A
sessão gera uma pasta `dados_handeye/<timestamp>/` com as imagens e um
`metadata.json` contendo pose TCP e ângulos de junta por captura.

Calibração intrínseca (tabuleiro 9×6, quadrados de 25 mm por padrão):

```
python 1_intrinsic_calibration.py --sessions dados_handeye/<timestamp>
```

Calibração hand-eye, resolvida pelos cinco métodos clássicos do OpenCV
(Tsai–Lenz, Park–Martin, Horaud–Dornaika, Andreff e Daniilidis), com relatório
de resíduos AX=XB para escolha do melhor:

```
python 2_handeye_calibration.py --sessions <timestamp> --robot-model UR10
```

Como a câmera da célula gira solidária ao último punho, o script aceita
`--mount-joint 5` para computar a cinemática direta apenas até a penúltima
junta, absorvendo a rotação do punho na transformação estimada.

### 2. Fine-tuning do modelo de segmentação

O dataset usado é o [Cardboard Box Detection](https://universe.roboflow.com/ps-uqsf0/cardboard-box-detection-mxqjh-yhbtc/dataset/1)
(Roboflow Universe, licença CC BY 4.0). Baixe-o no formato de segmentação
YOLO e extraia em `yolo-finetune/Dataset/`, mantendo o `data.yaml` incluído.

```
cd yolo-finetune
python train_segmentation.py                 # 150 épocas, imgsz 640, batch 128
python train_segmentation.py --export        # copia o best.pt para ~/Downloads
```

Reduza `--batch` conforme a VRAM disponível. O script também aceita
`--predict --source imagem.jpg` para inferência avulsa e `--visualize` para
testar o modelo em uma imagem aleatória do conjunto de teste.

### 3. Estimação de pose e paletização

```
cd pose-estimation
python paletizacao.py --calib handeye_calibration.json \
    --seg-backend ultralytics --seg-model caminho/para/best.pt
```

Teclas: `m` alterna o modo de detecção (auto/HSV/ML/aruco), `h` move o robô
para a pose inicial, `s` salva um snapshot, espaço inicia a sequência de
pick-and-place por visual servoing e `q` encerra.

A versão com interface gráfica expõe as mesmas funções, com ajuste em tempo
de execução dos limiares HSV, da confiança do modelo e do IP do robô:

```
python gui_app.py --calib handeye_calibration.json --no-robot
```

O `handeye_calibration.json` incluído corresponde à câmera e à montagem da
célula usada no trabalho; para outro hardware, gere o seu com os scripts da
pasta `calibracao/`.

### 4. Análise dos resultados

Cada sequência executada grava um `sequence_log.json` em `logs/`. O gráfico
comparativo de erro de reprojeção por método e cenário é gerado com:

```
python plot_reprojection_errors.py
```