"""Motor de inferência YOLO11-seg e extração explícita das previsões.

A inferência calcula caixas, classes e máscaras na mesma passagem da rede.
As vistas em renderizacao.py são etapas de explicação, não redes em série.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import time

import cv2
import numpy as np

MODELOS = ('yolo11n-seg.pt', 'yolo11s-seg.pt')
TAMANHOS = (320, 416, 512, 640, 768)

# Traduções de nomes das classes do COCO, usadas apenas para exibição.
COCO_PT = (
    'pessoa','bicicleta','carro','motocicleta','avião','ônibus','trem','caminhão',
    'barco','semáforo','hidrante','placa de pare','parquímetro','banco','pássaro',
    'gato','cachorro','cavalo','ovelha','vaca','elefante','urso','zebra','girafa',
    'mochila','guarda-chuva','bolsa','gravata','mala','frisbee','esquis',
    'snowboard','bola esportiva','pipa','taco de beisebol','luva de beisebol',
    'skate','prancha de surfe','raquete de tênis','garrafa','taça','xícara',
    'garfo','faca','colher','tigela','banana','maçã','sanduíche','laranja',
    'brócolis','cenoura','cachorro-quente','pizza','rosquinha','bolo','cadeira',
    'sofá','vaso de planta','cama','mesa de jantar','vaso sanitário','televisão',
    'notebook','mouse','controle remoto','teclado','celular','micro-ondas',
    'forno','torradeira','pia','geladeira','livro','relógio','vaso de flores',
    'tesoura','urso de pelúcia','secador de cabelo','escova de dentes'
)


@dataclass(frozen=True)
class Configuracao:
    modelo: str = MODELOS[0]
    confianca: float = 0.35
    iou: float = 0.50
    tamanho: int = 640
    opacidade: float = 0.46
    max_objetos: int = 30
    camera: int = 0

    def validar(self) -> 'Configuracao':
        if self.modelo not in MODELOS:
            raise ValueError('Modelo não permitido.')
        if not (0.05 <= self.confianca <= .95 and .10 <= self.iou <= .90):
            raise ValueError('Limites de confiança ou IoU inválidos.')
        if self.tamanho not in TAMANHOS or not (0 <= self.opacidade <= 1):
            raise ValueError('Tamanho ou opacidade inválidos.')
        if not (1 <= self.max_objetos <= 100 and 0 <= self.camera <= 9):
            raise ValueError('Quantidade de objetos ou câmera inválida.')
        return self


@dataclass
class Objeto:
    classe_id: int
    classe_en: str
    classe_pt: str
    confianca: float
    caixa: tuple[int, int, int, int]
    poligonos: list[np.ndarray] = field(default_factory=list)
    area_px: int = 0
    fracao_imagem: float = 0.0


@dataclass
class Predicao:
    imagem_bgr: np.ndarray
    objetos: list[Objeto]
    tempos_ms: dict[str, float]
    tempo_total_ms: float
    modelo: str
    tamanho: int
    confianca: float
    iou: float
    origem: str = 'Imagem'

    @property
    def largura(self) -> int:
        return int(self.imagem_bgr.shape[1])

    @property
    def altura(self) -> int:
        return int(self.imagem_bgr.shape[0])


def nome_pt(id_classe: int, nome_ingles: str) -> str:
    """COCO usa 80 classes e os índices correspondem a sua enumeração."""
    return COCO_PT[id_classe] if 0 <= id_classe < len(COCO_PT) else nome_ingles


def caixa_limitada(xyxy: np.ndarray, largura: int, altura: int) -> tuple[int, int, int, int]:
    x1,y1,x2,y2 = [int(round(float(n))) for n in xyxy]
    x1 = min(max(x1, 0), largura - 1)
    x2 = min(max(x2, x1), largura - 1)
    y1 = min(max(y1, 0), altura - 1)
    y2 = min(max(y2, y1), altura - 1)
    return x1,y1,x2,y2


def area_poligono(poligonos: list[np.ndarray], largura: int, altura: int) -> int:
    """Conta pixels da união de máscaras, mesmo para polígonos sobrepostos."""
    if not poligonos: return 0
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    cv2.fillPoly(mascara, poligonos, 255)
    return int(cv2.countNonZero(mascara))


def extrair_resultado(imagem_bgr: np.ndarray, resultado, configuracao: Configuracao, tempo_total_ms: float, origem='Imagem') -> Predicao:
    """Converte um Ultralytics Results sem depender de formas específicas de GPU.

    `masks.xy` são contornos nas dimensões da imagem original; a caixa é
    interpretada em pixels originais. `masks` pode ser None.
    """
    h, w = imagem_bgr.shape[:2]
    nomes = resultado.names
    boxes = resultado.boxes
    objetos: list[Objeto] = []
    contornos = resultado.masks.xy if getattr(resultado, 'masks', None) is not None else []
    if boxes is not None:
        caixas = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.cls.cpu().numpy()
        for i, (xyxy, score, classe_id) in enumerate(zip(caixas, confs, ids)):
            classe_id = int(classe_id)
            nome_en = nomes.get(classe_id, str(classe_id)) if isinstance(nomes, dict) else nomes[classe_id]
            poligonos = []
            if i < len(contornos) and len(contornos[i]) >= 3:
                poly = np.rint(np.asarray(contornos[i])).astype(np.int32)
                poly[:, 0] = np.clip(poly[:, 0], 0, w-1)
                poly[:, 1] = np.clip(poly[:, 1], 0, h-1)
                poligonos.append(poly.reshape((-1,1,2)))
            area = area_poligono(poligonos, w, h)
            objetos.append(Objeto(classe_id, nome_en, nome_pt(classe_id, nome_en),
                                  float(score), caixa_limitada(xyxy, w, h),
                                  poligonos, area, area/(w*h)))
    return Predicao(
        imagem_bgr=imagem_bgr.copy(), objetos=objetos,
        tempos_ms={str(k):float(v) for k,v in (getattr(resultado,'speed',{}) or {}).items()},
        tempo_total_ms=tempo_total_ms, modelo=configuracao.modelo,
        tamanho=configuracao.tamanho, confianca=configuracao.confianca,
        iou=configuracao.iou, origem=origem,
    )


class Detector:
    """Carrega pesos apenas na primeira inferência; roda inteiramente local depois.

    Na primeira utilização, o pacote Ultralytics pode baixar os pesos de modelo.
    """
    def __init__(self):
        self._modelo = None
        self._nome = None

    def prever(self, imagem_bgr: np.ndarray, cfg: Configuracao, origem='Imagem') -> Predicao:
        cfg.validar()
        if self._modelo is None or self._nome != cfg.modelo:
            from ultralytics import YOLO
            self._modelo = YOLO(cfg.modelo)
            self._nome = cfg.modelo
        inicio = time.perf_counter()
        resultado = self._modelo.predict(
            source=imagem_bgr, conf=cfg.confianca, iou=cfg.iou,
            imgsz=cfg.tamanho, max_det=cfg.max_objetos,
            retina_masks=True, verbose=False, device='cpu',
        )[0]
        total = (time.perf_counter() - inicio)*1000
        return extrair_resultado(imagem_bgr, resultado, cfg, total, origem)


def abrir_imagem(path: str | Path) -> np.ndarray:
    """Suporta caminhos UTF-8, incluindo arquivos nomeados com acentos no Windows."""
    p = Path(path)
    raw = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None: raise ValueError('Arquivo não contém uma imagem JPG, PNG, WEBP ou BMP válida.')
    if min(img.shape[:2]) < 20: raise ValueError('Imagem muito pequena para este exemplo.')
    h, w = img.shape[:2]
    # Preserva proporção e limita o volume de pixels para manter a aula fluida.
    if max(h, w) > 2400:
        f = 2400/max(h,w)
        img = cv2.resize(img,(round(w*f),round(h*f)),interpolation=cv2.INTER_AREA)
    return img
