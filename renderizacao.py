"""Renderização didática: detecção, máscaras, classes e resultado integrado."""
from __future__ import annotations
import cv2
import numpy as np
from motor_visao import Predicao

ETAPAS = ('Original', 'Detecção', 'Segmentação', 'Classificação', 'Resultado')
DESCRICOES = (
    ('01  ENTRADA DA IMAGEM',
     'Uma imagem RGB chega da câmera ou do arquivo. Cada pixel contém três canais de cor.',
     'Antes da rede, a imagem é redimensionada com preservação de proporção (letterbox) e normalizada.'),
    ('02  DETECÇÃO — ONDE?',
     'O YOLO prevê caixas delimitadoras (x1, y1, x2, y2) para as instâncias encontradas.',
     'As caixas menos confiáveis são filtradas; a supressão de não máximos (NMS) usa IoU.'),
    ('03  SEGMENTAÇÃO — QUAIS PIXELS?',
     'A máscara delimita quais pixels pertencem a cada instância detectada, e não só sua caixa.',
     'Duas pessoas da mesma classe podem ter máscaras diferentes: segmentação de instâncias.'),
    ('04  CLASSIFICAÇÃO — O QUE É?',
     'A rede atribui a cada objeto uma categoria conhecida (pessoa, carro, bicicleta etc.).',
     'O valor exibido é a confiança do modelo; não é garantia de acerto nem calibração.'),
    ('05  RESULTADO INTEGRADO',
     'Sobrepomos caixas, máscaras, classes e confianças previstas na imagem original.',
     'As etapas 02–04 são vistas separadas da mesma inferência, não três algoritmos sucessivos.'),
)

PALETA = ((55,214,223),(255,178,80),(165,124,238),(89,211,147),
          (248,118,136),(100,165,255),(242,211,104),(205,137,190))


def cor_objeto(classe_id: int) -> tuple[int,int,int]:
    """Cor BGR estável por classe."""
    r,g,b=PALETA[classe_id%len(PALETA)]
    return b,g,r


def desenhar_mascaras(frame: np.ndarray, pred: Predicao, opacidade: float, selecionado: int|None=None, contorno=True) -> np.ndarray:
    overlay=frame.copy()
    for n,obj in enumerate(pred.objetos):
        if not obj.poligonos: continue
        cor=cor_objeto(obj.classe_id)
        if selecionado==n: cor=(0,220,255)
        cv2.fillPoly(overlay,obj.poligonos,cor)
    result=cv2.addWeighted(overlay,float(opacidade),frame,1-float(opacidade),0)
    if contorno:
        for n,obj in enumerate(pred.objetos):
            cor=(0,220,255) if selecionado==n else cor_objeto(obj.classe_id)
            if obj.poligonos: cv2.polylines(result,obj.poligonos,True,cor,2,cv2.LINE_AA)
    return result


def desenhar_caixas(frame: np.ndarray, pred: Predicao, rotulos: bool, selecionado: int|None=None) -> np.ndarray:
    img=frame.copy()
    h,w=img.shape[:2]
    escala=max(.50,min(1.20,w/960))
    fonte=cv2.FONT_HERSHEY_SIMPLEX
    for idx,obj in enumerate(pred.objetos):
        cor=(0,220,255) if selecionado==idx else cor_objeto(obj.classe_id)
        x1,y1,x2,y2=obj.caixa
        cv2.rectangle(img,(x1,y1),(x2,y2),cor,3 if selecionado==idx else 2)
        if rotulos:
            # Índices são intencionalmente usados para que a tabela mostre os nomes PT.
            nome=f'{idx+1:02d}  {obj.classe_en}  {obj.confianca*100:.0f}%'
            (tw,th),base=cv2.getTextSize(nome,fonte,escala,2)
            x=max(0,min(x1,w-tw-12))
            y=max(th+base+8,y1)
            cv2.rectangle(img,(x,y-th-base-8),(min(w-1,x+tw+10),y+4),cor,-1)
            cv2.putText(img,nome,(x+5,y-base-3),fonte,escala,(10,16,23),2,cv2.LINE_AA)
    return img


def renderizar(pred: Predicao, etapa: int, opacidade: float=.46, selecionado: int|None=None) -> np.ndarray:
    frame=pred.imagem_bgr
    if etapa==0: return frame.copy()
    if etapa==1: return desenhar_caixas(frame,pred,False,selecionado)
    if etapa==2: return desenhar_mascaras(frame,pred,opacidade,selecionado)
    if etapa==3: return desenhar_caixas(frame,pred,True,selecionado)
    if etapa==4:
        return desenhar_caixas(desenhar_mascaras(frame,pred,opacidade,selecionado),pred,True,selecionado)
    raise ValueError('Etapa inválida.')


def selecionar_na_imagem(pred: Predicao, px: float, py: float) -> int|None:
    """Prioriza objetos pequenos e contorno de máscara, depois as caixas."""
    candidatos=[]
    for i,o in enumerate(pred.objetos):
        x1,y1,x2,y2=o.caixa
        if not (x1<=px<=x2 and y1<=py<=y2): continue
        mascara=any(cv2.pointPolygonTest(poly,(float(px),float(py)),False)>=0 for poly in o.poligonos)
        area=max(1,(x2-x1)*(y2-y1))
        candidatos.append((0 if mascara else 1,area,i))
    if not candidatos: return None
    return min(candidatos)[2]
