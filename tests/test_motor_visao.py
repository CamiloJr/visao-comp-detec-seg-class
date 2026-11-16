"""Testes offline: não exigem webcam, download de pesos ou rede."""
from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motor_visao import (Configuracao, MODELOS, COCO_PT, abrir_imagem,
    caixa_limitada, extrair_resultado, area_poligono, nome_pt)
from renderizacao import renderizar, selecionar_na_imagem, DESCRICOES


class CPUArray:
    def __init__(self, arr):self.arr=np.asarray(arr)
    def cpu(self):return self
    def numpy(self):return self.arr


def simular_predicao(n=2):
    im=np.zeros((200,300,3),dtype=np.uint8)
    im[:,:]=(55,70,95)
    cv2.rectangle(im,(25,25),(90,170),(185,195,207),-1)
    cv2.rectangle(im,(145,80),(280,180),(65,85,205),-1)
    boxes=SimpleNamespace(
        xyxy=CPUArray([[25,25,90,170],[145,80,280,180]][:n]),
        conf=CPUArray([.93,.71][:n]),
        cls=CPUArray([0,2][:n]),
    )
    polygons=[np.array([[25,25],[90,25],[90,170],[25,170]],dtype=np.float32),
              np.array([[145,80],[280,80],[280,180],[145,180]],dtype=np.float32)][:n]
    results=SimpleNamespace(names={0:'person',2:'car'},boxes=boxes,
                            masks=SimpleNamespace(xy=polygons),
                            speed={'preprocess':1.5,'inference':60.2,'postprocess':2.1})
    return extrair_resultado(im,results,Configuracao(),100.0)


def test_80_classes_coco_e_traducao():
    assert len(COCO_PT)==80
    assert nome_pt(0,'person')=='pessoa'
    assert nome_pt(2,'car')=='carro'
    assert nome_pt(90,'something')=='something'


def test_parametros_invalidos():
    for kwargs in [dict(confianca=1.1),dict(iou=.01),dict(tamanho=301),
                   dict(modelo='inexistente'),dict(max_objetos=0),dict(camera=-1)]:
        with pytest.raises(ValueError):Configuracao(**kwargs).validar()
    assert Configuracao().validar().modelo == MODELOS[0]


def test_mascaras_e_classes_individuais():
    p=simular_predicao()
    assert len(p.objetos)==2
    assert p.objetos[0].classe_pt=='pessoa'
    assert p.objetos[1].classe_pt=='carro'
    assert p.objetos[0].confianca==pytest.approx(.93)
    assert p.objetos[0].area_px>9000
    assert p.objetos[1].fracao_imagem>0
    assert p.tempos_ms['inference']==60.2
    assert p.largura==300 and p.altura==200
    assert selecionar_na_imagem(p,40,50)==0
    assert selecionar_na_imagem(p,200,90)==1
    assert selecionar_na_imagem(p,299,199) is None


def test_etapas_nao_modificam_original():
    p=simular_predicao()
    base=p.imagem_bgr.copy()
    assert len(DESCRICOES)==5
    for etapa in range(5):
        img=renderizar(p,etapa,.5,0)
        assert img.shape==base.shape
        assert img.dtype==np.uint8
    assert np.array_equal(base,p.imagem_bgr)
    assert np.array_equal(renderizar(p,0),base)
    assert not np.array_equal(renderizar(p,1),base)
    assert not np.array_equal(renderizar(p,2),base)
    assert not np.array_equal(renderizar(p,3),base)
    assert not np.array_equal(renderizar(p,4),base)
    with pytest.raises(ValueError):renderizar(p,8)


def test_sem_deteccoes_nao_quebra():
    p=simular_predicao(0)
    p.objetos=[]
    for etapa in range(5):
        img=renderizar(p,etapa)
        assert np.array_equal(img,p.imagem_bgr)
    assert selecionar_na_imagem(p,10,10) is None


def test_area_uniao_e_caixa_limites():
    poly=[np.asarray([[10,10],[50,10],[50,50],[10,50]],np.int32).reshape((-1,1,2))]
    assert area_poligono(poly,100,80)>1500
    assert area_poligono([],100,80)==0
    assert caixa_limitada(np.asarray([-2,-3,150,200]),100,80)==(0,0,99,79)


def test_abrir_imagem_fornecida_pelo_usuario(tmp_path):
    path=tmp_path/'foto.jpg'
    imagem_original=np.full((32,48,3),120,dtype=np.uint8)
    assert cv2.imwrite(str(path),imagem_original)
    imagem=abrir_imagem(path)
    assert imagem.ndim==3 and imagem.shape[2]==3
    assert min(imagem.shape[:2])>=20
    with pytest.raises(ValueError):abrir_imagem(__file__)
