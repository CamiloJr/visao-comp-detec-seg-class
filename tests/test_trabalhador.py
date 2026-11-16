"""Testa a comunicação UI-worker sem pesos do YOLO nem câmera física."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from motor_visao import Configuracao, Predicao
from trabalhador import Trabalhador


class DetectorFalso:
    def prever(self, frame, cfg, origem):
        return Predicao(frame.copy(),[],{'preprocess':1,'inference':4,'postprocess':1},
                        6,cfg.modelo,cfg.tamanho,cfg.confianca,cfg.iou,origem)


def test_worker_resultado_assincrono():
    worker=Trabalhador(detector=DetectorFalso())
    try:
        worker.analisar(np.zeros((64,96,3),dtype=np.uint8),Configuracao(),'Teste offline')
        limite=time.monotonic()+3
        while time.monotonic()<limite and worker.snapshot().revisao<1:
            time.sleep(.01)
        estado=worker.snapshot()
        assert estado.revisao==1
        assert estado.predicao.origem=='Teste offline'
        assert len(estado.predicao.objetos)==0
        assert not estado.erro
    finally:
        worker.fechar()
