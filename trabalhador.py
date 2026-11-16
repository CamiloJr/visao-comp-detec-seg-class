"""Executa a inferência e captura de webcam sem travar a janela Pygame."""
from __future__ import annotations
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from motor_visao import Configuracao, Detector, Predicao


@dataclass
class Estado:
    predicao: Predicao | None = None
    status: str = 'Escolha uma imagem ou webcam para começar.'
    ocupado: bool = False
    camera_ativa: bool = False
    revisao: int = 0
    erro: bool = False


class Trabalhador:
    def __init__(self, detector: Any = None):
        self._lock=threading.Lock()
        self._fila: queue.Queue[tuple] = queue.Queue()
        self._estado=Estado()
        self._detector=detector if detector is not None else Detector()
        self._thread=threading.Thread(target=self._loop,daemon=True,name='motor-visao')
        self._thread.start()

    def _escrever(self, **updates):
        with self._lock:
            for k,v in updates.items():setattr(self._estado,k,v)

    def snapshot(self)->Estado:
        with self._lock:
            return Estado(**vars(self._estado))

    def analisar(self, imagem: np.ndarray, cfg: Configuracao, origem: str):
        self._fila.put(('imagem',imagem.copy(),cfg.validar(),origem))
        self._escrever(ocupado=True,status='Imagem recebida. Preparando a inferência...',erro=False)

    def iniciar_camera(self, cfg: Configuracao):
        self._fila.put(('camera',cfg.validar()))
        self._escrever(ocupado=True,status='Abrindo câmera e preparando o modelo...',erro=False)

    def parar_camera(self):
        self._fila.put(('parar',))
        self._escrever(status='Solicitação de parada enviada. Finalizando o quadro atual...')

    def atualizar_config(self, cfg: Configuracao):
        self._fila.put(('config',cfg.validar()))

    def fechar(self):
        self._fila.put(('sair',))
        self._thread.join(timeout=2.0)

    def _loop(self):
        webcam=None
        cfg=Configuracao()
        ultima_imagem=None
        origem='Imagem'
        sair=False

        def parar():
            nonlocal webcam
            if webcam is not None:
                webcam.release()
                webcam=None
            self._escrever(camera_ativa=False)

        def processar(imagem, nome):
            nonlocal ultima_imagem, origem
            ultima_imagem=imagem
            origem=nome
            self._escrever(ocupado=True,status=f'Executando {cfg.modelo} em {nome.lower()}...',erro=False)
            try:
                pred=self._detector.prever(imagem,cfg,nome)
                texto=f'{len(pred.objetos)} objeto(s) previsto(s) • {pred.tempo_total_ms:.0f} ms por inferência'
                with self._lock:
                    self._estado.predicao=pred
                    self._estado.revisao+=1
                    self._estado.status=texto
                    self._estado.erro=False
            except Exception as exc:
                self._escrever(status=f'Falha na inferência: {type(exc).__name__}: {str(exc)[:160]}',erro=True)
            finally:
                self._escrever(ocupado=False)

        while not sair:
            try:
                if webcam is None:
                    comando=self._fila.get(timeout=0.15)
                else:
                    try: comando=self._fila.get_nowait()
                    except queue.Empty: comando=None
            except queue.Empty:
                continue
            if comando is not None:
                tipo=comando[0]
                if tipo=='sair':
                    sair=True
                    parar()
                    continue
                if tipo=='parar':
                    parar()
                    self._escrever(status='Webcam pausada. Último quadro preservado.',ocupado=False)
                elif tipo=='config':
                    cfg=comando[1]
                elif tipo=='imagem':
                    parar()
                    cfg=comando[2]
                    processar(comando[1],comando[3])
                elif tipo=='camera':
                    parar()
                    cfg=comando[1]
                    api=cv2.CAP_DSHOW if sys.platform=='win32' else cv2.CAP_ANY
                    cam=cv2.VideoCapture(cfg.camera,api)
                    if not cam.isOpened():
                        cam.release()
                        self._escrever(status=f'Não foi possível acessar a câmera {cfg.camera}.',ocupado=False,erro=True)
                    else:
                        webcam=cam
                        webcam.set(cv2.CAP_PROP_FRAME_WIDTH,960)
                        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,540)
                        # Buffer pequeno para evitar segundos de atraso em câmeras lentas.
                        webcam.set(cv2.CAP_PROP_BUFFERSIZE,1)
                        self._escrever(camera_ativa=True,erro=False)
            if webcam is not None and not sair:
                ok,quadro=webcam.read()
                if not ok or quadro is None:
                    parar()
                    self._escrever(ocupado=False,erro=True,status='A câmera parou de fornecer quadros.')
                else:
                    processar(quadro,f'Webcam {cfg.camera}')
                    if self.snapshot().erro:
                        # Se os pesos não carregarem, não repetir erro quadro a quadro.
                        parar()
                        self._escrever(status='Erro ao processar webcam; confira a mensagem anterior e tente novamente.')
        parar()
