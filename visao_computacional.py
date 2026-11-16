"""Aula 7: detecção, segmentação e classificação na mesma interface.

Janela em Pygame Community Edition, inspirada na 'Classificação Interativa'
fornecida como referência pelo docente. Inferência assíncrona e local com YOLO11.
"""
from __future__ import annotations

import ctypes
import dataclasses
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import pygame
except ImportError as exc:
    print('Pygame Community Edition ausente. Instale com: python -m pip install -r requirements.txt')
    raise SystemExit(1) from exc

from motor_visao import Configuracao, MODELOS, TAMANHOS, abrir_imagem
from renderizacao import ETAPAS, DESCRICOES, renderizar, selecionar_na_imagem
from trabalhador import Trabalhador

W,H,FPS=1600,980,45
BG=(10,12,17); PANEL=(19,23,31); DARK=(13,16,22); BORDER=(54,63,76)
TEXT=(235,240,246); MUTED=(155,168,183); DIM=(102,118,133)
CYAN=(70,211,220); GREEN=(88,208,145); RED=(244,116,116); GOLD=(246,202,91)
BUTTON=(42,50,62); HOVER=(60,72,86)
BASE=Path(__file__).resolve().parent


def dpi_awareness():
    if sys.platform=='win32':
        try:ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:ctypes.windll.user32.SetProcessDPIAware()
            except Exception:pass


def limitar_texto(fonte,valor,largura):
    valor=str(valor)
    if fonte.size(valor)[0] <= largura:return valor
    while valor and fonte.size(valor+'…')[0]>largura: valor=valor[:-1]
    return valor+'…'


class App:
    def __init__(self,size=None,worker=None):
        dpi_awareness()
        pygame.init()
        pygame.display.set_caption('Visão Computacional Interativa | Aula 7 — Introdução à IA')
        info=pygame.display.Info()
        if size is None:
            dw=info.current_w if info.current_w >=800 else W
            dh=info.current_h if info.current_h >=600 else H
            s=min(1, dw*.96/W, dh*.92/H)
            size=(max(800,round(W*s)),max(490,round(H*s)))
        self.window=pygame.display.set_mode(size,pygame.RESIZABLE)
        self.canvas=pygame.Surface((W,H))
        self.clock=pygame.time.Clock()
        self.fonts={
            'title':pygame.font.SysFont('segoeui,arial',29,bold=True),
            'head':pygame.font.SysFont('segoeui,arial',20,bold=True),
            'bold':pygame.font.SysFont('segoeui,arial',17,bold=True),
            'body':pygame.font.SysFont('segoeui,arial',17),
            'small':pygame.font.SysFont('segoeui,arial',14),
            'tiny':pygame.font.SysFont('segoeui,arial',12),
            'mono':pygame.font.SysFont('consolas,monospace',15),
        }
        self.cfg=Configuracao()
        self.worker=worker if worker is not None else Trabalhador()
        self.pred=None
        self.image=None
        self.fonte='Nenhuma imagem'
        self.last_revision=-1
        self.etapa=4
        self.selecionado=None
        self.lista_pagina=0
        self.modal=None
        self.status='Selecione uma imagem ou inicie a webcam para começar.'
        self.erro=False
        self.buttons={}
        self.viewport=(0,0,1.)
        self.image_rect_right=None
        self.cache={}
        self.output_cache_key=None
        self.output_cache=None
        self.running=True

    def text(self,txt,x,y,style='body',color=TEXT,max_width=None):
        font=self.fonts[style]
        txt=str(txt)
        if max_width is not None:txt=limitar_texto(font,txt,max_width)
        self.canvas.blit(font.render(txt,True,color),(int(x),int(y)))

    def center(self,txt,rect,style='small',color=TEXT):
        font=self.fonts[style]
        box=pygame.Rect(rect)
        txt=limitar_texto(font,str(txt),box.width-8)
        imagem=font.render(txt,True,color)
        self.canvas.blit(imagem,imagem.get_rect(center=box.center))

    def panel(self,rect,bg=PANEL,border=BORDER,radius=11):
        box=pygame.Rect(rect)
        pygame.draw.rect(self.canvas,bg,box,border_radius=radius)
        pygame.draw.rect(self.canvas,border,box,1,border_radius=radius)

    def section(self,num,title,x,y):
        self.text(num,x,y,'head',CYAN)
        self.text(title,x+41,y,'head')

    def pos_logico(self,mouse):
        ox,oy,s=self.viewport
        return ((mouse[0]-ox)/s,(mouse[1]-oy)/s)

    def button(self,key,label,rect,callback,kind='normal',enabled=True,style='small'):
        box=pygame.Rect(rect)
        hover=enabled and box.collidepoint(self.pos_logico(pygame.mouse.get_pos()))
        if not enabled:bg,fg=(31,36,44),(96,108,121)
        elif kind=='primary':bg,fg=((51,172,183) if hover else (34,140,153)),(255,255,255)
        elif kind=='active':bg,fg=(42,99,123),TEXT
        elif kind=='danger':bg,fg=((101,56,61) if hover else (71,44,49)),(250,208,208)
        else:bg,fg=(HOVER if hover else BUTTON),TEXT
        pygame.draw.rect(self.canvas,bg,box,border_radius=6)
        pygame.draw.rect(self.canvas,BORDER,box,1,border_radius=6)
        self.center(label,box,style,fg)
        self.buttons[key]=(box,callback,enabled)

    def carregar_imagem(self,path,rotulo):
        try:
            if self.worker.snapshot().camera_ativa:
                self.worker.parar_camera()
            self.image=abrir_imagem(path)
            self.pred=None
            self.cache.clear()
            self.fonte=rotulo
            self.selecionado=None
            self.lista_pagina=0
            self.status=f'{rotulo} carregado; pressione Analisar para executar o modelo.'
            self.erro=False
        except (OSError,ValueError) as exc:
            self.status=f'Falha ao abrir imagem: {exc}'
            self.erro=True

    def dialogo_arquivo(self,save=False):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root=tk.Tk();root.withdraw();root.attributes('-topmost',True)
            try:
                if save:
                    return filedialog.asksaveasfilename(title='Salvar resultado de visão computacional',
                        defaultextension='.png',initialfile='visao_computacional_resultado.png',
                        filetypes=[('Imagem PNG','*.png')])
                return filedialog.askopenfilename(title='Abrir imagem para análise',
                    filetypes=[('Imagens','*.jpg *.jpeg *.png *.bmp *.webp'),('Todos os arquivos','*.*')])
            finally:root.destroy()
        except Exception as exc:
            self.status=f'Selecionador de arquivos indisponível: {exc}'
            self.erro=True
            return ''

    def abrir_arquivo(self):
        path=self.dialogo_arquivo()
        if not path:return
        self.worker.parar_camera()
        self.carregar_imagem(path,Path(path).name)

    def analisar(self):
        if self.image is None:
            self.status='Escolha uma imagem ou use a webcam primeiro.'
            return
        if self.worker.snapshot().camera_ativa:
            self.status='Pause a webcam para reanalisar uma imagem estática.'
            return
        self.worker.analisar(self.image,self.cfg,self.fonte)
        self.status='Inferência solicitada. Carregando pesos (se for a primeira execução)...'
        self.erro=False

    def webcam(self):
        estado=self.worker.snapshot()
        if estado.camera_ativa:
            self.worker.parar_camera()
            self.status='Solicitando pausa. O último quadro permanecerá disponível.'
        else:
            self.pred=None
            self.worker.iniciar_camera(self.cfg)
            self.status=f'Abrindo webcam {self.cfg.camera}...'

    def captura(self):
        estado=self.worker.snapshot()
        if estado.camera_ativa:
            self.worker.parar_camera()
            self.status='Congelando o último quadro inferido; agora é possível inspecionar objetos.'
        elif self.pred:
            self.status='O último quadro já está congelado.'
        else:self.status='Não há quadro para capturar.'

    def salvar_png(self):
        if self.pred is None:
            self.status='Analise uma imagem antes de salvar o resultado.'
            return
        path=self.dialogo_arquivo(True)
        if not path:return
        try:
            imagem=renderizar(self.pred,4,self.cfg.opacidade,self.selecionado)
            ok,encoded=cv2.imencode('.png',imagem)
            if not ok:raise ValueError('Falha ao codificar PNG.')
            encoded.tofile(path)
            self.status=f'Imagem de resultado exportada: {Path(path).name}'
        except (OSError,ValueError) as exc:
            self.status=f'Não foi possível exportar: {exc}'
            self.erro=True

    def alterar(self,op,direcao):
        cfg=self.cfg
        if op=='modelo':
            idx=MODELOS.index(cfg.modelo)
            cfg=dataclasses.replace(cfg,modelo=MODELOS[(idx+direcao)%len(MODELOS)])
        elif op=='conf':cfg=dataclasses.replace(cfg,confianca=round(min(.90,max(.10,cfg.confianca+direcao*.05)),2))
        elif op=='iou':cfg=dataclasses.replace(cfg,iou=round(min(.85,max(.15,cfg.iou+direcao*.05)),2))
        elif op=='tam':
            idx=TAMANHOS.index(cfg.tamanho)
            cfg=dataclasses.replace(cfg,tamanho=TAMANHOS[min(len(TAMANHOS)-1,max(0,idx+direcao))])
        elif op=='opacidade':cfg=dataclasses.replace(cfg,opacidade=round(min(.90,max(.10,cfg.opacidade+direcao*.1)),2))
        elif op=='max':
            valores=(10,20,30,50,100)
            idx=valores.index(cfg.max_objetos) if cfg.max_objetos in valores else 2
            cfg=dataclasses.replace(cfg,max_objetos=valores[min(len(valores)-1,max(0,idx+direcao))])
        elif op=='cam':cfg=dataclasses.replace(cfg,camera=min(5,max(0,cfg.camera+direcao)))
        cfg.validar()
        self.cfg=cfg
        self.worker.atualizar_config(cfg)
        if op=='opacidade':
            self.cache.clear()
            self.status='Opacidade atualizada; apenas a visualização foi modificada.'
        elif self.worker.snapshot().camera_ativa:
            self.status='Parâmetro enviado. A webcam usará a configuração nos próximos quadros.'
        else:
            self.status='Parâmetro alterado. Clique em Analisar para executar outra inferência.'

    def mudar_etapa(self,etapa):
        self.etapa=etapa
        self.status=f'Etapa didática {etapa+1}/5: {ETAPAS[etapa]}. As previsões vêm da mesma rede.'

    def selecionar(self,n):
        self.selecionado=n
        self.cache.clear()
        if self.pred and n is not None and n<len(self.pred.objetos):
            o=self.pred.objetos[n]
            self.status=f'Objeto #{n+1} selecionado: {o.classe_pt}, confiança {o.confianca*100:.1f}%.'
        else:self.status='Seleção removida.'

    def atualizar_estado(self):
        snap=self.worker.snapshot()
        if snap.revisao!=self.last_revision and snap.predicao is not None:
            self.last_revision=snap.revisao
            self.pred=snap.predicao
            self.image=self.pred.imagem_bgr
            self.fonte=self.pred.origem
            self.cache.clear()
            self.selecionado=None
            self.lista_pagina=0
            self.status=snap.status
            self.erro=False
        if snap.erro and not self.erro:
            self.status=snap.status
            self.erro=True

    def header(self):
        self.text('Visão Computacional Interativa',26,15,'title')
        self.text('Detecção, segmentação e classificação de objetos com Inteligência Artificial',28,52,'body',MUTED)
        self.text('AULA 7  •  YOLO11  •  SEGMENTAÇÃO DE INSTÂNCIAS',1120,25,'small',CYAN)
        self.text('MODELO',29,88,'small',MUTED)
        self.text('Imagem → pré-processamento → rede neural → caixas + máscaras + classes',106,85,'bold',TEXT)
        estado=self.worker.snapshot()
        if estado.camera_ativa:
            self.text('● WEBCAM ATIVA',795,87,'small',GREEN)
        elif estado.ocupado:
            self.text('● PROCESSANDO',795,87,'small',GOLD)
        elif self.pred is not None:
            self.text('● RESULTADO PRONTO',795,87,'small',CYAN)
        else:self.text('● AGUARDANDO ANÁLISE',795,87,'small',MUTED)
        y=114
        # Controles principais distribuídos em uma linha, sem imagens demonstrativas pré-carregadas.
        larguras = (240, 240, 240, 240, 240, 240)
        posicoes = (27, 289, 551, 813, 1075, 1337)
        self.button('abrir','Abrir imagem',(posicoes[0],y,larguras[0],36),self.abrir_arquivo)
        self.button('analisar','Analisar',(posicoes[1],y,larguras[1],36),self.analisar,'primary',self.image is not None and not estado.camera_ativa and not estado.ocupado)
        self.button('camera','Parar webcam' if estado.camera_ativa else 'Iniciar webcam',(posicoes[2],y,larguras[2],36),self.webcam,'active' if estado.camera_ativa else 'normal',not estado.ocupado or estado.camera_ativa)
        self.button('captura','Capturar quadro',(posicoes[3],y,larguras[3],36),self.captura,enabled=estado.camera_ativa or self.pred is not None)
        self.button('salvar','Exportar PNG',(posicoes[4],y,larguras[4],36),self.salvar_png,enabled=self.pred is not None)
        self.button('help','Como usar',(posicoes[5],y,larguras[5],36),self.mostrar_ajuda)

    def imagem_em_rect(self,img,quadro,cache_id):
        quadro=pygame.Rect(quadro)
        self.panel(quadro,DARK,(65,75,90),6)
        interior=quadro.inflate(-8,-8)
        if img is None:
            self.center('Nenhuma imagem',interior,'body',MUTED)
            return None
        ih,iw=img.shape[:2]
        s=min(interior.w/iw,interior.h/ih)
        cw,ch=max(1,int(iw*s)),max(1,int(ih*s))
        bounds=pygame.Rect(0,0,cw,ch);bounds.center=interior.center
        k=(cache_id,iw,ih,cw,ch)
        surf=self.cache.get(k)
        if surf is None:
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            frame=pygame.image.frombuffer(rgb.tobytes(),(iw,ih),'RGB').copy()
            surf=pygame.transform.smoothscale(frame,(cw,ch))
            self.cache[k]=surf
            # O cache se limita a poucos quadros, mesmo em webcam.
            if len(self.cache)>12:
                primeiro=next(iter(self.cache))
                if primeiro!=k:self.cache.pop(primeiro)
        self.canvas.blit(surf,bounds.topleft)
        return bounds

    def painel_visual(self):
        self.panel((24,164,1030,614))
        self.section('01','VISUALIZAÇÃO DA INFERÊNCIA',44,182)
        if self.pred:
            summary=f'{self.pred.largura} × {self.pred.altura} px   •   {len(self.pred.objetos)} objetos'
        elif self.image is not None:
            summary=f'{self.image.shape[1]} × {self.image.shape[0]} px   •   não analisada'
        else:summary='Aguardando entrada'
        self.text(summary,734,188,'small',MUTED,max_width=293)
        for i,nome in enumerate(ETAPAS):
            self.button('etapa'+str(i),f'{i+1}  {nome}',(43+i*198,222,190,38),lambda i=i:self.mudar_etapa(i),
                        'active' if i==self.etapa else 'normal')
        self.text('IMAGEM ORIGINAL',47,273,'small',MUTED)
        self.text('SAÍDA: '+ETAPAS[self.etapa].upper(),547,273,'small',CYAN)
        imagem=self.image if self.image is not None else None
        self.imagem_em_rect(imagem,(43,299,484,378),('original',self.last_revision,id(imagem)))
        if self.pred is not None:
            render_key=(self.last_revision,self.etapa,self.cfg.opacidade,self.selecionado)
            if render_key!=self.output_cache_key:
                self.output_cache=renderizar(self.pred,self.etapa,self.cfg.opacidade,self.selecionado)
                self.output_cache_key=render_key
            out=self.output_cache
        else:
            out=imagem
        self.image_rect_right=self.imagem_em_rect(out,(544,299,487,378),('output',self.last_revision,self.etapa,self.cfg.opacidade,self.selecionado,id(imagem)))
        self.text('Fonte: '+self.fonte,46,692,'small',MUTED,max_width=476)
        if self.pred:
            ms=self.pred.tempo_total_ms
            self.text(f'Tempo de inferência: {ms:.0f} ms  (~{1000/max(1,ms):.1f} inferências/s)',548,692,'small',TEXT,max_width=470)
        else:
            self.text('Clique em Analisar após abrir uma imagem.',548,692,'small',MUTED)
        pygame.draw.line(self.canvas,BORDER,(44,725),(1031,725))
        self.text('DICA',47,741,'small',CYAN)
        self.text('Clique em uma caixa ou máscara da saída para inspecionar o objeto.',100,739,'body',TEXT,max_width=899)
        self.text('Teclas 1–5: alternar as etapas  |  Espaço: analisar  |  S: exportar resultado',47,761,'tiny',MUTED)

    def linha_opcao(self,y,titulo,valor,chave):
        self.text(titulo,1095,y+9,'small',MUTED,max_width=214)
        self.button(chave+'-','−',(1345,y,35,34),lambda chave=chave:self.alterar(chave,-1),'normal',style='bold')
        self.panel((1385,y,124,34),DARK,BORDER,5)
        self.center(valor,(1387,y,120,34),'small',CYAN)
        self.button(chave+'+','+',(1515,y,35,34),lambda chave=chave:self.alterar(chave,1),'normal',style='bold')

    def painel_config(self):
        self.panel((1070,164,506,367))
        self.section('02','CONFIGURAÇÕES DA REDE',1090,182)
        self.text('Alterações: reaplique com Analisar (webcam atualiza automaticamente).',1093,213,'tiny',MUTED,max_width=460)
        self.linha_opcao(239,'Modelo pré-treinado','YOLO11n-seg' if self.cfg.modelo==MODELOS[0] else 'YOLO11s-seg','modelo')
        self.linha_opcao(280,'Confiança mínima',f'{self.cfg.confianca:.2f}','conf')
        self.linha_opcao(321,'IoU (supressão / NMS)',f'{self.cfg.iou:.2f}','iou')
        self.linha_opcao(362,'Resolução de entrada',f'{self.cfg.tamanho} px','tam')
        self.linha_opcao(403,'Opacidade da máscara',f'{self.cfg.opacidade*100:.0f}%','opacidade')
        self.linha_opcao(444,'Máximo de objetos',str(self.cfg.max_objetos),'max')
        self.linha_opcao(485,'Câmera (índice)',str(self.cfg.camera),'cam')

    def painel_objetos(self):
        self.panel((1070,545,506,386))
        self.section('03','OBJETOS IDENTIFICADOS',1090,563)
        n=len(self.pred.objetos) if self.pred else 0
        self.text(f'{n} instância(s)',1465,572,'small',CYAN)
        if self.pred is None:
            self.text('A lista aparecerá após executar o modelo.',1094,619,'body',MUTED)
            self.text('Abra uma imagem ou inicie a webcam.',1094,650,'small',MUTED)
            return
        if not n:
            self.text('Nenhum objeto ultrapassou o limiar de confiança.',1092,614,'body',MUTED,max_width=470)
            self.text('Tente uma foto diferente ou reduza a confiança mínima.',1092,647,'small',MUTED,max_width=470)
            return
        por_pagina=5
        max_pagina=(n-1)//por_pagina
        self.lista_pagina=min(self.lista_pagina,max_pagina)
        if max_pagina:
            self.button('pagina-', '◀',(1093,598,42,29),lambda:self.pagina(-1),enabled=self.lista_pagina>0)
            self.text(f'Página {self.lista_pagina+1}/{max_pagina+1}',1208,605,'small',MUTED)
            self.button('pagina+','▶',(1505,598,42,29),lambda:self.pagina(1),enabled=self.lista_pagina<max_pagina)
        else:
            self.text('ID       CLASSE PREVISTA                 CONFIANÇA',1094,607,'tiny',MUTED)
        for p in range(self.lista_pagina*por_pagina,min(n,(self.lista_pagina+1)*por_pagina)):
            o=self.pred.objetos[p]
            yy=636+(p-self.lista_pagina*por_pagina)*34
            ativo=p==self.selecionado
            self.button('obj'+str(p),f'{p+1:02d}    {o.classe_pt:<19} {o.confianca*100:5.1f}%',
                        (1093,yy,456,29),lambda p=p:self.selecionar(p),'active' if ativo else 'normal')
        pygame.draw.line(self.canvas,BORDER,(1093,816),(1550,816),1)
        if self.selecionado is not None and self.selecionado < n:
            obj=self.pred.objetos[self.selecionado]
            self.text(f'INSPEÇÃO DO OBJETO #{self.selecionado+1}',1095,830,'small',CYAN)
            self.text(f'Classe: {obj.classe_pt}  |  confiança: {obj.confianca:.3f}',1095,852,'small',TEXT,max_width=445)
            self.text(f'Caixa: (x1,y1,x2,y2) = {obj.caixa}',1095,875,'small',TEXT,max_width=445)
            self.text(f'Máscara: {obj.area_px:,} pixels  ({obj.fracao_imagem*100:.1f}% da imagem)',1095,898,'small',TEXT,max_width=445)
        else:
            self.text('INSPEÇÃO DE INSTÂNCIA',1095,835,'small',CYAN)
            self.text('Clique em um item ou em um objeto da imagem.',1095,868,'small',MUTED,max_width=450)
            self.text('Veja classe, confiança, caixa e área da máscara.',1095,892,'small',MUTED,max_width=450)

    def pagina(self,d):
        if not self.pred:return
        n=len(self.pred.objetos)
        self.lista_pagina=max(0,min((n-1)//5,self.lista_pagina+d))

    def painel_etapas(self):
        self.panel((24,792,1030,139))
        titulo,linha1,linha2=DESCRICOES[self.etapa]
        self.text(titulo,43,808,'bold',CYAN)
        self.text(linha1,44,848,'body',TEXT,max_width=982)
        self.text(linha2,44,886,'small',MUTED,max_width=987)

    def rodape(self):
        pygame.draw.line(self.canvas,BORDER,(25,947),(1574,947))
        self.text('STATUS',27,958,'small',RED if self.erro else CYAN)
        self.text(self.status,95,956,'small',TEXT,max_width=1464)

    def mostrar_ajuda(self):
        self.modal=('Roteiro de aula — como usar',[
            '1. Abra uma imagem JPG/PNG do computador ou inicie a webcam.',
            '2. Clique em Analisar; na primeira análise o YOLO pode baixar seus pesos.',
            '3. Compare a imagem original com Detecção, Segmentação, Classificação e Resultado.',
            '4. Altere confiança, IoU ou tamanho e clique novamente em Analisar.',
            '5. Inicie a webcam para inferir quadros ao vivo; capture para congelar um quadro.',
            '6. Clique em uma caixa ou máscara para destacar a instância correspondente.',
            '7. Exporte a visualização final para uma imagem PNG.',
            'IMPORTANTE: as três saídas são produzidas conjuntamente por uma única rede.',
            'Não há treinamento aqui: utilizamos pesos previamente treinados no COCO.',
            'Limites: modelo pode errar, confundir classes e não reconhecer objetos fora das 80 categorias.'
        ])

    def desenhar_modal(self):
        titulo,linhas=self.modal
        shade=pygame.Surface((W,H),pygame.SRCALPHA);shade.fill((0,0,0,190))
        self.canvas.blit(shade,(0,0))
        rect=pygame.Rect(250,207,1100,565)
        self.panel(rect,(25,30,40),(83,104,121),12)
        self.text(titulo,280,236,'head',CYAN)
        for i,linha in enumerate(linhas):
            self.text(linha,280,289+i*38,'body',TEXT,max_width=1030)
        self.button('fechar_ajuda','Fechar  (Esc)',(1080,700,220,43),lambda:setattr(self,'modal',None),'primary')

    def desenhar(self):
        self.canvas.fill(BG)
        self.buttons.clear()
        self.header()
        self.painel_visual()
        self.painel_config()
        self.painel_objetos()
        self.painel_etapas()
        self.rodape()
        if self.modal:
            self.buttons.clear()
            self.desenhar_modal()
        sw,sh=self.window.get_size()
        s=min(sw/W,sh/H)
        nw,nh=int(W*s),int(H*s)
        ox=(sw-nw)//2;oy=(sh-nh)//2
        self.viewport=(ox,oy,s)
        self.window.fill(BG)
        if (nw,nh)==(W,H):self.window.blit(self.canvas,(ox,oy))
        else:self.window.blit(pygame.transform.smoothscale(self.canvas,(nw,nh)),(ox,oy))
        pygame.display.flip()

    def clicar_no_preview(self,pos):
        if self.pred is None or self.image_rect_right is None:return
        rect=self.image_rect_right
        if not rect.collidepoint(pos):return
        x=(pos[0]-rect.left)/rect.w*self.pred.largura
        y=(pos[1]-rect.top)/rect.h*self.pred.altura
        self.selecionar(selecionar_na_imagem(self.pred,x,y))

    def tratar_evento(self,event):
        if event.type==pygame.QUIT:self.running=False
        elif event.type==pygame.VIDEORESIZE:
            self.window=pygame.display.set_mode((max(800,event.w),max(490,event.h)),pygame.RESIZABLE)
        elif event.type==pygame.KEYDOWN:
            if event.key==pygame.K_ESCAPE:
                if self.modal:self.modal=None
                else:self.running=False
            elif not self.modal:
                if pygame.K_1<=event.key<=pygame.K_5:self.mudar_etapa(event.key-pygame.K_1)
                elif event.key==pygame.K_SPACE:self.analisar()
                elif event.key==pygame.K_s:self.salvar_png()
                elif event.key==pygame.K_h:self.mostrar_ajuda()
        elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            pos=self.pos_logico(event.pos)
            for rect,callback,enabled in list(self.buttons.values()):
                if enabled and rect.collidepoint(pos):
                    callback();return
            if not self.modal:self.clicar_no_preview(pos)

    def executar(self):
        try:
            while self.running:
                for event in pygame.event.get():self.tratar_evento(event)
                self.atualizar_estado()
                self.desenhar()
                self.clock.tick(FPS)
        finally:
            self.worker.fechar()
            pygame.quit()


def main():
    App().executar()


if __name__=='__main__':main()
