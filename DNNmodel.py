
## 모듈 로딩 
import torch                                            ## 텐서 및 기본 함수들 모듈
import torch.nn as nn                                   ## 인공신경망 관련 모듈
import torch.nn.functional as F                         ## 인공신경망 관련 함수들 모듈
import torch.optim as optim                             ## 인공신경망 관련 최적화 모듈
from torch.optim.lr_scheduler import ReduceLROnPlateau  ## 학습률 조정 

from torchinfo import summary                           ## 모델 정보 및 구조 확인 모듈
from torchmetrics.classification import *               ## 모델 성능 지표 관련 모듈

from torchvision.datasets import ImageFolder            ## 이미지용 데이터셋 생성 모듈
from torch.utils.data import DataLoader                 ## 데이터 셋 관련 모듈
from torch.utils.data import Subset, random_split       
from torchvision.transforms import transforms           ## 이미지 전처리 및 증강 모듈

import matplotlib.pyplot as plt                         ## 이미지 시각화 

from utils2 import * 
import os


class FEARDNN(nn.Module):
    def __init__(self, isDebug=False):
        super(FEARDNN, self).__init__()
        
        # 입력층: 이미지 크기가 48x48
        self.in_layer   = nn.Flatten()
        
        # 은닉층
        self.hd_layer1  = nn.Linear(48 * 48, 512) 
        self.drop_layer = nn.Dropout(0.5)
        self.hd_layer2  = nn.Linear(512, 256)
        self.hd_layer3  = nn.Linear(256, 130)
        
        # 출력층: 이진 분류이므로 출력 노드를 1개로 설정
        self.out_layer  = nn.Linear(130, 1)  
        
        # 디버그 모드 설정
        self.isDebug    = isDebug

    ## 순방향 학습 진행 메서드 
    def forward(self, data):
        ## 3D (BS, H, W) ==> 2D (BS, H*W)
        if self.isDebug: print(f'data shape : {data.shape}') # True 일때만 출력 (디버깅 용)
        
        out = self.in_layer(data)
        if self.isDebug: print(f'out shape : {out.shape}')

        out = F.relu(self.hd_layer1(out))
        out = self.drop_layer(out)

        out = F.relu(self.hd_layer2(out))
        out = self.drop_layer(out)

        out = F.relu(self.hd_layer3(out))
        out = self.out_layer(out)  # Sigmoid 제거 (BCEWithLogitsLoss가 Sigmoid 포함)
        
        if self.isDebug: print(f'out shape : {out.shape}')
        
        return out
