# 두려움 감정 인식 프로젝트 😨📷

## 📦 데이터셋
- Kaggle 기반 감정 이미지 데이터셋
    - [MMA Facial Expression](https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression)
    - [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

## 🎯 목표
- 얼굴 이미지를 입력으로 받아 **두려움(Fear) 감정을 인식**하는 DNN 기반의 이진 분류 모델 구축
- **PyTorch**를 활용하여 DNN 모델 설계 및 학습
- **Streamlit** 기반의 웹 애플리케이션으로 구현하여 실시간 예측 시스템 완성

## ✅ 진행 과정
1. 데이터 수집 및 전처리:
   - Fear와 Others 두 클래스 구성
   - 이미지 크기 조정 (48x48), Grayscale 변환, 벡터화 및 정규화
   - 데이터 증강: 회전, 반전, 노이즈 추가 등
2. 모델 학습:
   - DNN 모델 설계 및 학습
   - 손실 함수: `BCEWithLogitsLoss`
   - 최적화 기법: `Adam`
   - Early Stopping 및 Dropout 적용
3. 성능 평가:
   - Accuracy: 0.96
   - Precision: 0.94
   - Recall: 0.97
   - F1-score: 0.95
4. 웹 구현:
   - Streamlit을 통해 간단한 이미지 업로드 및 예측 결과 시각화 구현
   - 예측 결과와 함께 Confidence Score 출력

## 📌 결론
- DNN 기반 모델을 통해 **두려움(Fear) 감정을 96% 정확도로 예측**할 수 있었음
- Streamlit 기반 웹 애플리케이션으로 실시간 예측 시스템을 구현하여 사용자 친화적인 인터페이스를 제공

## 🚀 시사점
- 감정 인식 기술이 아동 돌봄, 상담, 교육, 위기 대응 등 다양한 실생활에서 활용될 가능성 확인
- 향후 CNN 기반 모델 도입 시, 공간적 패턴을 더욱 효과적으로 반영 가능

## 🛠️ 개선점
- 감정 간 시각적 유사성 해결을 위해 **다중 감정 분류로 확장**할 필요 있음
- 현재는 DNN만을 사용했으나, **CNN 또는 Transfer Learning 모델 도입 검토**
- 데이터셋 불균형 문제 개선 및 다양한 증강 기법 추가 필요
