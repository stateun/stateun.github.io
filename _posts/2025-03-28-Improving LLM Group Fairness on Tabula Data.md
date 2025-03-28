---
layout: single
title: "Improving LLM Group Fairness on Tabular Data via In-Context Learning"
date: 2025-03-28
author_profile: true
use_math: true
categories:
  - Large Language Model
tags:
  - Fairness
  - In-Context Learning
  - Tabular Data
  - Prompt Tuning
permalink: /improving-llm-group-fairness/
---

### Link 🔗 : [Improving LLM Group Fairness on Tabular Data via In-Context Learning](https://arxiv.org/abs/2412.04642)

## 1. 논문 개요
최근 대규모 언어 모형(LLM)은 적은 학습 데이터 상황에서도 탭형 데이터에 대한 예측 성능을 보여주지만, 동일한 입력 조건에서 남녀 등 민감 속성에 따른 예측 불균형(그룹 공정성) 문제가 발생하는 것으로 나타났다. 본 논문은 LLM의 in-context learning 기법을 활용하여, 별도의 대규모 학습 데이터 없이도 예측 시 그룹 간의 긍정 라벨 비율을 균형 있게 맞추기 위한 네 가지 방법론을 제안한다.

## 2. 목적

논문의 주요 목표는 다음과 같다:
- 그룹 공정성 개선 : 동일한 조건 하에서 남성과 여성 등 민감한 집단 간 긍정 예측(Positive Prediction)비율이 균등하도록 한다. 
- 데이터 효율성 : 제한된 데이터 상황에서도 효과적으로 in-context learning 기반의 공정성 향상 기법을 적용할 수 있음을 실험적으로 증명한다.

## 3. 방법론

논문은 총 네 가지 접근법을 제안하며, 테이블 데이터에 대한 예측에서 공정성(Fairness)를 개선한다.

### Fair Prompt Optimization
- 개요 : 기존의 task-specific instruction에 공정성 관련 추가 지시문(fair instruction)을 결합하여, 남녀 간 긍정 라벨 비율의 불균형을 완화한다.
- 세부 방법 :
  - 50개의 예제를 사용하여, Accuracy와 DP간의 trade-off를 고려한 Pareto-optimal 프롬프트를 선택한다.
  - 초기에는 사람이 직접 만든 프롬프트를 Fairness-specific prompt로 사용을 하고 이후, meta-LLM을 통해 100 에폭 반복 학습으로 공정성을 개선하는 최적의 지시문을 생성한다.

### Soft Prompt Tuning
- 개요 : 텍스트 기반의 하드 프롬프트 대신, 임베딩 공간에서의 soft token들을 최적화하여 공정성 패널티를 직접 반영한다.
- 세부 방법 :
  - 주어진 instruction을 embedding vector로 변환한 뒤 이를 input embedding 앞에 붙여 이를 fine-tuning한다.
  - 논문에서는 라벨에 접근할 수 없다는 가정을 사용하므로 Downstream tast를 수행할 LLM과 동일한 모델로 pseudo-label 샘플을 붙여 활용한다.

### Fair Few-Shot Examples
- 개요 : 테스트 인스턴스와 가장 유사한 예시들을 선택해 in-context 학습을 수행, 공정성 불균형을 완화한다.
- 세부 방법 : 
  - Nearest Neighbor Search를 통해 테스트 인스턴스와 같은 Sensitive attribute를 가진 데이터를 선택하고, LLM의 zero-shot 예측을 이용해 pseudo label을 생성한다.
  - 입력 예시에서 긍정 라벨의 비율을 조정하여, 여성과 남성의 예측 비율을 균등하게 맞춘다.

![Results based on the ratio of sensitive attributes](/images/Improving_LLM_Group_Fairness/2.PNG)

Fair Few-Shot Examples 방법론 실험에서는 민감 속성(Sensitive attribute)의 각 속성에 대해 positive label 비율을 다양하게 설정한다. 이때, 예를 들어 여성과 같은 속성의 비율이 높아질수록 성능이 개선되는 경향을 보인다.

### Self-Refinement
- 개요 : 초기 예측 결과를 후처리(post-processing)하여, 그룹 간 긍정 예측 비율 차이가 일정 기준을 초과할 경우 chain-of-thought reasoning을 통해 예측을 보정한다.
- 세부 방법 :
  - 배치 단위로 예측 후, 각 그룹의 positive label 비율을 평가한 다음, 결정 경계 근처에 위치한 샘플들의 라벨을 조정하여 DP를 개선한다.


## Performance
![Results based on the ratio of sensitive attributes](/images/Improving_LLM_Group_Fairness/1.PNG)

위의 결과는 Optimized Fair Prompts, Soft Prompts, Fair Few-shot Examples에 대한 성능 테이블이다.

![Results based on the ratio of sensitive attributes](/images/Improving_LLM_Group_Fairness/3.PNG)

위의 결과는 Self-Refining에 대한 성능 테이블이다.

## 4. 실험 셋팅 및 평가

실험은 4개의 테이블 데이터셋(Adult, German Credit, ACS Income, ACS Coverage)을 대상으로 진행되었으며, 다음과 같은 기준으로 평가되었다.
- 데이터 분할 : 실제 고객 프롬프트 및 라벨러 작성 프롬프트를 기반으로 train/validation/test 세트를 구성하여 중복 및 개인정보를 제거하였다.

- 평가 지표 : 
  - Demographic Parity (DP) : 남성과 여성 간 긍정 예측 비율의 균형 정도를 측정한다.

  - Equalized Odds (EO) : 오분류율이 민감 집단 간에 균형을 이루는지를 평가한다.

- 비교 모델 : CatBoost와 같은 전통적 테이블 데이터 모델과 다양한 LLM 기반 접근법을 비교 평가하였다.

- 실험 결과 : 
  - Fair Prompt Optimization과 Fair Few-Shot Examples가 최적의 정확도와 공정성 trade-off를 달성하는 것으로 나타났다.
  - Soft Prompt Tuning과 Self-Refinement는 특정 상황에서 보조적 효과를 발휘함을 확인할 수 있었다.

## 5. 결론
본 연구는 대규모 언어 모델이 탭형 데이터에 대해 예측할 때 발생하는 그룹 공정성 문제를, in-context learning 기반의 네 가지 방법론을 통해 효과적으로 개선할 수 있음을 보여준다.

- 주요 성과 : 
  - 제한된 데이터 환경에서도 공정성을 달성할 수 있는 다양한 기법을 제시함.
  - Fair Prompt Optimization과 Fair Few-Shot Examples는 정확도와 DP 간 최적의 균형을 이루어, 실무 적용 가능성을 높였다.

- 한계 및 향후 연구 :
  - 일부 방법은 계산 비용 및 하이퍼파라미터 민감도가 존재하며, 보다 다양한 공정성 지표와 실제 응용 환경에서의 검증이 필요하다.
  - 향후 연구에서는 다른 fairness notion(예: equalized odds) 및 대규모 데이터셋을 대상으로 한 확장이 요구된다.