---
layout: single
title: "Training Language Models to Follow Instructions with Human Feedback"
date: 2025-03-21
author_profile: true
use_math: true
categories:
  - Large Language Model
tags:
  - Fine tuning
  - Reinforcement Learning
permalink: /Instruct GPT/
---

### Link 🔗 :  [Instruct GPT](https://arxiv.org/abs/2203.02155)

## 1. 논문 개요
기존 대규모 언어 모형(GPT-3 등)은 사용자의 의도를 충분히 반영하지 못하며, 왜곡, 독성 언어 생성, 비효율적 응답 등 여러 문제점을 내포하고 있음이 확인되었다.
이러한 한계를 극복하고자 OpenAI는 인간 피드백(Human Feedback) 을 활용하여 모델을 사용자 의도에 맞게 정렬(alignment)하는 새로운 학습 방법론을 제안한다.

## 2. 목적
핵심 목적:
대규모 언어 모형이 사용자의 명령과 의도에 보다 정확히 작동하도록 개선하는 것이다. 본 연구의 목표는 인간의 평가를 반영한 피드백 기반 학습을 통해 신뢰성, 안전성, 그리고 유용성이 강화된 모델을 개발하는 데 있다.

## 3. 방법론 (SFT → RM → RLHF)
본 논문은 세 단계의 학습 과정을 통해 모델을 점진적으로 정렬한다.

아래는 InsructGPT 학습 과정이다.
![InstructGPT 학습 과정](/images/Instruct_GPT/1.PNG)

### Supervised Fine-Tuning (SFT)

데이터셋:
- OpenAI API를 통해 수집된 실제 고객 프롬프트와 라벨러가 작성한 prompt
- 약 13,000개의 SFT 학습 프롬프트

방법:
- 사전 학습된 GPT-3 모델을 Instruction Learning 방식으로 Fine-tuning하여, 인간 라벨러가 제공한 데모를 통해 모델이 사용자의 지시를 올바르게 따르도록 지도 학습을 수행함

### Reward Modeling (RM)

데이터셋:
- 각 프롬프트에 대해 모델이 생성한 다수의 출력 결과
- 라벨러들이 각 출력의 선호도를 평가하여 비교 데이터를 구축 (약 33,000 프롬프트 기반)

방법:
- 인간 평가자의 선호에 따라 출력에 스칼라 보상을 부여하는 보상 모델을 학습함으로써, 주어진 입력에 대해 바람직한 출력이 무엇인지를 예측하도록 설계함

### Reinforcement Learning from Human Feedback (RLHF) via PPO

데이터셋:
- SFT 결과와 비교 데이터를 활용한 약 31,000개의 PPO 프롬프트

방법:
- 학습된 보상 모델을 활용하여 Proximal Policy Optimization (PPO) 알고리즘으로 SFT 모델을 추가 미세 조정함
- 기존 PPO 방식의 한계를 보완하기 위해, 사전 학습 분포의 업데이트를 혼합한 PPO-ptx 방식을 도입하여 공공 NLP 데이터셋에서 발생하는 성능 저하(Alignment Tax)를 완화함

전체 학습 흐름:
먼저 SFT로 기본적인 지시 따르기 능력을 확보하고, 이어서 RM을 통해 인간 선호를 반영한 보상 체계를 확립한 후, 최종적으로 RLHF(PPO/PPO-ptx)를 적용하여 인간 피드백에 최적화된 정렬 모델(InstructGPT)을 구축하는 단계별 접근법을 취함.

## 4. 실험 셋팅 및 평가
논문에서는 모델의 성능을 다각도로 평가하기 위해 다음과 같은 실험 환경을 구성하였다.

데이터 수집 및 분할:

![InstructGPT Data Set](/images/Instruct_GPT/2.PNG)

프롬프트 데이터셋:
- OpenAI API를 통해 실제 고객 프롬프트와 라벨러가 작성한 프롬프트를 수집
- 사용자 ID 기반으로 train/validation/test 세트를 분리하여 중복 및 개인정보(PII)를 필터링함

각 단계별 데이터 규모:
- SFT: 약 13,000 프롬프트
- RM: 약 33,000 프롬프트
- PPO (RLHF): 약 31,000 프롬프트

라벨러 및 평가 기준:
- 약 40명의 계약직 라벨러(Upwork, Scale AI)를 통해 데이터 수집 및 평가
- 평가 항목은 명령 따름, 제약 조건 준수, 환각률, 독성 등으로 세분화되어 있으며, 1~7 Likert 척도 및 다양한 메타데이터를 활용함

비교 기준 및 벤치마크:
- 기본 GPT-3 (Few-shot 프롬프트 포함), SFT 모델, 그리고 RLHF(PPO 및 PPO-ptx) 모델을 상호 비교
- 추가로 FLAN, T0와 같은 공개 NLP 데이터셋 기반 모델과의 평가를 수행
- 평가 지표는 인간 선호도, TruthfulQA, RealToxicityPrompts, Winogender, CrowS-Pairs 등 다양한 벤치마크를 활용함

실험 결과:
- 인간 선호: 175B InstructGPT 출력이 GPT-3 대비 85% 이상의 선호도를 보였으며, Held-out 라벨러 평가에서도 일관된 개선 효과를 확인
- 진실성 및 환각 감소: 폐쇄 도메인 작업에서 환각 발생률이 크게 감소하였으며, TruthfulQA 평가에서도 보다 진실된 출력을 제공함
- 안전성 및 독성: “안전한 출력” 프롬프트 조건 하에서 독성 생성이 현저히 낮아졌으나, 편향 문제는 개선 효과가 제한적임

## 5. 결론
본 연구는 대규모 언어 모형이 단순히 파라미터 수의 확장만으로는 해결되지 않는 사용자 의도 정렬 문제를, 인간 피드백을 통한 학습 방식(SFT → RM → RLHF)을 통해 효과적으로 개선할 수 있음을 실험적으로 증명하였다. 특히, RLHF 기반의 InstructGPT는 기존 GPT-3 대비 인간 선호도, 진실성, 그리고 안전성 측면에서 현저한 성능 향상을 보였으며, 모델 크기와 상관없이 비용 효율적인 정렬 기법의 가능성을 제시한다.

그러나, 연구 결과는 일부 편향 문제와 복잡한 명령 내 다중 제약 조건 처리 등의 한계를 여전히 내포하고 있어, 향후 보다 정교한 라벨링 및 데이터 수집 방법과 다양한 사용자 집단의 의견을 반영한 정렬 기법 개발이 필요함을 시사한다. 이러한 점은 향후 초대규모 AI 시스템의 안전하고 신뢰할 수 있는 정렬을 위한 중요한 연구 방향으로 제시될 수 있다.