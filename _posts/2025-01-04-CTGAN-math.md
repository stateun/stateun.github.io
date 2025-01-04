---
layout: single
title: "[Review] CTGAN Code"
date: 2025-01-04
author_profile: true
use_math: true
categories:
  - Code
tags:
  - Generative Models
permalink: /CTGAN-1/
---

오늘은 **CTGAN Python 코드** 중 **`ctgan.py`** 모듈의 Condition을 부여하는 방법을 리뷰해보겠습니다.  

CTGAN의 **공식적인 코드**는 아래 주소에서 다운받을 수 있습니다.  

[![View on GitHub](https://img.shields.io/badge/GitHub-sdv--dev%2FCTGAN-blue?logo=github&style=flat)](https://github.com/sdv-dev/CTGAN)


---

## 1. 주요 아이디어: Discrete Column에 Condition을 어떻게 주는가?

CTGAN 코드에서 가장 핵심적인 부분은 **이산형(Discrete) Column에 Condition을 어떻게 부여하는지**입니다.

---

## 2. 예제 데이터셋: `adult.csv`

- **Feature 개수**: 총 15개  
  - 이산형 변수(Discrete): 9개  
  - 연속형 변수(Continuous): 6개
- **데이터 개수**: 32,561개

~~~python
self._transformer = DataTransformer()
self._transformer.fit(train_data, discrete_columns)
train_data = self._transformer.transform(train_data)
~~~

위 코드는 `train_data`를 다음과 같이 **전처리(Preprocessing)** 합니다.

1. **이산형 변수**: One-hot encoding  
2. **연속형 변수**: Variational Gaussian Mixture Model (VGM)  
   - 코드에서 `max_cluster` 수를 10으로 설정  

전처리 결과, $(32561, 15) \quad \longrightarrow \quad (32561, 156)$ 으로 차원이 증가하는 것을 확인할 수 있습니다.  

(One-hot encoding으로 인한 이산형 변수 차원 증가 및 VGM을 통해 얻은 one-hot 벡터, 연속형 변수 scaling 등이 합쳐진 결과)

논문에서는 이러한 전처리 후의 데이터를 아래와 같은 표기법으로 다룹니다.

$\mathbf{r}_{j} = \alpha_{1,j} \oplus \beta_{1,j} \oplus \alpha_{N_c, j} \oplus \cdots \oplus d_{1,j} \oplus d_{N_d, j}$

- $ \alpha $: Scaled continuous value  
- $ \beta $: Indicate the mode (VGM에서 어떤 mixture component에 속했는지)  
- $ N_c $: 연속형 변수 개수  
- $ N_d $: 이산형 변수 개수  

추가적으로, Normalized 부분은 코드에서 `ClusterBasedNormalier()`를 사용한 것을 확인할 수 있습니다.

---

## 3. Condition을 부여하는 핵심 파트

~~~python
self._data_sampler = DataSampler(
    train_data, 
    self._transformer.output_info_list, 
    self._log_frequency
)
~~~

위에서부터 본격적으로 **Condition을 어떻게 주는지** 확인해봅시다.  

조건(Conditional) 벡터를 추출하는 아이디어가 상당히 기발합니다.

~~~python
cond = np.zeros((batch, self._n_categories), dtype='float32')

discrete_column_id = np.random.choice(np.arange(self._n_discrete_columns), batch)
category_id_in_col = self._random_choice_prob_index(discrete_column_id)
category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col

cond[np.arange(batch), category_id] = 1
~~~

위 코드들을 하나씩 살펴보겠습니다.

### 3.1 `discrete_column_id` 선택

~~~python
discrete_column_id = np.random.choice(np.arange(n_discrete_columns), batch)
discrete_column_id
~~~

이는, 우리가 가진 $N_{d}$개의 이산형 변수들 중에서 무작위로 `batch`만큼 **이산형 변수를 추출**하겠다는 의미입니다.

### 3.2 `category_id_in_col` 선택

~~~python
category_id_in_col = self._random_choice_prob_index(discrete_column_id)
~~~

이 부분이 **정확히 어떤 category로 Condition을 줄지를 결정**합니다.  
  
`_random_choice_prob_index()` 함수의 내부를 확인해봅시다.

~~~python
def _random_choice_prob_index(self, discrete_column_id):
    probs = self._discrete_column_category_prob[discrete_column_id]
    r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
    return (probs.cumsum(axis=1) > r).argmax(axis=1)
~~~

#### 3.2.1 `self._discrete_column_category_prob`

여기서 `self._discrete_column_category_prob`를 먼저 살펴봐야 합니다.

~~~python
max_category = max(
    [column_info[0].dim for column_info in output_info if is_discrete_column(column_info)],
    default=0,
)

self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
~~~

- **`max_category`**: 모든 이산형 변수 중에서 **가장 많은 category**를 갖는 변수의 category 수  
- `self._discrete_column_category_prob`는 (이산형 변수 수, max_category) 형태의 **확률**을 저장하는 배열입니다.

> **Note**  
> "만약 어떤 이산형 변수가 8개의 category를 갖고 있는데, `max_category`가 10개이면?"  
> → 해당 변수에서 존재하지 않는 category에 대한 확률은 **0**으로 처리됩니다.

~~~python
self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))

st = 0
current_id = 0
current_cond_st = 0
for column_info in output_info:
    if is_discrete_column(column_info):
        span_info = column_info[0] 
        ed = st + span_info.dim
        
        category_freq = np.sum(data[:, st:ed], axis=0)

        if log_frequency:
            category_freq = np.log(category_freq + 1)
        
        category_prob = category_freq / np.sum(category_freq)
        self._discrete_column_category_prob[current_id, : span_info.dim] = category_prob
        
        self._discrete_column_cond_st[current_id] = current_cond_st
        self._discrete_column_n_category[current_id] = span_info.dim
        
        current_cond_st += span_info.dim
        current_id += 1
        st = ed
    else:
        st += sum([span_info.dim for span_info in column_info])
~~~

위 과정을 통해 **각 이산형 변수별 category 등장 확률**을 계산하게 됩니다.  

이를 통해 **클래스 불균형(Class Imbalance)** 문제를 완화하기 위해 `log_frequency`(기본값 True)를 사용하여 확률을 조정하는 기법도 확인할 수 있습니다.

#### 3.2.2 `_random_choice_prob_index` 함수 동작

~~~python
def _random_choice_prob_index(self, discrete_column_id):
    probs = self._discrete_column_category_prob[discrete_column_id]
    
    r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
    
    return (probs.cumsum(axis=1) > r).argmax(axis=1)
~~~

즉, 아래와 같은 방식으로 category를 샘플링합니다.

1. $r^{*} ~ U(0,1) 에서 뽑은 r^{*}$ 값을 기준으로,  
2. 각 category 확률의 누적합이 $r^{*}$ 를 초과하는 첫 번째 index를 찾습니다.

이를 통해 **무작위로 category를 선택**하되, 각 category별 **빈도수 기반 확률**로 샘플링하는 효과를 얻을 수 있습니다.

### 3.3 최종 Condition 벡터 생성

~~~python
category_id_in_col = self._random_choice_prob_index(discrete_column_id)
category_id = self._discrete_column_cond_st[discrete_column_id] + category_id_in_col

cond[np.arange(batch), category_id] = 1
~~~

- **`self._discrete_column_cond_st`**: 각 이산형 변수(One-hot encoding)에서 **시작 위치**를 의미  
  - 예: 첫 번째 이산형 변수의 one-hot 위치가 0~8, 두 번째 이산형 변수가 9~24, ... 와 같은 식  
- 최종적으로, `cond` 행렬 내에서 해당 category의 위치(`category_id`)에 **1**을 대입합니다.  
  - 예: 100번째 category를 선택했다면, `cond[*, 100] = 1`이 되고 나머지는 0이 됩니다.

결과적으로, 이것이 우리가 흔히 말하는 CTGAN의 **Conditional Vector**가 됩니다.

---