# 목차

1. [서론](#서론)
2. [모델 설명](#모델-설명)
   * [기존 연구](#기존-연구)
   * [과제 모델](#과제-모델)
3. [실험 설계](#실험-설계)
4. [실험 결과 및 분석](#실험-결과-및-분석)
5. [결론](#결론)

# 서론

Sentence classification model을 구현하고 성능을 평가하여 기존 연구와 정량적 성능을 비교해 보기로 한다. 모델의 설계와 구현은 아래의 Yoon Kim의 논문과 그 구현 코드들을 참고하였다.

* [Yoon Kim - Convolutional Neural Networks for Sentence Classification]("http://www.aclweb.org/anthology/D14-1181")
* [Implementing a CNN for Text Classification in Tensorflow]("http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/")
* [yoonkim - CNN_sentence]("https://github.com/yoonkim/CNN_sentence")

* [dennybritz - cnn-text_classification-tf]("https://github.com/dennybritz/cnn-text-classification-tf")

기존 연구는 문장의 단어들을 table lookup을 이용해 임베딩한 후 CNN 모델의 input으로 사용하였으나 본 과제에서는 CNN과 RNN 두 가지 모델로 각각 Character-level word embeding을 수행하여 그것을 모델의 input으로 사용하고, 각각의 성능을 테스트하여 기존 연구의 성능과 비교해 본다.

# 모델 설명

## 기존 연구

### Yoon Kim (2014) 의 아키텍쳐

![](D:\git\nlp-sentence-classify\report\cnn_sc.png)

첫 번째 레이어는 단어를 저차원 벡터로 임베딩한다. 다음 레이어는 여러 사이즈의 필터로, 임베딩 된 단어 벡터에 대해 concatnation 을 수행한다. 다음에는 그것을 맥스 풀링하고, 드롭아웃을 적용하고, 소프트맥스 레이어의 결과로 분류를 수행한다.

## 과제 모델

### Word Embeding

논문에는 Word2Vec 벡터를 사용했으나 본 과제에서는 Character-level embeding을 수행하여 얻은 벡터를 사용한다. Character-level embeding에는 CNN을 이용한 방법과 RNN을 이용한 방법 두 가지를 시도해 본다. 전반적인 성능 향상, 특히 Out-Of-Vocabulary 단어에 대하여 큰 성능 향상을 기대할 수 있다.

#### Character-level Feature - CNN

<img src="D:\git\nlp-sentence-classify\report\char-level-cnn.png" width="600"/>

단어의 모든 문자들을 CNN 레이어의 input으로 넣고 output을 맥스풀링한 결과를 취한다.

#### Character-level Feature - RNN

<img src="D:\git\nlp-sentence-classify\report\char-level-rnn.png" width="600" />

단어의 모든 문자들을 Bi-directional LSTM 레이어의 input으로 넣고 forward output과 backward output을 concatnation 한 결과를 취한다.

### Core Layer

위에서 설명한 기존 연구의 모델과 동일하다.

# 실험 설계



# 실험 결과 및 분석



# 결론

