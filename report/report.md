# 목차

1. [서론](#서론)
2. [모델 설명](#모델-설명)
   * [Data and Preprocessing](#data-and-preprocessing)
   * [Embeding Layer](#embeding-layer)
   * [Classification Layer](#classification-layer)
   * [Hyper Parameters](#hyper-parameters)
3. [실험 설계](#실험-설계)
4. [실험 결과 및 분석](#실험-결과-및-분석)
5. [결론](#결론)
6. [레퍼런스](#레퍼런스)

# 서론

Sentence classification model을 구현하고 성능을 평가하여 기존 연구와 정량적 성능을 비교해 보기로 한다. 모델의 설계와 구현은 아래의 Yoon Kim의 논문과 그 구현 코드들을 참고하였다.

* Yoon Kim - Convolutional Neural Networks for Sentence Classification[^1]
* Implementing a CNN for Text Classification in Tensorflow[^2]
* yoonkim - CNN_sentence[^3]

* dennybritz - cnn-text_classification-tf[^4]

기존 연구는 문장의 단어들을 table lookup을 이용해 임베딩한 후 CNN 모델의 input으로 사용하였으나 본 과제에서는 CNN으로 Character-level word embeding을 수행한다. 그리고 기존 연구 모델에 더해서 RNN 기반 모델을 설계하여 기존 연구와 두 개의 모델간의 성능을 비교한다.

# 모델 설명

## Data and Preprocessing

사용한 데이터 셋은 [Pang and Lee's movie review dataset ]("http://www.cs.cornell.edu/people/pabo/movie-review-data/")이다. 해당 데이터는 긍정과 부정 문장이 각각 절반씩 있는 10,662개의 예시 리뷰 문장으로 이루어져 있다. Vocabulary 크기는 약 20k개이다. 이 데이터는 train 데이터와 test 데이터로 분리되어 있지 않으므로 데이터의 10%를 테스트 데이터로 하여 cross-validation 방식으로 학습 및 테스트를 수행한다.

다음은 데이터 전처리 과정이다.

```python
def preprocess():
    print("loading data...")
    text, label = load_data(FLAGS.pos_data_dir, FLAGS.neg_data_dir)

    #vocab
    max_doc_length = max([len(x.split(" ")) for x in text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_doc_length)
    x = np.array(list(vocab_processor.fit_transform(text)))

    np.random.seed(10)
    shuffle_index = np.random.permutation(np.arange(len(label)))
    x_shuffled = x[shuffle_index]
    y_shuffled = label[shuffle_index]

    CV_index = -1 * int(FLAGS.CV_percentage * float(len(label)))
    x_train, x_test = x_shuffled[:CV_index], x_shuffled[CV_index:]
    y_train, y_test = y_shuffled[:CV_index], y_shuffled[CV_index:]

    del x, label, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
    return x_train, y_train, vocab_processor, x_test, y_test
```

1. Positive 문장들과 Negative 문장들을 파일에서 읽어온다.
2. 각 문장들에 최대 문장 길이에 맞춰서 패딩을 넣는다.
3. 모든 단어들을 정수에 매핑하는 Vocabulary를 만든다. 모든 문장들은 단어들의 id, 즉 정수들의 Vector로 표현된다.
4. 데이터를 무작위로 섞고 train 데이터 셋과 test 데이터 셋을 나눈다.

## Embeding Layer

논문에는 Word2Vec 벡터를 사용했으나 본 과제에서는 Character-level embeding을 수행하여 얻은 벡터를 사용한다. Character-level embeding에는 CNN을 이용한다. 전반적인 성능 향상, 특히 Out-Of-Vocabulary 단어에 대하여 큰 성능 향상을 기대할 수 있다.

### Character-level Feature - CNN

<img src="D:\git\nlp-sentence-classify\report\char-level-cnn.png" width="500"/>

단어의 모든 문자들을 CNN 레이어의 input으로 넣고 output을 맥스풀링한 결과를 취한다.

## Classification Layer

### CNN for Sentence Classification (Yoon Kim (2014) 의 아키텍쳐)

![](D:\git\nlp-sentence-classify\report\cnn_sc.png)

첫 번째 레이어는 단어를 저차원 벡터로 임베딩한다. 다음 레이어는 여러 사이즈의 필터로, 임베딩 된 단어 벡터에 대해 concatenation 을 수행한다. 다음에는 그것을 max pooling 하고, dropout을 적용하고, softmax layer의 결과로 분류를 수행한다.

#### Convolution and Max-Pooling Layers

크기가 다른 필터를 반복적으로 사용하여 vector를 생성하고 이를 하나의 큰 featur vector로 합친다.

```

```

`W`는 filter 행렬이고 `h`는 output에 ReLU 함수를 적용한 결과이다. 모든 vector는 `[batch_size, num_filters_total]` 형태를 갖는 feature vector로 결합된다.

#### Dropout Layer

```

```

드롭아웃은 오버피팅을 방지하는 가장 유명한 방법이다. dropout rate를 학습 중에는 0.5, 평가 중에는 1로 비활성화 한다.

#### Scores and Predictions

```

```

max pooling 으로 featur vector를 사용하여 행렬곱을 수행하고 가장 높은 점수로 분류를 선택하는 예측을 수행한다.

#### Loss and Accuracy

```

```

손실함수는 cross-entropy loss를 사용한다.

### RNN for Sentence Classification

<img src="D:\git\nlp-sentence-classify\report\rnn_sc.png" width="500" />

Character-level embeding 결과 벡터를 Bi-directional LSTM의 input으로 활용하여 그  output으로 분류를 수행한다. LSTM의 output은 3차원 Vector([neut, neg, pos] ) 이며, 가장 값이 큰 요소를 결과로서 출력한다.

## Hyper Parameters

* embedding size = 128
* rnn hidden size = 100
* cnn filter windows = 3, 4, 5
* number of  cnn filters = 128
* cross validation ratio = 0.1
* dropout rate = 0.5

# 실험 설계

## 평가 지표

<img src="D:\git\nlp-sentence-classify\report\confusion_matrix.png" width="500" />

1. Accuracy

   * 모델이 샘플을 정확히 분류한 비율.

   * $$
     Accuracy = {TP + TN \over TP + FN + FP + TN}
     $$

2. Precision

   * 모델이 positive로 분류한 샘플 중 실제로 positive인 비율.

   * $$
     Precision = {TP \over TP + FP}
     $$

     

3. Recall

   * positive 샘플 중 모델이 positive로 분류한 비율.

   * $$
     Recall = {TP \over TP + FN}
     $$

     

4. Fallout

   * negative 샘플 중 모델이 positive로 잘못 분류한 비율.

   * $$
     Fallout = {FP \over TN + FP}
     $$

     

5. F1 score

   * Precision과 Recall의 조화 평균.

   * $$
     F_1 = 2 \times{Precision \times Recall \over Precision + Recall} = {2TP \over 2TP + FP + FN}
     $$

     

6. ROC (Receiver Operating Characteristic) Curve

   * Fallout과 Recall의 변화를 시각화한 것.

   * Recall이 크고, Fallout이 작은 모형을 좋은 모형으로 생각할 수 있다.

   * 곡선이 왼쪽 위 모서리에 가까울 수록 모델 성능이 좋다.

     ![](D:\git\nlp-sentence-classify\report\ROC_curve_Snap8.gif)

7. AUC (Area Under the Curve)

   * ROC Curve의 밑면적을 계산한 값.
   * Fallout 대비 Recall값이 클 수록 AUC가 1에 가까운 값이며 우수한 모형이다.

## 평가

10% 비율로 테스트 데이터 셋과 학습 데이터 셋을 나누어 cross-validation 으로 모델을 평가한다.  타 모델들에 MR 데이터 셋을 이용하여 평가한 Accuracy와 우리 모델의 Accuracy를 비교해 본다. 타 모델의 Accuracy는 CNN for Sentence Classification 논문[^1]의 Table2를 참고하였다.

# 실험 결과 및 분석



# 결론



# 레퍼런스

* Yoon Kim - Convolutional Neural Networks for Sentence Classification[^1]

  [^1]: http://www.aclweb.org/anthology/D14-1181

* Implementing a CNN for Text Classification in Tensorflow[^2]

  [^2]: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/

* yoonkim - CNN_sentence[^3]

  [^3]: https://github.com/yoonkim/CNN_sentence

- dennybritz - cnn-text_classification-tf[^4]

  [^4]: https://github.com/dennybritz/cnn-text-classification-tf

- 분류모델 성능 평가 지표[^5]

  [^5]: http://here.deepplus.co.kr/?p=24