# ch07 - RNN을 사용한 문장 생성



1. generate_text.py

- 아무런 학습도 수행하지 않은 모델을 사용한 결과 (모델의 가중치 초기값이 무작위)

  ```python
  model = RnnlmGen()
  #model.load_params('../ch06/Rnnlm.pkl')
  ```

  ```python
  "you feels stronger clouds tailored belts succeeding \* posture unanimously ltd. reach scowcroft exported valley proceeding specialty benton highlight hit kemp naming parks recording suitors holidays seng adjuster wrongdoing kinds beings minnesota revived cooperative law surprised 26-week tuesday bearing aligned k supervisor frederick junk-holders gin gerard evaluating ally newly gambling chandler specialize o'brien crop cigarettes expense presents bone manufactures tenants qintex philadelphia hence neglected reminded financial wendy exclusivity cincinnati stressed hopes sights unanimously inaccurate billionaire restraints moments advisory larry bologna allocated pork-barrel will kohl forecast filing answers filed forward combustion conner offering lighting need phrase armonk executives dire leslie detroit"
  ```

  👉 모델의 가중치 초깃값으로 무작위한 값을 사용했기 때문에 의미가 통하지 않는 문장이 출력됨.

- 앞 단원에서 학습을 끝낸 모델을 사용한 결과 

  학습을 끝낸 가중치 매개변수를 읽어들인다. 

  ```python
  model = RnnlmGen()
  model.load_params('../ch06/Rnnlm.pkl')      # 앞 단원에서 학습을 끝낸 가중치 매개변수를 읽어들임
  ```

  ```python
  "you along advocate equity-purchase busy centered pence statistical struggled issuance confident reflecting far ind. new-issue protesters cela navy malignant productivity supports aggregates compete boards couple components departure extraordinarily here checks recreation virgin chores las third mmi serves u.s. neil output uncovered breaker francs violent ballot olympics acceptable rhone-poulenc presence red leg weisfield balked commitments laboratory thrown projection adjuster estimate but sharper react criticisms imposes manville informal bankamerica tripled garden milan tumultuous philippine start schwab grows batch pat capcom tickets luck charts sperry kageyama non-food dole bomb rothschilds 45-year-old desperate fibers pa discovery ltv exemption authors fluor preparation manipulation follow niche"
  ```

  👉 앞의 예제보다 훨씬 더 자연스러운 문장으로 나타난다.

  하지만 아직 부자연스러운 문장이 발견된다. → 더 나은 언어모델을 사용하면 된다. (**7.1.3 더 좋은 문장으로 부터** 다시 공부)

2. rnnlm_gen.py

   👉 여기에서 주목할 것은 이렇게 생성한 문장은 훈련 데이터에 존재하지 않는, 말 그대로 새로 생성된 문장이라는 것

   왜냐하면 언어 모델은 훈련데이터를 암기한 것이 아니라, **훈련 데이터에서 사용된 단어의 정렬 패턴을 학습한 것** 이기 때문.

3. generate_better_text.py

   👉 더 좋은 언어 모델 
  

4. train_seq2seq.py

   - 덧셈 문제 seq2seq 모델 학습
   - seq2seq 모델 개선 방법 : Reverse, Peeky

   👉 입력 데이터 반전(Reverse)
   - 입력 데이터를 반전시키는 것
   - **x_train[:, ::-1]** --> 배열의 행을 반전시킨다.
    # coding: utf-8...
    [[ 3  0  2 ...  0 11  5]
    [ 4  0  9 ...  8  8 10]
    [ 1  1  2 ...  9  0  5]
    ...
    [ 3  1 10 ...  8  0  3]
    [ 1  2  8 ...  0  5  5]
    [ 8  2  4 ... 10  5  5]]
    [[ 5 11  0 ...  2  0  3]
    [10  8  8 ...  9  0  4]
    [ 5  0  9 ...  2  1  1]
    ...
    [ 3  0  8 ... 10  1  3]
    [ 5  5  0 ...  8  2  1]
    [ 5  5 10 ...  4  2  8]]
  - 입력 데이터 반전 후 정답률 : 50% 까지 올라감

  👉 엿보기(Peeky) (peeky_seq2seq.py)
  - Encoder의 출력벡터 h 를 Decoder 의 최초 시각 LSTM 계층 뿐만 아니라 **다른 계층** 에게도 전해주는 것
  - 모든 시각의 LSTM 계층, Affine 계층
  LSTM 계층의 입력 : 인코더의 hidden state + embedding 출력
  Affine 계층의 입력 : 인코더의 hidden state + lstm hidden state
  --> 2개의 입력이 되는데 이것을 concatnate 로 연결
  - 엿보기 후 정답률 : 80% 이상까지 올라감

  💥 Peeky 주의점
    : Peeky 를 이용하게 되면 신경망의 가중치 매개변수가 커져 계산량이 늘어남

  👉 Reverse + Peeky
  - 정답률 : 10에폭을 넘어서면서 정답률이 90%를 넘고 최종적으로 100%에 가까워짐

  👀 여기까지는 Seq2Seq의 작은 개선이고 더 큰 개선은 다음 장 --> **어텐션**


