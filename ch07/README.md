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