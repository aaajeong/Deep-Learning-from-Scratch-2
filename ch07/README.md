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

  

