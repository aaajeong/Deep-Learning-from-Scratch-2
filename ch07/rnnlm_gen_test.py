# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):
    # generate : 문장 생성을 수행하는 메서드
    # start_id : 최초로 주는 단어의 ID
    # sample_size : 샘플링하는 단어의 수
    # skip_ides : 이 리스트에 속하는 단어 ID 는 샘플링 되지 않게 해줌
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        np.set_printoptions(threshold=sys.maxsize) # np 배열 한번에 출력

        x = start_id        # 시작단어 - 예제에서는 'you'
        while len(word_ids) < sample_size:      # 100개의 단어로 이루어진 문장을 만들때 까지.
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)     # 각 단어의 점수를 출력한다. (정규화 되기 전 값)0

            # for i in score:
            #     print(i, '번째 단어 점수: ', score)
            

            p = softmax(score.flatten())        # 이 점수들을 소프트맥스 함수를 사용하여 정규화
            # print("각 단어 확률 정규화 : " , p.reshape(100, 100))

            
            p_test = np.argmax(p)
            print('p_test : ', p_test)

            sampled = np.random.choice(len(p), size=5, p=p)     # 확률분포 p 로부터 다음 단어를 샘플링한다.
            sampled = np.argmax(p)
            print('sampled 5개 : ', sampled , ', 확률 : ', p[sampled] ) 
        
            i = np.argmax(p[sampled])
            sampled = sampled[i]
            if (skip_ids is None) or (sampled not in skip_ids): 
                x = sampled
                word_ids.append(int(x))
        print("------------------------------")
        # print(x)
        # print(score)
        print(word_ids)
        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)