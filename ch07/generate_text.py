# coding: utf-8
import sys
sys.path.append('..')
# from rnnlm_gen_test import RnnlmGen
from rnnlm_gen import RnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()  # 가중치 매개변수 : 무작위 초깃값(학습 X)
# model.load_params('../ch06/Rnnlm.pkl')      # 앞 단원에서 학습을 끝낸 가중치 매개변수를 읽어들임

# start 문자와 skip 문자 설정
start_word = 'you'      #  첫 단어를 you 로 설정
start_id = word_to_id[start_word]       # you 에 해당하는 id 를 start_id 로 설정
skip_words = ['N', '<unk>', '$']        # 샘플링하지 않을 단어 : ['N', '<unk>', '$'] 로 설정
skip_ids = [word_to_id[w] for w in skip_words]      # 샘플링 하지 않을 단어에 해당하는 skip_ids
# 문장 생성
word_ids = model.generate(start_id, skip_ids)       # 문장 생성 -> 단어 ID 들을 배열 형태로 반환
txt = ' '.join([id_to_word[i] for i in word_ids])   # 각 배열 요소 사이에 ' ' 넣어서 문장 형태로 변환
txt = txt.replace(' <eos>', '.\n')                  # <eos> 는 줄바꿈 문자로.
print(txt)



