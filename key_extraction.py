from model import SentenceBert #모델 로드
from sklearn.metrics.pairwise import cosine_similarity #사이킷런방식으로 코사인유사도
from transformers import ElectraModel, ElectraTokenizerFast #모델 로드
from typing import Union, Tuple, List, Dict #typing
from collections import Counter #데이터 갯수 세기
from kiwipiepy import Kiwi #형태소 분석
from itertools import chain, islice #반복자 생성
import pandas as pd #데이터 분석
import numpy as np #수학적 연산
import torch #pytorch

class keywordExtractor: # Encoder 기반 모델을 활용해 문서 정보의 핵심 키워드를 추출하는 클래스
	# model: Encoder 기반 언어모델을 사용, 기본 값으로 "monologg/koelectra-base-v3-discriminator" 사용
	# tokenizer: 해당 모델에 맞는 토크나이저 사용
    	# dir: 영어 단어 -> 한국어 단어 또는 오탈자 -> 정상 단어로 변환하기 위해 사용하는 파일 로드
        	# ex) python -> 파이썬 || 파이선 -> 파이썬

	def __init__(self, model=None, tokenizer=None, dir: str = None) -> None:
		## 언어모델 및 형태소분석기 불러오기
		
		# models
		name = "monologg/koelectra-base-v3-discriminator" #기본 모델
		self.model = model if model else ElectraModel.from_pretrained(name)
				# 모델이 입력된 경우 입력된 모델로 설정 / None인 경우 기본 모델로 설정
		self.tokenizer = tokenizer if tokenizer else ElectraTokenizerFast.from_pretrained(name)
				#토크나이저가 입력된 경우 입력된 토크나이저로 설정 / None인 경우 기본 토크나이저로 설정
		self.sbert = SentenceBert(self.model)
		self.sbert.eval()

				# noun extractor
		self.noun_extractor = Kiwi(model_type="knlm") #kiwi를 이용해 명사 추출을 위한 객체 생성
		self.dir = dir if dir else "../../data/preprocess/eng_han.csv" 

				# mapper
		self.eng_kor_df = pd.read_csv(dir) # eng_han.csv 로드
		self._update_noun_words()

	def _update_noun_words(self):
		## kiwi에 등록되지 않은 단어 추가 
		kor_words = self.eng_kor_df
		for val in kor_words.kor.values:
			self.noun_extractor.add_user_word(val)
	
	def _extract_keywords(self, words: List[str]) -> List[List[str]]:
        	## 연결된 str을 형태소 분석하여 한글 명사 및 영단어 추출
		tokenized_words = self.noun_extractor.tokenize(" ".join(words)) 
		return [word.form for word in tokenized_words if word.tag in ("NNG", "NNP", "SL")] # 한글명사 (NNG,NNP), 영단어(SL)

	def _map_english_to_korean(self, word_list: list[str]) -> list[str]:
		## 영단어를 한국어 단어로 치환
		converter = dict(self.eng_kor_df.dropna().values)

		def map_eng_to_kor(word: str) -> str:
			## 해당 단어에 대한 한글 단어가 딕셔너리에 존재하지 않으면 영어단어 그대로 반환
			kor_word = converter.get(word)
			return kor_word if kor_word else word

		return list(map(lambda x: map_eng_to_kor(x.lower()), word_list)) #w.lower -> 소문자로 변환

	def _eliminate_min_count_words(self, candidate_keyword, min_count: int = 3): # min_count (기본값 -> 3)
		## min_count 이상으로 집계되지 않은 단어 제거
		refined_kor_words = filter(lambda x: x[1] >= min_count, Counter(candidate_keyword).items())
		return list(map(lambda x: x[0], refined_kor_words))

	def tokenize_keyword(self, text: Union[list[str], str], max_length=128) -> Dict: # max_length (기본값 -> 128)
		## 텍스트와 최대 길이를 입력 받아 텍스트를 토크나이저하고 딕셔너리 형태로 반환

		# 텍스트 유무
		if text:
			pass
		else:
			text = ["에러"]
		token = self.tokenizer( # 입력된 텍스트 토크나이저
			text,
			truncation=True,
			padding=True,
			max_length=max_length,
			stride=20,
			return_overflowing_tokens=True,
			return_tensors="pt",
			)
		token.pop("overflow_to_sample_mapping")
		return token

	def _create_keyword_embedding(self, tokenized_keyword: dict) -> torch.Tensor:
		## 토큰화된 키워드를 입력받아 각 키워드 임베딩				

		# extract attention_mask, keyword_embedding
		attention_mask = tokenized_keyword["attention_mask"] # 키워드 위치 표시
		keyword_embedding = self.model(**tokenized_keyword)["last_hidden_state"]

		# optimize attention_mask, keyword_embedding
		attention_mask, optimized_keyword_embedding = self._delete_cls_sep(
		attention_mask, keyword_embedding
		)

		# mean pooling
		keyword_embedding = self._pool_keyword_embedding(
		attention_mask, optimized_keyword_embedding
		)

		return keyword_embedding

	def _delete_cls_sep(
		self, attention_mask: torch.Tensor, keyword_embedding: torch.Tensor
		) -> Tuple[torch.Tensor, torch.Tensor]:
		## [CLS],[SEP] 토큰 제거
		attention_mask = attention_mask.detach().clone()
		keyword_embedding = keyword_embedding.detach().clone()

		# delete [cls], [sep] in attention_mask
		num_keyword = attention_mask.size(0) # 텍스트에서 추출한 키워드 수
		for i in range(num_keyword):
			sep_idx = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][-1]
			attention_mask[i][0] = 0  # [CLS] => 0
			attention_mask[i][sep_idx] = 0  # [SEP] => 0

		# delete [cls], [sep] in keyword_embedding
		boolean_mask = attention_mask.unsqueeze(-1).expand(keyword_embedding.size()).float()
		keyword_embedding = keyword_embedding * boolean_mask
		return attention_mask, keyword_embedding

	def _pool_keyword_embedding(
		self, attention_mask: torch.Tensor, keyword_embedding: torch.Tensor
		) -> torch.Tensor:
		## keyword embedding에 대해 mean_pooling 수행

		num_of_tokens = attention_mask.unsqueeze(-1).expand(keyword_embedding.size()).float() # 실제 토큰인 부분의 갯수를 구한 후, 임베딩 벡터의 차원과 크기 맞춤
		total_num_of_tokens = num_of_tokens.sum(1) # 각 입력 시퀀스의 실제 토큰 갯수 총합
		total_num_of_tokens = torch.clamp(total_num_of_tokens, min=1e-9) # 0이 될 수 없도록 최소값 설정

		sum_embeddings = torch.sum(keyword_embedding, 1) # 임베딩 벡터를 실제 토큰의 수로 나눈 값

		# Mean Pooling
		mean_pooling = sum_embeddings / total_num_of_tokens
		return mean_pooling

	def _create_doc_embedding(self, tokenized_doc: Union[list[str], str]) -> torch.Tensor:
		## sbert를 활용해 doc_embedding 생성
		return self.sbert(**tokenized_doc)["sentence_embedding"]

	def extract_keyword(self, docs: pd.DataFrame) -> Dict:
		## 도서 데이터를 기반으로 키워드 추출
		# docs : pd.DataFrame 타입의 데이터, column은 [isbn13, title, toc, intro, publisher]이어야 함

		if docs.columns.tolist() != ["isbn13", "title", "toc", "intro", "publisher"]: # 입력데이터 확인
			raise ValueError(
				f"{docs.columns.tolist()} doesn't match with ['isbn13', 'title', 'toc', 'intro', 'publisher']"
			)

		keyword_embedding = map(lambda x: self.create_keyword_embedding(x[1]), docs.iterrows()) # 키워드 임베딩
		doc_embedding = map(lambda x: self.create_doc_embedding(x[1]), docs.iterrows()) # 도서 임베딩
		keyword_list = map(lambda x: self.extract_keyword_list(x[1]), docs.iterrows()) # 키워드 리스트

		co_sim_score = map( # 각 도서와 키워드 간의 코사인 유사도 계산
			lambda x: self._calc_cosine_similarity(*x).flatten(),
			zip(doc_embedding, keyword_embedding),
		)
		top_n_keyword = list( # 코사인 유사도 계산에 따른 상위 N개의 키워드 선택
			map(lambda x: self._filter_top_n_keyword(*x), zip(keyword_list, co_sim_score))
						# N -> self.top_n_keyword 변수에 저장
		)

		return dict(isbn13=docs["isbn13"].tolist(), keywords=top_n_keyword)

	def _calc_cosine_similarity(
		self, doc_embedding: torch.Tensor, keyword_embedding: torch.Tensor
		) -> np.array:
		## 단어와 문장 간 코사인 유사도 계산
		doc_embedding = doc_embedding.detach()
		keyword_embedding = keyword_embedding.detach()

		doc_score = list(
			map(lambda x: cosine_similarity(x.unsqueeze(0), keyword_embedding), doc_embedding)
		)

		max_pooling = np.max(doc_score, axis=0)  # Max
		return max_pooling

	def _filter_top_n_keyword(
		self, keyword_list: List, co_sim_score: np.array, rank: int = 20
		) -> List:
		## top_n 키워드 추출
		keyword = dict(zip(keyword_list, co_sim_score))
		sorted_keyword = sorted(keyword.items(), key=lambda k: k[1], reverse=True)
		return list(dict(islice(sorted_keyword, rank)).keys())
