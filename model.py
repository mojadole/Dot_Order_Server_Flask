from transformers import ElectraModel, ElectraTokenizerFast # 모델 로드
from torch.utils.data import DataLoader # 배치사이트 형태로 만들어서 실제로 학습할 때 이용할 수 있게 형태를 만들어주는 라이브러리
from typing import List, Dict # typing
import torch.nn as nn # 파이토치를 사용해서 신경망을 정의할 때 사용하는 패키지
import torch # pytorch


class SentenceBert(nn.Module):

    def __init__(self, model=None, pooling_type: str = "mean") -> None:
	## 모델
        super().__init__() # 파이토치로 layer나 model 구현할 때 -> 기본적으로 작성 (변수 상속 ...)

        name = "monologg/koelectra-base-v3-discriminator" # 기본 모델
        self.model = model if model else ElectraModel.from_pretrained(name) # 모델 입력된 경우 입력된 모델로 설정 / None인 경우 기본모델로 설정

	## 풀링 타입 (기본 값 -> mean)
        if pooling_type in ["mean", "max", "cls"] and type(pooling_type) == str:
            self.pooling_type = pooling_type
        else:
            raise ValueError("'pooling_type' only ['mean','max','cls'] possible")


    def forward(self, **kwargs):
	## 설정
        attention_mask = kwargs["attention_mask"]
        last_hidden_state = self.model(**kwargs)["last_hidden_state"]

	## cls 풀링 : [cls] token을 sentence embedding으로 활용
        if self.pooling_type == "cls":
            result = last_hidden_state[:, 0]

	## max 풀링 : 문장 내 여러 토큰 중 max token만 추출하여 sentence embedding으로 활용
        if self.pooling_type == "max":
            num_of_tokens = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[num_of_tokens == 0] = -1e9
            result = torch.max(last_hidden_state, 1)[0]

	## mean 풀링 : 문장 내 토큰을 평균하여 sentence embedding으로 활용 => 우리가 사용하는 풀링 방법
        if self.pooling_type == "mean":
            num_of_tokens = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float() # 토큰 개수

            sum_embeddings = torch.sum(last_hidden_state * num_of_tokens, 1) # 총합

            total_num_of_tokens = num_of_tokens.sum(1) # 총 개수
            total_num_of_tokens = torch.clamp(total_num_of_tokens, min=1e-9) # 최소값 설정한 총 개수로 업뎃

            result = sum_embeddings / total_num_of_tokens # 평균

        return {"sentence_embedding": result}
