from key_extraction import keywordExtractor
from transformers import ElectraModel, ElectraTokenizerFast
import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Dict
from itertools import chain, islice
import torch
import openai
from gensim.models import keyedvectors
import pickle
import os
from transformers import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.set_verbosity_error()

# load model and tokenizer
name = "monologg/koelectra-base-v3-discriminator"
model = ElectraModel.from_pretrained(name)
tokenizer = ElectraTokenizerFast.from_pretrained(name)
#base_dir 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 절대 경로를 가져옵니다.
ENGHAN_DIR = os.path.join(BASE_DIR, 'data', 'preprocess', 'eng_han.csv')



# load keywordExtractor
key = keywordExtractor(model,tokenizer,dir=ENGHAN_DIR)



# load food data
FOOD_DIR = os.path.join(BASE_DIR, 'data', 'food_data.csv')
FOOD2_DIR = os.path.join(BASE_DIR, 'data', 'food_data2.csv')
#scraping_result = pd.read_csv('data/food_data.csv',encoding='cp949')
scraping_result = pd.read_csv(FOOD2_DIR)


##################################### 따로 만든 함수 #####################################
API_KEY = 'sk-' ####### 키

# chatGPT API 사용 함수
def callChatGPT(prompt, API_KEY=API_KEY):
    
    messages = []

    #get api key
    openai.api_key = API_KEY

    messages.append({"role":"user", "content":prompt})

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    chat_response = completion.choices[0].message.content
    messages.append({"role":"assitant", "content":chat_response})

    return messages[1]["content"]

# chatGPT한테 필요한 데이터 얻는 함수
def obtain_data(menu_name):

    #chat_res_cat = callChatGPT(menu_name + " 는 [밥], [국], [면], [분식] 중에 뭐야")
    #chat_res_cat = chat_res_cat[chat_res_cat.find('[')+1:chat_res_cat.find(']')] # GPT 답변 : 메뉴 카테고리

    chat_res_des = callChatGPT(menu_name + "한줄 설명") # GPT 답변 : 메뉴 설명

    #menu_name = "라면"
    #chat_res_cat = "면"
    #chat_res_des = '라면은 말이죠'

    #menu_str = menu_name + " " + chat_res_cat + " " + chat_res_des
    menu_str = menu_name + " " + chat_res_des
    menu_list = menu_str.split()

    return menu_list

# 새로운 메뉴명 -> 메뉴명, 카테고리, 메뉴설명 -> 키워드 리스트
def get_keyword_list(menu_name):
    min_count = 2
    min_length = 1

    raw_data = obtain_data(menu_name) # 
    keyword_list = key._extract_keywords(raw_data)
    translated_keyword_list = key._map_english_to_korean(keyword_list)
    refined_keyword_list = key._eliminate_min_count_words(translated_keyword_list, min_count)
    result = list(filter(lambda x: len(x) >= min_length, refined_keyword_list))
    return (result)

# 인덱스 번호 -> 기존 메뉴들 키워드 리스트 가져오기 (지금은 사용 X)
def get_keyword_idx(num):
    min_count = 2
    min_length = 1

    doc = scraping_result.iloc[num]
  
    raw_data = _convert_series_to_list_in_main(doc)
    keyword_list = key._extract_keywords(raw_data)
    translated_keyword_list = key._map_english_to_korean(keyword_list)
    refined_keyword_list = key._eliminate_min_count_words(translated_keyword_list, min_count)
    result = list(filter(lambda x: len(x) >= min_length, refined_keyword_list))
    return (result)

# 기존 메뉴들의 food_name과 각 메뉴들에서 추출한 키워드 뽑아서 초기화하는 함수
def init_function():
    food_name = []
    food_keyword = []

    for i in range(len(scraping_result)):
        docs_keywords = extract_keyword_in_main(scraping_result.iloc[[i]])
        food_name.append(docs_keywords["food_name"][0])
        food_keyword.append(docs_keywords["keywords"][0])

    return [food_name, food_keyword]

# 메뉴 검색하는 함수
def search_menu(menu_name, food_name_list, food_keyword_list):
    search = get_keyword_list(menu_name)  # 입력된 메뉴에서 키워드 추출

    """w2v_model = keyedvectors.load_word2vec_format('data/w2v2')

    # 키워드 확장 
    recommand_keyword = w2v_model.most_similar(positive=search, topn=15)
    np_recommand_keyword = np.array(list(map(lambda x: x[0], recommand_keyword)))
    print('W2V을 활용한 키워드 확장 :', np_recommand_keyword)
    print('')"""

    # 키워드와 유사한 도서 검색

    user_point = [int(0)] * len(food_name_list)

    for search_key in search:
        for i in range(len(food_name_list)):
            if search_key in food_keyword_list[i]:
                user_point[i] = user_point[i] + int(1)

    """recommand_point = [int(0)] * len(food_name_list)

    for search_key in np_recommand_keyword:
        for i in range(len(food_name_list)):

            if search_key in food_keyword_list[i]:
                recommand_point[i] = recommand_point[i] + int(1)

    total_point = [int(0)] * len(user_point)
    for i in range(len(user_point)):
        total_point[i] = (user_point[i] * 3) + recommand_point[i]"""

    total_point = user_point

    top_k_idx = np.argsort(total_point)[::-1][:20]

    # 메뉴명 연관 점수 저장
    food_name_list = np.array(food_name_list)
    total_point = np.array(total_point)

    result = dict(zip(food_name_list[top_k_idx], total_point[top_k_idx]))

    # 음식 정보 추출
    food_info = pd.read_csv(FOOD_DIR, encoding='cp949')
    IDX = food_info.food_name.isin(list(result.keys()))

    food_recommandation_result = food_info[["food_name", "food_category"]][IDX].sort_values(
        by="food_name", key=lambda x: x.map(result), ascending=False
    ).reset_index(drop=True)

    return list(food_recommandation_result.food_name)

##################################### 기존 함수 수정한 함수 #####################################

def extract_keyword_list_in_main(doc: pd.Series, min_count: int = 2, min_length: int = 2) -> List:

	raw_data = _convert_series_to_list_in_main(doc)
	keyword_list = key._extract_keywords(raw_data)
	translated_keyword_list = key._map_english_to_korean(keyword_list)
	refined_keyword_list = key._eliminate_min_count_words(translated_keyword_list, min_count)
	return list(filter(lambda x: len(x) >= min_length, refined_keyword_list))

def _convert_series_to_list_in_main(series: pd.Series) -> List[List[str]]:

	raw_data = list(series.values)
	return list(chain(*map(lambda x: x.split(), raw_data)))

def create_keyword_embedding_in_main(doc: pd.Series) -> torch.Tensor:

	keyword_list = extract_keyword_list_in_main(doc)
	tokenized_keyword = key.tokenize_keyword(keyword_list)
	return key._create_keyword_embedding(tokenized_keyword)

def create_doc_embedding_in_main(doc: pd.Series) -> torch.Tensor:

	stringified_doc = _convert_series_to_str_in_main(doc)
	tokenized_doc = key.tokenize_keyword(stringified_doc)
	return key._create_doc_embedding(tokenized_doc)

def _convert_series_to_str_in_main(series: pd.Series) -> str:
	return " ".join(list(series.values))

def extract_keyword_in_main(docs: pd.DataFrame) -> Dict:

	keyword_embedding = map(lambda x: create_keyword_embedding_in_main(x[1]), docs.iterrows())
	doc_embedding = map(lambda x: create_doc_embedding_in_main(x[1]), docs.iterrows())
	keyword_list = map(lambda x: extract_keyword_list_in_main(x[1]), docs.iterrows())

	co_sim_score = map(
		lambda x: key._calc_cosine_similarity(*x).flatten(),
		zip(doc_embedding, keyword_embedding),
	)
	top_n_keyword = list(
		map(lambda x: key._filter_top_n_keyword(*x), zip(keyword_list, co_sim_score))
	)

	return dict(food_name=docs["food_name"].tolist(), keywords=top_n_keyword)


##################################### 전체 알고리즘 #####################################
#
# menu_name = "라면" ## 입력
#
# lst = []
#
# with open("data/food_name_data.pickle","rb") as fr:
#     food_name_list = pickle.load(fr)
#
# with open("data/food_keyword_data.pickle","rb") as fr:
#     food_keyword_list = pickle.load(fr)
#
# print('\n\n\n키워드에 따른 상위 20개 음식 추천 결과\n')
# print(search_menu(menu_name, food_name_list, food_keyword_list))

"""if menu_name in food_name:
    print("일치하는 메뉴가 있습니다.")
    lst.append(menu_name)

else :
    search_menu(menu_name, food_name, food_keyword)

if len(lst) == 0:
    print("해당 메뉴가 없습니다.")
else:
    print(lst)"""

def main_search_menu(menu_name: str):
    food_name_list, food_keyword_list = init_function()
    return search_menu(menu_name, food_name_list, food_keyword_list)