from main import *
import pickle

food_name_list, food_keyword_list = init_function() #


print("####################################################")


### 피클 파일로 저장 ###
with open("data/food_name_data.pickle","wb") as fw:
    pickle.dump(food_name_list, fw)
    
with open("data/food_keyword_data.pickle", "wb") as fw:
    pickle.dump(food_keyword_list, fw)
 
### 피클 파일 불러오기 ###
with open("food_name_data.pickle","rb") as fr:
    food_name_data = pickle.load(fr)

print(food_name_data)


with open("food_keyword_data.pickle","rb") as fr:
    food_keyword_data = pickle.load(fr)

print(food_keyword_data)