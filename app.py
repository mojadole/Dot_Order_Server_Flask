import flask
from google.cloud import dialogflow
#from dialogflow_v2 import SessionsClient
from flask import request, make_response, jsonify,Flask
import urllib
import json
import os
import requests
from google.api_core.exceptions import InvalidArgument
from google.cloud.dialogflow_v2 import SessionsClient
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google.cloud import dialogflow_v2 as dialogflow
import uuid

app = flask.Flask(__name__)

# JSON 키 파일 경로
KEY_FILE_PATH = '/Users/choehyeyeong/Documents/develop/Dot_Order_Server_Flask/desktop.json'
service_account_info = json.load(open(KEY_FILE_PATH))

# Dialogflow 클라이언트 초기화
creds = Credentials.from_authorized_user_info(service_account_info, scopes=['https://www.googleapis.com/auth/cloud-platform'])
#creds = Credentials.from_authorized_user_file(KEY_FILE_PATH, scopes=['https://www.googleapis.com/auth/cloud-platform'])


#with open('/Users/choehyeyeong/Documents/develop/Dot_Order_Server_Flask/test.json', 'r') as json_file:
#    creds_json = json.load(json_file)
#creds = Credentials.from_authorized_user_info(creds_json, scopes=['https://www.googleapis.com/auth/cloud-platform'])

client = dialogflow.SessionsClient(credentials=creds)

@app.route('/dialogflow-webhook', methods=['POST'])
def dialogflow_webhook():

    # Dialogflow에서 보낸 요청 데이터를 파싱
    req = request.get_json(force=True)

    # Dialogflow에서 보낸 메시지와 파라미터를 추출
    query = req['queryResult']['queryText']
    parameters = req['queryResult']['parameters']

    # Dialogflow API v2와의 통신을 위한 인증 정보 로드
    credentials = service_account.Credentials.from_service_account_file(
        './kusitms-chatbot.json'  # 생성한 서비스 계정 키 파일의 경로
    )

    # SessionsClient를 사용하여 Dialogflow와 세션 생성
    session_id = str(uuid.uuid4())
    session_client = SessionsClient(credentials=credentials)
    session = session_client.session_path('kusitms-chatbot-vvhu', session_id)  # 프로젝트 ID와 유니크한 세션 ID

    # Dialogflow에 보낼 쿼리 생성
    query_input = {
        'text': {
            'text': query,
            'language_code': 'ko'
        }
    }

    # Dialogflow에 쿼리 전송 및 응답 받기
    response = session_client.detect_intent(
        session=session,
        query_input=query_input
    )

    # 받은 메시지와 파라미터를 처리하는 로직 구현

    # 응답 데이터를 생성
    fulfillment_text = response.query_result.fulfillment_text
    parameters = response.query_result.parameters
    response = {
        'fulfillmentText': fulfillment_text,
        'parameters': parameters
    }

    # 응답 데이터를 JSON 형태로 변환하여 전송
    return jsonify(response)

'''
@app.route('/')
def hello_world():
    return '아니 apple 5000포트 뭔데;;;;;'
'''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port ='4444')

'''
# create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    req = request.get_json(force=True)
    action = req['queryResult']['action']  # 1
    if action == 'room':
        name = req['queryResult']['parameters']['services']  # 2

    else:
        return "test"

    return {'fulfillmentText': name}  # 3
'''