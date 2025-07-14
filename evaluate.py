import os
import json
import random
import glob
import subprocess
import re
from typing import List, Dict, Tuple
from collections import defaultdict

# For evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from bert_score import score as bert_score_score

# For OpenAI API
from openai import OpenAI
from dotenv import load_dotenv

# For Konlpy
from konlpy.tag import Okt

# Import general legal advisor
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from legal_multiagent.agents.general_legal_advisor import general_legal_advice

# Load environment variables
load_dotenv(override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Assuming the root directory of the project is the current working directory
PROJECT_ROOT = os.getcwd()
PRECEDENT_DATA_DIR = os.path.join(PROJECT_ROOT, '판례데이터')
MAIN_APP_PATH = os.path.join(PROJECT_ROOT, 'legal_multiagent', 'main.py')

okt = Okt() # Initialize Konlpy Okt globally

def select_random_precedent_file() -> str:
    """
    판례 데이터 디렉토리에서 무작위로 하나의 JSON 파일 경로를 선택합니다.
    """
    json_files = glob.glob(os.path.join(PRECEDENT_DATA_DIR, '**', '*.json'), recursive=True)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {PRECEDENT_DATA_DIR}")
    return random.choice(json_files)

def load_precedent_data(file_path: str) -> Dict:
    """
    주어진 판례 JSON 파일에서 필요한 데이터를 로드합니다.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    basic_facts = data.get('facts', {}).get('bsisFacts', [])
    court_dcss = data.get('dcss', {}).get('courtDcss', [])
    cnclsns = data.get('close', {}).get('cnclsns', [])
    
    # courtDcss와 cnclsns를 결합하여 최종 결론의 참조 텍스트로 사용
    reference_conclusion = " ".join(court_dcss + cnclsns)
    
    return {
        'basicFacts': basic_facts,
        'referenceConclusion': reference_conclusion
    }

def generate_client_request_text(basic_facts: List[str]) -> str:
    """
    추출된 basicFacts를 기반으로 OpenAI API를 사용하여 클라이언트 요청 텍스트를 생성합니다.
    """
    facts_str = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate(basic_facts)])
    
    messages = [
        {
            "role": "system",
            "content": """
            당신은 법률 전문가입니다. 주어진 '기초 사실'을 바탕으로, 해당 사건에 대해 일반인이 변호사에게 문의할 법률 상담 요청 텍스트를 생성해야 합니다.
            요청 텍스트는 다음과 같은 내용을 포함해야 합니다:
            1. 자신이 처한 상황에 대한 설명 (주어진 기초 사실을 바탕으로 자연스럽게 서술)
            2. 궁금한 법적 조치, 절차, 필요한 증거, 예상 소요 시간 등 구체적인 질문
            3. 실제 사람이 작성한 것처럼 자연스러운 문체
            """
        },
        {
            "role": "user",
            "content": f"""
            ## 기초 사실
            {facts_str}

            ## 생성할 법률 상담 요청 텍스트 예시:
            저는 다음과 같은 상황에 처해 있습니다. [기초 사실을 바탕으로 상황 설명].
            이러한 상황에서 제가 취할 수 있는 법적 조치에는 어떤 것들이 있으며,
            실제로 소송을 진행하게 된다면 어떤 절차와 증거가 필요한지,
            보증금을 돌려받기까지 얼마나 걸릴 수 있는지 등 구체적인 조언을 받고 싶습니다.
            """
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 또는 "gpt-3.5-turbo" 등 사용 가능한 모델
            messages=messages,
            temperature=0.7, # 창의성을 위해 적절한 온도 설정
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating client request text with OpenAI API: {e}")
        # 오류 발생 시 기존 템플릿 사용
        request_template = f"""\
저는 다음과 같은 상황에 처해 있습니다.\n\
{facts_str}\n\
\n\
이러한 상황에서 제가 취할 수 있는 법적 조치에는 어떤 것들이 있으며,\n\
실제로 소송을 진행하게 된다면 어떤 절차와 증거가 필요한지,\n\
보증금을 돌려받기까지 얼마나 걸릴 수 있는지 등 구체적인 조언을 받고 싶습니다.\n\
"""
        return request_template.strip()

def run_main_and_get_output(client_request_text: str) -> Dict:
    """
    main.py를 실행하고 그 결과를 파싱하여 반환합니다.
    main.py 파일을 임시로 수정하여 client_request_text를 입력으로 받도록 합니다.
    """
    original_main_content = ""
    with open(MAIN_APP_PATH, 'r', encoding='utf-8') as f:
        original_main_content = f.read()

    # main.py 수정: user_input을 하드코딩
    # client_request_text 내의 삼중 따옴표를 임시로 대체하여 구문 오류 방지
    escaped_client_request_text = client_request_text.replace('"""', '___TRIPLE_QUOTE___')
    modified_main_content = re.sub(
        r"user_input = sys\\.stdin\\.read\\(\\)",
        f"user_input = \"\"\"{escaped_client_request_text}\"\"\"",
        original_main_content
    )
    
    with open(MAIN_APP_PATH, 'w', encoding='utf-8') as f:
        f.write(modified_main_content)

    result = {}
    try:
        process = subprocess.run(
            ["python", "-m", "legal_multiagent.main"],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        # main.py의 stdout에서 JSON 결과 파싱
        result = json.loads(process.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running main.py: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw stdout: {process.stdout}")
    finally:
        # main.py를 원래 상태로 되돌림
        with open(MAIN_APP_PATH, 'w', encoding='utf-8') as f:
            f.write(original_main_content)
    return result

def run_general_model_and_get_output(client_request_text: str) -> Dict:
    """
    general_legal_advisor를 직접 호출하여 결과를 파싱하여 반환합니다.
    이제 general_legal_advisor는 JSON 형식의 문자열을 반환합니다.
    """
    try:
        general_advice_json_str = general_legal_advice(user_input=client_request_text)
        # JSON 문자열을 파싱하여 딕셔너리로 변환
        advice_dict = json.loads(general_advice_json_str)
        return advice_dict
    except Exception as e:
        print(f"Error running general model or parsing JSON: {e}")
        return {"reconstructed_facts": "", "legal_conclusion": ""}

def extract_generated_texts(main_output: Dict, is_general_model: bool = False) -> Dict:
    """
    모델의 출력에서 생성된 사실관계와 최종 결론 텍스트를 추출합니다.
    is_general_model 플래그를 추가하여 일반 모델의 출력을 다르게 처리합니다.
    """
    if is_general_model:
        # 일반 모델의 출력은 이미 구조화된 딕셔너리 형태임
        return {
            'generatedBasicFacts': main_output.get('reconstructed_facts', ''),
            'generatedConclusion': main_output.get('legal_conclusion', '')
        }

    # 기존 사용자 모델(main.py) 출력 처리 로직
    final_answer = main_output.get('final_answer', '')
    
    # 마크다운 파싱하여 기초 사실과 최종 결론 추출
    basic_facts_match = re.search(r'\*\*기초 사실\*\*\n([\s\S]*?)(?=\n\n\*\*법적 쟁점\*\*|$)', final_answer)
    generated_basic_facts_raw = basic_facts_match.group(1).strip() if basic_facts_match else ''
    
    # Remove markdown bullet points and ensure consistent numbering/spacing
    cleaned_facts_lines = []
    for line in generated_basic_facts_raw.split('\n'):
        cleaned_line = re.sub(r'^\*\s*(\d+\.\s*)?', r'\1', line).strip()
        if cleaned_line:
            cleaned_facts_lines.append(cleaned_line)
    generated_basic_facts = '\n'.join(cleaned_facts_lines)
    
    final_conclusion_match = re.search(r'### 🧑‍⚖️ 최종 결론\n\n([\s\S]*)', final_answer)
    generated_conclusion = final_conclusion_match.group(1).strip() if final_conclusion_match else ''

    return {
        'generatedBasicFacts': generated_basic_facts,
        'generatedConclusion': generated_conclusion
    }

def calculate_metrics(reference_text: str, generated_text: str) -> Dict:
    """
    주어진 참조 텍스트와 생성된 텍스트 간의 BLEU, ROUGE, BertScore를 계산합니다.
    """
    metrics = {}

    # Initialize Konlpy Okt for Korean tokenization
    okt = Okt()

    # Tokenize texts using Konlpy for better Korean processing
    reference_tokens = okt.morphs(reference_text)
    generated_tokens = okt.morphs(generated_text)

    # BLEU Score
    # SmoothingFunction().method1은 짧은 문장에 대한 BLEU 점수 계산 시 ZeroDivisionError 방지
    if len(generated_tokens) > 0: # generated_tokens가 비어있지 않을 때만 계산
        metrics['BLEU'] = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=SmoothingFunction().method1)
    else:
        metrics['BLEU'] = 0.0

    # ROUGE Score
    rouge = Rouge()
    try:
        # ROUGE는 토큰화되지 않은 문자열을 입력으로 받으므로, 다시 문자열로 조인
        scores = rouge.get_scores(" ".join(generated_tokens), " ".join(reference_tokens))
        metrics['ROUGE-1_f'] = scores[0]['rouge-1']['f']
        metrics['ROUGE-2_f'] = scores[0]['rouge-2']['f']
        metrics['ROUGE-L_f'] = scores[0]['rouge-l']['f']
    except ValueError: # 입력 텍스트가 너무 짧거나 비어있을 경우
        metrics['ROUGE-1_f'] = 0.0
        metrics['ROUGE-2_f'] = 0.0
        metrics['ROUGE-L_f'] = 0.0

    # BertScore
    # BertScore는 모델을 다운로드해야 할 수 있습니다.
    # 'bert-base-multilingual-cased'와 같은 다국어 모델 사용 고려
    try:
        P, R, F1 = bert_score_score([generated_text], [reference_text], lang="ko", verbose=False)
        metrics['BertScore_P'] = P.mean().item()
        metrics['BertScore_R'] = R.mean().item()
        metrics['BertScore_F1'] = F1.mean().item()
    except Exception as e:
        print(f"BertScore calculation error: {e}")
        metrics['BertScore_P'] = 0.0
        metrics['BertScore_R'] = 0.0
        metrics['BertScore_F1'] = 0.0

    return metrics

def main_evaluation_loop(num_evaluations: int = 1) -> List[Dict]:
    """
    평가 루프를 실행하고 각 평가의 결과를 반환합니다.
    """
    all_results = []
    for i in range(num_evaluations):
        print(f"--- Evaluation {i+1}/{num_evaluations} ---")
        try:
            # 1. 판례 데이터 로드
            precedent_file = select_random_precedent_file()
            print(f"Selected precedent: {os.path.basename(precedent_file)}")
            precedent_data = load_precedent_data(precedent_file)
            
            reference_basic_facts = "\\n".join(precedent_data['basicFacts'])
            reference_conclusion = precedent_data['referenceConclusion']

            # 2. 클라이언트 요청 텍스트 생성
            client_request = generate_client_request_text(precedent_data['basicFacts'])

            # 3. 사용자 모델 실행 및 결과 추출
            user_model_output = run_main_and_get_output(client_request)
            user_generated_texts = extract_generated_texts(user_model_output, is_general_model=False)
            
            # 4. 일반 모델 실행 및 결과 추출
            general_model_output = run_general_model_and_get_output(client_request)
            general_generated_texts = extract_generated_texts(general_model_output, is_general_model=True)

            # 5. 지표 계산
            user_facts_metrics = calculate_metrics(reference_basic_facts, user_generated_texts['generatedBasicFacts'])
            user_conclusion_metrics = calculate_metrics(reference_conclusion, user_generated_texts['generatedConclusion'])
            
            general_facts_metrics = calculate_metrics(reference_basic_facts, general_generated_texts['generatedBasicFacts'])
            general_conclusion_metrics = calculate_metrics(reference_conclusion, general_generated_texts['generatedConclusion'])

            all_results.append({
                'precedent_file': os.path.basename(precedent_file),
                'user_model': {
                    'facts_metrics': user_facts_metrics,
                    'conclusion_metrics': user_conclusion_metrics
                },
                'general_model': {
                    'facts_metrics': general_facts_metrics,
                    'conclusion_metrics': general_conclusion_metrics
                }
            })
        except Exception as e:
            print(f"Error during evaluation {i+1}: {e}")
            continue
    return all_results

def display_results(results: List[Dict]):
    """
    평가 결과를 보기 쉬운 형태로 재구성하여 출력합니다.
    """
    if not results:
        print("No evaluation results to display.")
        return

    print("\n--- Evaluation Results ---")

    for i, res in enumerate(results):
        print(f"\n--- Result for Precedent {i+1}: {res['precedent_file']} ---")

        # --- Facts Metrics Table ---
        print("\n[1. Facts Metrics]")
        facts_header = ["Metric", "User Model", "General Model"]
        print(f"| {' | '.join(facts_header)} |")
        print(f"|{'|'.join(['---' * len(h) for h in facts_header])}|")
        
        facts_metrics = [
            ("BLEU", res['user_model']['facts_metrics'].get('BLEU', 0.0), res['general_model']['facts_metrics'].get('BLEU', 0.0)),
            ("ROUGE-L", res['user_model']['facts_metrics'].get('ROUGE-L_f', 0.0), res['general_model']['facts_metrics'].get('ROUGE-L_f', 0.0)),
            ("BertScore-F1", res['user_model']['facts_metrics'].get('BertScore_F1', 0.0), res['general_model']['facts_metrics'].get('BertScore_F1', 0.0))
        ]

        for metric_name, user_score, general_score in facts_metrics:
            print(f"| {metric_name:<12} | {user_score:<10.4f} | {general_score:<13.4f} |")

        # --- Conclusion Metrics Table ---
        print("\n[2. Conclusion Metrics]")
        conclusion_header = ["Metric", "User Model", "General Model"]
        print(f"| {' | '.join(conclusion_header)} |")
        print(f"|{'|'.join(['---' * len(h) for h in conclusion_header])}|")

        conclusion_metrics = [
            ("BLEU", res['user_model']['conclusion_metrics'].get('BLEU', 0.0), res['general_model']['conclusion_metrics'].get('BLEU', 0.0)),
            ("ROUGE-L", res['user_model']['conclusion_metrics'].get('ROUGE-L_f', 0.0), res['general_model']['conclusion_metrics'].get('ROUGE-L_f', 0.0)),
            ("BertScore-F1", res['user_model']['conclusion_metrics'].get('BertScore_F1', 0.0), res['general_model']['conclusion_metrics'].get('BertScore_F1', 0.0))
        ]

        for metric_name, user_score, general_score in conclusion_metrics:
            print(f"| {metric_name:<12} | {user_score:<10.4f} | {general_score:<13.4f} |")


if __name__ == "__main__":
    # NLTK 데이터 다운로드 (최초 1회 실행 필요)
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.Downloader:
        nltk.download('punkt')
    
    # 평가 실행 (예: 3개의 판례에 대해 평가)
    evaluation_results = main_evaluation_loop(num_evaluations=1) # 일단 1개로 테스트
    display_results(evaluation_results)
