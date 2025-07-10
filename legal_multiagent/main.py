import sys
import json
import time
from legal_multiagent.graph.workflow import create_workflow
from dotenv import load_dotenv

def main():
    # .env 파일 로드
    load_dotenv(override=True)

    # 1. 워크플로우 그래프 생성
    graph = create_workflow()

    # 2. 표준 입력에서 user_input 읽기
    user_input = """저는 2022년 3월 1일부터 서울시 마포구에 있는 다세대주택의 2층을 전세로 임차하여 거주해 왔습니다.
계약 당시 보증금은 1억 5천만 원이었고, 계약기간은 2년으로 설정되어 있었으며, 2024년 2월 29일에 종료되었습니다.
집주인과는 표준임대차계약서를 작성하였고, 보증금 반환과 관련하여 특약사항은 따로 명시하지 않았습니다.
계약 갱신은 하지 않기로 하고, 저는 계약 종료일에 맞춰 이사를 준비하고 새로운 거처도 마련하였습니다.

하지만 이사 하루 전인 2024년 2월 28일, 집주인이 갑자기 보증금이 당장 마련되지 않았다며 반환을 미룰 수밖에 없다고 통보해 왔습니다.
이에 따라 저는 일단 새 집으로 이사를 완료한 후, 보증금 반환을 요청하는 내용증명을 보냈지만,
집주인은 계속해서 “자금 사정이 어렵다”는 이유로 반환을 미루고 있습니다.

현재 해당 주택에는 새로운 세입자도 들어오지 않은 상태이며,
집주인은 전세보증금을 반환할 계획도, 일정도 명확하게 제시하지 않고 있습니다.
저는 해당 보증금으로 새 집 전세자금을 충당해야 하는 상황이었기에
현재 은행 대출을 받아 이사비용을 충당한 상태이며, 이로 인해 경제적 손해와 정신적 스트레스가 상당합니다.

또한 집주인은 연락을 회피하고 있으며, 전화나 메시지에도 제대로 응답하지 않고 있어 자력으로 보증금을 돌려받기 어려운 상황입니다.
이러한 상황에서 제가 취할 수 있는 법적 조치에는 어떤 것들이 있으며,
실제로 소송을 진행하게 된다면 어떤 절차와 증거가 필요한지,
보증금을 돌려받기까지 얼마나 걸릴 수 있는지 등 구체적인 조언을 받고 싶습니다."""

    # 3. 입력 데이터 정의
    initial_state = {"user_input": user_input}

    # 4. 그래프 실행
    # print("✅ LangGraph 실행 준비 완료", file=sys.stderr) # 디버깅용
    start_time = time.time()
    result = graph.invoke(initial_state)

    # 5. 결과 JSON으로 표준 출력
    output_data = {
        "final_answer": result["final_answer"],
        "execution_time": f"{time.time() - start_time:.2f}초"
    }
    print(json.dumps(output_data, ensure_ascii=False))

if __name__ == "__main__":
    main()