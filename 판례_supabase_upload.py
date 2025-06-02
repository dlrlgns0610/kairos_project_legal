import os
import json
from datetime import datetime
from supabase import create_client
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Supabase 연결
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# 판례 데이터 최상위 폴더 경로
BASE_DIR = "판례데이터"

# 사건 유형 폴더
CASE_TYPES = ["민사", "형사", "행정"]

# 실패 목록 저장
failed_files = []

# 중복 및 갱신 필요 확인 함수
def needs_update(case_no):
    try:
        response = supabase.table("precedents_full").select("id").eq("caseno", case_no).execute()
        if not response.data:
            return False, None
        return True, response.data[0]["id"]
    except Exception as e:
        print(f"❗ 업데이트 여부 확인 중 오류: {case_no} / {e}")
        return False, None

# 판례 삽입 함수
def process_json_file(file_path, case_year):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        info = data.get("info", {})
        case_no = info.get("caseNo", "")
        if not case_no:
            raise ValueError("caseNo 없음")

        needs_update_flag, existing_id = needs_update(case_no)
        if existing_id and not needs_update_flag:
            print(f"⚠️ 이미 존재하고 갱신 불필요: {case_no}")
            return

        judgment_date = info.get("judmnAdjuDe", "").strip() or None

        case_type = file_path.split(os.sep)[1]

        record = {
            "casefield": info.get("caseField", ""),
            "detailfield": info.get("detailField", ""),
            "trailfield": info.get("trailField", ""),
            "casenm": info.get("caseNm", ""),
            "courtnm": info.get("courtNm", ""),
            "judmnadjude": judgment_date,
            "caseno": info.get("caseNo", ""),
            "relatelaword": json.dumps(info.get("relateLaword", []), ensure_ascii=False),
            "qotatprcdnt": json.dumps(info.get("qotatPrcdnt", []), ensure_ascii=False),
            "acusr": data.get("concerned", {}).get("acusr", ""),
            "dedat": data.get("concerned", {}).get("dedat", ""),
            "orgjdgmncourtnm": data.get("org", {}).get("orgJdgmnCourtNm", ""),
            "orgjdgmnadjude": data.get("org", {}).get("orgJdgmnAdjuDe", ""),
            "orgjdgmncaseno": data.get("org", {}).get("orgJdgmnCaseNo", ""),
            "disposalform": data.get("disposal", {}).get("disposalform", ""),
            "disposalcontent": json.dumps(data.get("disposal", {}).get("disposalcontent", []), ensure_ascii=False),
            "rqestobjet": json.dumps(data.get("mentionedItems", {}).get("rqestObjet", []), ensure_ascii=False),
            "acusrassrs": json.dumps(data.get("assrs", {}).get("acusrAssrs", []), ensure_ascii=False),
            "dedatassrs": json.dumps(data.get("assrs", {}).get("dedatAssrs", []), ensure_ascii=False),
            "bsisfacts": json.dumps(data.get("facts", {}).get("bsisFacts", []), ensure_ascii=False),
            "courtdcss": json.dumps(data.get("dcss", {}).get("courtDcss", []), ensure_ascii=False),
            "cnclsns": json.dumps(data.get("close", {}).get("cnclsns", []), ensure_ascii=False),
            "casetype": case_type,
        }

        if existing_id:
            supabase.table("precedents_full").update(record).eq("id", existing_id).execute()
            print(f"🔄 기존 항목 갱신: {file_path}")
        else:
            supabase.table("precedents_full").insert(record).execute()
            print(f"✅ 신규 업로드 완료: {file_path}")

    except Exception as e:
        print(f"❌ 오류 발생: {file_path} → {e}")
        failed_files.append((file_path, str(e)))

# 전체 폴더 순회
for case_type in CASE_TYPES:
    type_path = os.path.join(BASE_DIR, case_type)
    if not os.path.isdir(type_path):
        continue

    for case_year in os.listdir(type_path):
        year_path = os.path.join(type_path, case_year)
        if not os.path.isdir(year_path):
            continue

        json_files = [f for f in os.listdir(year_path) if f.endswith(".json")]
        for file_name in tqdm(json_files, desc=f"{case_type}/{case_year}"):
            file_path = os.path.join(year_path, file_name)
            process_json_file(file_path, case_year)

# 실패한 파일 출력
if failed_files:
    print("\n❗ 업로드 실패 목록:")
    for file_path, reason in failed_files:
        print(f" - {file_path} → {reason}")
else:
    print("\n✅ 모든 판례 업로드 성공!")