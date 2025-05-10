from openai import OpenAI
import os
from typing import List, Dict, Any, Optional, Tuple, Callable
from dotenv import load_dotenv
import json
import time
import re
import concurrent.futures
from functools import lru_cache
import threading
import random

# 환경 변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API 호출 관련 설정
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1  # 초 단위
MAX_WORKERS = 3  # 병렬 처리 워커 수

# API 호출 세마포어 추가
api_semaphore = threading.Semaphore(3)  # 최대 3개 동시 요청

def api_call_with_retry(func: Callable, *args, **kwargs) -> Any:
    """
    API 호출 함수를 재시도 로직으로 감싸는 유틸리티 함수
    지수 백오프 전략 사용
    
    Args:
        func: 호출할 함수
        *args, **kwargs: 함수에 전달할 인자들
        
    Returns:
        함수 호출 결과
    """
    with api_semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                # 첫 번째 시도가 아니면 약간의 지연 추가
                if attempt > 0:
                    # 지수 백오프 + 무작위성(jitter) 추가
                    base_delay = BASE_RETRY_DELAY * (2 ** attempt)
                    jitter = random.uniform(0, 0.5 * base_delay)
                    delay = base_delay + jitter
                    print(f"⚠️ API 호출 실패 ({attempt+1}/{MAX_RETRIES}), {delay:.2f}초 후 재시도")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    # 마지막 시도였다면 예외 발생
                    raise
                
                print(f"⚠️ API 호출 실패 ({attempt+1}/{MAX_RETRIES}): {str(e)}")

def advanced_summarize_texts(texts: List[str], topic: str, structure: str, style: str = "international_relations_expert", output_dir: str = "output_analysis", additional_instructions: str = "", content_types: List[str] = ["longform", "shortform1", "shortform2"]) -> Dict[str, str]:
    """
    여러 텍스트를 통합 요약하고, 주제와 논리 구조에 맞는 콘텐츠 스크립트 생성
    향상된 버전: 국제관계/지정학/세계사 전문가 관점 강화, 한국어 스크립트 작성
    병렬 처리 및 오류 처리 강화
    롱폼 및 숏폼 콘텐츠 생성
    
    Args:
        texts: 파싱된 소스 텍스트 리스트
        topic: 콘텐츠 주제
        structure: 논리 구조 (예: "서론-본론-결론")
        style: 스크립트 스타일 (international_relations_expert로 고정)
        output_dir: 중간 분석 결과물 저장 디렉토리
        additional_instructions: 스크립트 작성에 대한 추가 지시사항
        content_types: 생성할 콘텐츠 유형 리스트 (롱폼, 숏폼1, 숏폼2, 숏폼3)
        
    Returns:
        생성된 스크립트 딕셔너리 {'longform': 롱폼스크립트, 'shortform1': 숏폼스크립트1, 'shortform2': 숏폼스크립트2, ...}
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    if not texts:
        print("⚠️ 요약할 텍스트가 없습니다.")
        return {content_type: "" for content_type in content_types}
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정하세요.")
        return {content_type: "" for content_type in content_types}
    
    print(f"📊 총 {len(texts)}개 소스 분석 중...")
    
    # 1. 각 소스별 국제관계/지정학/세계사 전문가 관점 분석 (병렬 처리)
    source_summaries = analyze_sources_parallel(texts, topic, output_dir)
    
    # 분석 결과가 없는 경우 종료
    if len(source_summaries) == 0:
        print("❌ 모든 소스 분석이 실패했습니다.")
        return {content_type: "" for content_type in content_types}
    
    print("✅ 개별 소스 국제관계/지정학/세계사 분석 완료")
    
    # 2. 국제관계/지정학/세계사 전문가 관점의 통합 분석
    print("🔄 국제관계/지정학/세계사 전문가 관점에서 소스 간 통합 분석 중...")
    integrated_analysis = create_integrated_analysis(source_summaries, topic, structure, output_dir)
    
    if not integrated_analysis:
        print("❌ 통합 분석 생성 실패")
        # 실패한 경우 간단한 대체 통합 분석 생성
        integrated_analysis = create_fallback_integrated_analysis(source_summaries)
    
    # 결과 딕셔너리 초기화
    result = {content_type: "" for content_type in content_types}
    
    # 3. 선택적 콘텐츠 생성
    if "longform" in content_types:
        # 롱폼 스크립트 생성
        print("📝 국제관계/지정학/세계사 전문가 스타일의 롱폼 스크립트 생성 중...")
        result["longform"] = create_longform_script(integrated_analysis, topic, structure, additional_instructions, output_dir)
    
    # 숏폼 스크립트 생성
    shortform_indices = [int(content_type.replace("shortform", "")) for content_type in content_types if content_type.startswith("shortform")]
    
    for idx in shortform_indices:
        print(f"📝 숏폼 스크립트 #{idx} 생성 중...")
        result[f"shortform{idx}"] = create_shortform_script(integrated_analysis, topic, idx, output_dir)
    
    print("✅ 모든 스크립트 생성 완료")
    return result

def analyze_sources_parallel(texts: List[str], topic: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    병렬 처리를 사용하여 각 소스를 분석
    
    Args:
        texts: 파싱된 소스 텍스트 리스트
        topic: 콘텐츠 주제
        output_dir: 결과물 저장 디렉토리
        
    Returns:
        각 소스의 분석 결과 리스트
    """
    source_summaries = []
    
    # 텍스트가 빈 경우 건너뛰는 필터링
    valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
    
    if not valid_texts:
        print("⚠️ 분석할 유효한 텍스트가 없습니다.")
        return []

    # 작업량에 따라 워커 수 동적 조정
    worker_count = min(MAX_WORKERS, len(valid_texts))
    
    # 많은 텍스트나 큰 텍스트인 경우 워커 수 감소
    if len(valid_texts) > 5 or any(len(text) > 10000 for _, text in valid_texts):
        worker_count = max(1, worker_count - 1)
    
    print(f"📊 총 {len(valid_texts)}개 소스 분석 중... (워커: {worker_count}개)")

    # 분석 함수 정의
    def analyze_source(index_text_tuple: Tuple[int, str]) -> Dict[str, Any]:
        index, text = index_text_tuple
        try:
            print(f"📝 소스 #{index+1} 국제관계/지정학/세계사 전문가 관점 분석 중...")
            
            # 텍스트가 너무 긴 경우 앞부분만 사용
            max_chars = 15000  # 약 15,000자 제한
            truncated_text = text[:max_chars] if len(text) > max_chars else text
            if len(text) > max_chars:
                truncated_text += "\n\n[텍스트가 너무 길어 나머지는 생략되었습니다]"
                
            summary_prompt = f"""
당신은 국제관계, 지정학, 세계사 분야의 최고 전문가로, 소스 내용을 국제정치 및 역사적 관점에서 분석합니다.

소스 #{index+1}에 대한 심층 국제정치/지정학/세계사 분석을 제공해주세요. 주제는 "{topic}"입니다.
다음을 포함해야 합니다:

1. 국제정치적 핵심 요점과 지정학적 의미 (가능한 많은 구체적 정보 추출)
   - 관련 국가 및 주요 행위자 
   - 국제관계 및 세력 균형에 미치는 영향
   - 지역 및 글로벌 안보 구조와의 연관성
   - 관련된 국제기구 및 다자협력체

2. 해당 사안의 역사적 맥락과 배경 (3-5개 요점)
   - 유사한 역사적 선례와 비교
   - 시간적 흐름과 전개 과정
   - 현대 국제관계에 미치는 영향
   - 관련 조약, 협정, 국제법적 측면

3. 지정학적 함의 및 전략적 중요성
   - 관련 지역의 지리적 특성과 의미
   - 자원, 에너지, 해상 교통로 등 지정학적 요소
   - 지역 내 세력 경쟁과 패권 구도
   - 군사전략적 의미 및 안보 함의

4. 주요 관련국들의 이해관계와 정책적 입장
   - 주요국 외교정책 및 전략 분석
   - 국가 간 협력과 갈등 관계
   - 국내정치와 외교정책의 연관성
   - 주요 정책결정자들의 관점과 접근법

5. 다양한 관점과 이론적 분석틀 적용
   - 현실주의, 자유주의, 구성주의 등 IR 이론 관점
   - 지정학적 분석 모델 적용 (예: 매킨더, 스파이크먼 이론)
   - 세계체제론, 지역 안보 복합체 등 거시적 분석
   - 지역 통합과 분열의 역학 관계

6. 미래 전망 및 정책적 함의
   - 단기 및 중장기 시나리오 분석
   - 잠재적 위기와 기회 요인
   - 주요 변수와 불확실성 지점
   - 주요 행위자들의 정책옵션과 전략적 선택지

7. 국제정치/지정학적 관점에서의 가치 평가 (1-5점, 전문가 관점 평가)
   - 국제질서에 대한 중요성과 영향력
   - 지역 안정과 평화에 대한 함의
   - 국제법적, 규범적 중요성
   - 세계사적 의미와 중요도

8. 다음 요소들이 있다면 특별히 정리하세요:
   - 관련 국제회의, 정상회담, 협상 과정
   - 주요 국제조약과 협정 내용
   - 지정학적 변화를 유발한 주요 사건들
   - 국제관계 변화의 핵심 전환점들

소스 내용:
{truncated_text}
"""
            # OpenAI API 호출 (재시도 로직 포함)
            def make_api_call():
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.3,
                )
                return res.choices[0].message.content.strip()
            
            analysis = api_call_with_retry(make_api_call)
            
            # 분석 결과 저장
            source_file = os.path.join(output_dir, f"source_{index+1}_intl_analysis.txt")
            with open(source_file, "w", encoding="utf-8") as f:
                f.write(f"소스 #{index+1} 국제관계/지정학/세계사 전문가 분석\n")
                f.write("="*50 + "\n\n")
                f.write(analysis)
            
            print(f"✅ 소스 #{index+1} 국제관계/지정학 분석 완료")
            
            return {
                "index": index+1,
                "analysis": analysis,
                "success": True
            }
            
        except Exception as e:
            print(f"⚠️ 소스 #{index+1} 분석 중 오류: {e}")
            # 실패한 경우에도 간단한 요약 시도
            return {
                "index": index+1,
                "analysis": f"[분석 실패: {str(e)}]\n\n소스 내용 일부:\n{text[:500]}...",
                "success": False
            }
    
    # 병렬 처리 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        # 일괄 제출 대신 하나씩 제출하고 완료될 때마다 다음 작업 제출
        # 이렇게 하면 메모리 사용량을 줄이고 부하를 분산시킬 수 있음
        future_to_item = {}
        remaining_items = list(valid_texts)
        
        # 첫 번째 배치 제출 (워커 수만큼)
        initial_batch = remaining_items[:worker_count]
        remaining_items = remaining_items[worker_count:]
        
        for item in initial_batch:
            future = executor.submit(analyze_source, item)
            future_to_item[future] = item[0]
        
        # 완료된 작업 처리 및 새 작업 제출
        while future_to_item:
            # 완료된 작업 하나 가져오기
            done, _ = concurrent.futures.wait(
                future_to_item, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                try:
                    result = future.result()
                    source_summaries.append(result)
                    
                    # 완료된 항목 출력
                    index = future_to_item[future]
                    if result.get("success", False):
                        print(f"✅ 소스 #{index+1} 분석 완료")
                    else:
                        print(f"⚠️ 소스 #{index+1} 분석 결과 불충분")
                    
                    # 새 작업 제출 (남은 항목이 있는 경우)
                    if remaining_items:
                        new_item = remaining_items.pop(0)
                        new_future = executor.submit(analyze_source, new_item)
                        future_to_item[new_future] = new_item[0]
                    
                except Exception as e:
                    index = future_to_item[future]
                    print(f"⚠️ 소스 #{index+1} 처리 중 예외 발생: {str(e)}")
                    source_summaries.append({
                        "index": index+1,
                        "analysis": f"[분석 중 예외 발생: {str(e)}]",
                        "success": False
                    })
                
                # 처리된 future 삭제
                del future_to_item[future]
    
    # 인덱스 순으로 정렬
    source_summaries.sort(key=lambda x: x["index"])
    
    # 성공한 분석 개수 확인
    success_count = sum(1 for s in source_summaries if s.get("success", False))
    print(f"📊 {len(source_summaries)}개 소스 중 {success_count}개 성공적으로 분석 완료")
    
    return source_summaries

def create_integrated_analysis(source_summaries: List[Dict[str, Any]], topic: str, structure: str, output_dir: str) -> str:
    """
    개별 소스 분석을 통합하여 종합적인 분석 생성
    
    Args:
        source_summaries: 각 소스별 분석 결과
        topic: 콘텐츠 주제
        structure: 논리 구조
        output_dir: 결과물 저장 디렉토리
        
    Returns:
        통합 분석 텍스트
    """
    # 모든 분석 결과 연결
    all_analyses = "\n\n".join([
        f"--- 소스 #{s['index']} 국제관계/지정학 분석 ---\n{s['analysis']}" 
        for s in source_summaries
    ])
    
    # 전체 분석 결과 저장
    with open(os.path.join(output_dir, "all_intl_analyses.txt"), "w", encoding="utf-8") as f:
        f.write(all_analyses)
    
    integration_prompt = f"""
당신은 국제관계, 지정학, 세계사 분야의 최고 전문가로, 여러 소스의 정보를 종합해 국제정치와 지정학에 관한 전문적인 통합 분석을 제공합니다.

주제: {topic}
구조: {structure}

다음은 {len(source_summaries)}개 소스에 대한 개별 국제관계/지정학/세계사 전문가 분석입니다:

{all_analyses}

위 분석을 바탕으로 주제에 관한 종합적인 국제관계/지정학 전문 분석을 제공해주세요. 다음을 포함해야 합니다:

1. 사안의 명확한 국제정치적 맥락과 배경
2. 관련된 주요 국가들과 행위자들의 입장과 이해관계
3. 지정학적 중요성과 전략적 함의
4. 관련 역사적 선례와 비교 분석
5. 주요 국제관계 이론(현실주의, 자유주의, 구성주의 등)의 관점에서 해석
6. 지역 및 글로벌 안보 구조에 미치는 영향
7. 단기 및 중장기 시나리오와 전망
8. 정책적 시사점과 함의
9. 국제법 및 국제규범적 관점에서의 고려사항
10. 이슈의 역사적, 문화적, 경제적 차원의 복합적 분석

특히 다음에 주의하세요:
- 객관적이고 균형 있는 관점에서 분석하세요
- 구체적인 사례와 역사적 선례를 포함하세요
- 중요한 날짜, 사건, 합의, 조약 등의 정확한 정보를 제시하세요
- 모순되는 정보가 있을 경우 출처의 신뢰성을 평가하여 가장 정확한 정보를 제시하세요
- 지정학적 분석과 함께 지역의 사회문화적, 경제적, 역사적 맥락도 고려하세요
- 정치적 중립성을 유지하면서도 전문가적 통찰력을 보여주세요
"""
    
    try:
        def make_api_call():
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": integration_prompt}],
                temperature=0.4,
            )
            return res.choices[0].message.content.strip()
        
        # 재시도 로직으로 API 호출
        integrated_analysis = api_call_with_retry(make_api_call)
        
        # 통합 분석 결과 저장
        with open(os.path.join(output_dir, "integrated_intl_analysis.txt"), "w", encoding="utf-8") as f:
            f.write(integrated_analysis)
            
        print("✅ 국제관계/지정학/세계사 전문가 통합 분석 완료")
        return integrated_analysis
        
    except Exception as e:
        print(f"⚠️ 통합 분석 생성 중 오류: {e}")
        return ""
def create_fallback_integrated_analysis(source_summaries: List[Dict[str, Any]]) -> str:
    """
    통합 분석 생성 실패 시 대체 요약을 생성하는 함수
    
    Args:
        source_summaries: 각 소스별 분석 결과
        
    Returns:
        대체 통합 분석 텍스트
    """
    print("⚠️ 대체 통합 분석 생성 중...")
    
    # 각 소스의 첫 5줄 추출하여 결합
    fallback_analysis = "## 소스별 핵심 분석 요약\n\n"
    
    for s in source_summaries:
        lines = s['analysis'].split('\n')
        summary_lines = lines[:min(5, len(lines))]
        
        fallback_analysis += f"### 소스 #{s['index']} 주요 내용\n"
        fallback_analysis += '\n'.join(summary_lines)
        fallback_analysis += "\n\n"
    
    fallback_analysis += "## 종합 관점\n"
    fallback_analysis += "여러 소스의 분석을 종합한 결과, 다음과 같은 공통된 군사/국제정치적 관점이 도출됩니다. "
    fallback_analysis += "각 소스의 주요 관점을 고려하여 이 주제에 대한 균형 잡힌 이해가 필요합니다."
    
    return fallback_analysis

def create_longform_script(integrated_analysis: str, topic: str, structure: str, additional_instructions: str, output_dir: str) -> str:
    """
    통합 분석을 바탕으로 9-11분 길이의 롱폼 스크립트 생성
    
    Args:
        integrated_analysis: 통합 분석 텍스트
        topic: 콘텐츠 주제
        structure: 논리 구조
        additional_instructions: 추가 지시사항
        output_dir: 결과물 저장 디렉토리
        
    Returns:
        최종 롱폼 스크립트 텍스트
    """
    script_prompt = f"""
당신은 국제관계, 지정학, 세계사 전문가입니다. 복잡한 국제정치와 지정학적 주제를 전문적이면서도 흥미롭게 전달하는 콘텐츠를 제작합니다.

주제: {topic}
구조: {structure}

통합 국제관계/지정학/세계사 분석:
{integrated_analysis}

위 분석을 바탕으로 "{structure}" 구조를 따르는 전문적인 콘텐츠 스크립트를 작성해주세요.
이 스크립트는 9-11분 분량의 영상(약 2700-3300자)이 되어야 합니다.

다음 사항에 특별히 유의하세요:

1. 국제관계/지정학/세계사 전문가로서의 권위와 전문성을 유지하되, 흥미롭고 매력적인 스토리텔링으로 내용을 전달하세요
2. 시청자의 관심을 사로잡는 강력한 오프닝으로 시작하고, 핵심 질문이나 주요 명제로 호기심을 유발하세요
3. 복잡한 국제정치적 개념과 지정학 이론을 명확한 비유와 시각적 예시로 설명하세요
4. 적절한 지점에서 주요 국제관계 학자, 역사적 인물, 정치 지도자, 연구기관 등을 인용하여 신뢰성을 강화하세요
5. 영상 요소는 [영상: 설명] 형식으로 스크립트에 통합하세요
6. 역사적 사례와 현대 국제관계의 연결점을 강조하며 맥락을 제공하세요
7. 균형 잡힌 관점을 제시하되, 국제관계/지정학 전문가로서의 통찰력을 강조하세요
8. 분석적이면서도 매력적인 스토리텔링 요소를 포함하여 시청자가 끝까지 시청하게 하세요
9. 논리적인 흐름을 유지하고, 각 포인트 간의 연결을 명확히 하세요
10. 스크립트 결론에서는 핵심 내용을 요약하고, 시청자에게 생각해볼 만한 질문이나 전망을 제시하세요
11. "좋아요", "구독", "알림 설정" 등 채널 프로모션 언급은 완전히 배제하세요
12. 서두와 결론에서 채널 소개나 환영 인사를 포함하지 마세요 - 즉시 주제로 들어가세요

중요: 섹션 제목(예: "Introduction", "Development" 등)과 [영상: ...] 태그는 TTS 음성으로 읽히지 않아야 합니다. 이러한 요소는 편집용으로만 사용됩니다.

스크립트에 다음 스타일 요소를 포함하세요:
- 권위 있고 전문적이지만 친근하고 매력적인 톤
- 국제관계/지정학 전문용어 사용 시 간결한 설명 병행
- 객관적이고 분석적인 접근 방식
- 적절한 지점에서 전문가적 통찰 추가
- 각 섹션이 끝날 때마다 다음 내용으로 자연스럽게 연결되는 흐름
- 시각각적 표현이 풍부한 설명
- 논리적인 흐름과 해당 단락에 내용에 맞는 사례와 수치 자료 활용
- 중요한 포인트는 명확하고 기억하기 쉬운 문구로 강조
- 결론 부분에서는 주제와 관련된 깊이 있는 질문이나 전망으로 마무리

{additional_instructions}

이 스크립트는 한국어로 작성하며, 자연스럽고 전문적인 한국어 표현을 사용하세요. 한국 시청자들에게 친숙하면서도 전문적인 느낌을 줄 수 있도록 작성해주세요. 최종 스크립트는 약 2700-3300자 정도가 되어야 합니다.
"""
    
    try:
        # API 호출 함수
        def make_api_call():
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": script_prompt}],
                temperature=0.7,
                max_tokens=4000,
            )
            return res.choices[0].message.content.strip()
        
        final_script = api_call_with_retry(make_api_call)
        
        # 스크립트 포맷팅 개선
        final_script = format_script(final_script)
        
        # 결과물 저장
        script_file = os.path.join(output_dir, "final_longform_script.txt")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(final_script)
            
        return final_script
        
    except Exception as e:
        print(f"❌ 롱폼 스크립트 생성 중 오류: {e}")
        return ""

def create_shortform_script(integrated_analysis: str, topic: str, shortform_number: int, output_dir: str) -> str:
    """
    통합 분석을 바탕으로 숏폼 스크립트 생성
    
    Args:
        integrated_analysis: 통합 분석 텍스트
        topic: 콘텐츠 주제
        shortform_number: 숏폼 번호(1, 2, 3)
        output_dir: 결과물 저장 디렉토리
        
    Returns:
        숏폼 스크립트 텍스트
    """
    # 숏폼별 차별화 포인트 설정
    if shortform_number == 1:
        shortform_focus = "지정학적으로 가장 충격적이거나 흥미로운 사실에 집중하세요. 시청자들이 '와, 이런 사실이!' 하고 반응할 만한 콘텐츠를 제작하세요."
        shortform_title = "흥미로운 사실"
    elif shortform_number == 2:
        shortform_focus = "역사적 맥락과 현대 국제관계의 연결점을 강조하세요. 과거 사례가 현재에 어떤 함의를 갖는지 집중적으로 설명하세요."
        shortform_title = "역사적 맥락"
    else:
        shortform_focus = "이 이슈의 미래 전망과 가능한 시나리오에 집중하세요. 주요 행위자들의 다음 행보와 장기적 영향을 분석하세요."
        shortform_title = "미래 전망"
    
    script_prompt = f"""
당신은 국제관계, 지정학, 세계사 전문가입니다. 소셜 미디어 숏폼 콘텐츠용 짧고 강력한 스크립트를 작성해야 합니다.

주제: {topic}
숏폼 유형: #{shortform_number} - {shortform_title}

통합 국제관계/지정학/세계사 분석:
{integrated_analysis}

위 분석을 바탕으로 50-80초 길이의 숏폼 비디오를 위한 스크립트를 작성하세요(약 250-400자).
{shortform_focus}

국제관계/지정학/세계사의 핵심에 중점을 두고 다음 요소를 반드시 포함하세요:
1. 시청자의 관심을 즉시 사로잡는 도입부
2. 주제에 대한 핵심 통찰 1-2개
3. 국제관계에 미치는 영향과 중요성 설명
4. 놀라운 사실이나 흥미로운 관점

숏폼 스크립트 작성 지침:
1. 즉시 관심을 끄는 강력한 문장이나 질문으로 시작하세요
2. 핵심 메시지 하나에 집중하세요
3. 불필요한 세부사항은 제거하고 핵심 내용만 담백하게 전달하세요
4. 역동적이고 시각적인 언어를 사용하세요
5. 영상 요소는 **[영상: 설명]** 형식으로 스크립트에 통합하세요
6. 시청자의 궁금증을 자극하는 질문이나 생각거리로 마무리하세요
7. 절대로 채널 홍보나 구독 요청 등을 포함하지 마세요

중요: 스크립트 내 모든 형식은 일관되게 유지하세요. 섹션 구분이 필요하면 항상 **[영상: 설명]** 형식만 사용하세요.

이 스크립트는 한국어로 작성하며, 소셜 미디어 사용자들의 주의를 끌 수 있도록 흥미롭고 매력적인 내용으로 구성하세요. 스크립트 길이는 250-400자 사이로 유지해주세요.
"""
    
    try:
        # API 호출 함수
        def make_api_call():
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": script_prompt}],
                temperature=0.8,
                max_tokens=1000,
            )
            return res.choices[0].message.content.strip()
        
        shortform_script = api_call_with_retry(make_api_call)
        
        # 스크립트 포맷팅 개선
        shortform_script = format_script(shortform_script)
        
        # 결과물 저장
        script_file = os.path.join(output_dir, f"final_shortform{shortform_number}_script.txt")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(shortform_script)
            
        return shortform_script
        
    except Exception as e:
        print(f"❌ 숏폼 스크립트 #{shortform_number} 생성 중 오류: {e}")
        return ""


# 더 이상 사용되지 않음 - create_longform_script로 대체됨
# def create_final_script(integrated_analysis: str, topic: str, structure: str, additional_instructions: str, output_dir: str) -> str:
#     """
#     통합 분석을 바탕으로 최종 스크립트 생성
    
#     Args:
#         integrated_analysis: 통합 분석 텍스트
#         topic: 콘텐츠 주제
#         structure: 논리 구조
#         additional_instructions: 추가 지시사항
#         output_dir: 결과물 저장 디렉토리
        
#     Returns:
#         최종 스크립트 텍스트
#     """
#     script_prompt = f"""
# 당신은 군사 및 국제정치 전문가입니다. 복잡한 군사 및 지정학적 주제를 전문적이고 담백하게 전달하는 콘텐츠를 제작합니다.

# 주제: {topic}
# 구조: {structure}

# 통합 군사/국제정치 분석:
# {integrated_analysis}

# 위 분석을 바탕으로 "{structure}" 구조를 따르는 전문적인 콘텐츠 스크립트를 작성해주세요.
# 다음 사항에 특별히 유의하세요:

# 1. 국제정치/군사 전문가로서의 권위와 전문성을 유지하고, 담백하고 정보 중심적인 내용 전달에 집중하세요
# 2. 시청자의 관심을 사로잡는 강력한 오프닝으로 시작하고, 핵심 질문이나 주요 명제로 호기심을 유발하세요
# 3. 복잡한 군사적/지정학적 개념을 명확한 비유와 시각적 예시로 설명하세요
# 4. 적절한 지점에서 주요 군사 이론가, 안보 전문가, 국제관계 학자, 연구기관, 언론 등을 인용하여 신뢰성을 강화하세요
# 5. 영상 요소는 [영상: 설명] 형식으로 스크립트에 통합하세요
# 6. 중요한 군사적/지정학적 사례나 역사적 사건을 활용하여 시청자의 이해를 돕고 관심을 유지하세요
# 7. 균형 잡힌 관점을 제시하되, 국제정치/군사 전문가로서의 통찰력을 강조하세요
# 8. 분석적이면서도 매력적인 스토리텔링 요소를 포함하여 시청자가 끝까지 시청하게 하세요
# 9. 논리적인 흐름을 자연스럽게 유지하고, 각 포인트 간의 연결을 명확히 하세요
# 10. 스크립트 결론에서는 핵심 내용을 요약하고, 시청자에게 생각해볼 만한 질문이나 전망을 제시하세요
# 11. "좋아요", "구독", "알림 설정" 등 채널 프로모션 언급은 완전히 배제하세요
# 12. 서두와 결론에서 채널 소개나 환영 인사를 포함하지 마세요 - 즉시 주제로 들어가세요

# 스크립트에 다음 스타일 요소를 포함하세요:
# - 권위 있고 전문적인 톤
# - 군사/국제정치 전문용어 사용 시 간결한 설명 병행
# - 객관적이고 분석적인 접근 방식
# - 적절한 지점에서 전문가적 통찰 추가
# - 각 섹션이 끝날 때마다 다음 내용으로 자연스럽게 연결되는 흐름
# - 논리적인 흐름과 해당 단락에 내용에 맞는 수치 자료가 있다면 활용
# - 중요한 포인트는 명확하고 기억하기 쉬운 문구로 강조
# - 결론 부분에서는 주제와 관련된 깊이 있는 질문이나 전망으로 마무리

# {additional_instructions}

# 이 스크립트는 영어권 유튜브 시청자를 대상으로 합니다. 완벽한 원어민 수준의 영어로 작성하여 자연스럽고 전문적인 콘텐츠를 제공하세요.
# """
    
#     try:
#         # API 호출 함수
#         def make_api_call():
#             res = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[{"role": "user", "content": script_prompt}],
#                 temperature=0.7,
#                 max_tokens=4000,
#             )
#             return res.choices[0].message.content.strip()
        
#         final_script = api_call_with_retry(make_api_call)
        
#         # 스크립트 포맷팅 개선
#         final_script = format_script(final_script)
        
#         # 결과물 저장
#         script_file = os.path.join(output_dir, "final_military_script.txt")
#         with open(script_file, "w", encoding="utf-8") as f:
#             f.write(final_script)
            
#         return final_script
        
#     except Exception as e:
#         print(f"❌ 최종 스크립트 생성 중 오류: {e}")
#         return ""


def format_script(script: str) -> str:
    """스크립트 형식을 개선"""
    # 영상 지시사항 포맷 통일
    script = re.sub(r'\[영상[ \t]*:[ \t]*(.*?)\]', r'[영상: \1]', script)
    script = re.sub(r'\[Video[ \t]*:[ \t]*(.*?)\]', r'[영상: \1]', script)
    
    # 불필요한 여백 제거
    script = re.sub(r'\n{3,}', '\n\n', script)
    
    # 구간 표시가 있는 경우 강조
    sections = ["서론", "본론", "결론", "도입", "전개", "마무리", "Introduction", "Main Body", "Conclusion"]
    for section in sections:
        script = re.sub(f'(^|\n)({section})(:|\.|\n)', f'\\1\n## {section} ##\\3', script, flags=re.IGNORECASE)
    
    return script

@lru_cache(maxsize=32)
def extract_military_references(analysis: str) -> List[Dict[str, str]]:
    """통합 분석에서 군사/국제정치 관련 전문가, 연구기관, 이론 등의 참조 추출"""
    references = []
    
    # 전문가 이름 패턴
    expert_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?:\s*(?:\(|\,|\는|\은))'
    expert_matches = re.finditer(expert_pattern, analysis)
    for match in expert_matches:
        name = match.group(1).strip()
        # 일반적인 이름이 아닌 경우만 (최소 2단어 이상)
        if ' ' in name and len(name) > 5:
            references.append({
                "type": "expert",
                "name": name,
                "context": analysis[max(0, match.start() - 30):match.end() + 50]
            })
    
    # 연구기관 패턴
    institution_pattern = r'((?:[A-Z][a-z]*\s*)+(?:Institute|Center|Council|College|University|연구소|연구원|센터|기관))'
    institution_matches = re.finditer(institution_pattern, analysis)
    for match in institution_matches:
        institution = match.group(1).strip()
        if len(institution) > 8:  # 어느 정도 길이가 있는 기관명만
            references.append({
                "type": "institution",
                "name": institution,
                "context": analysis[max(0, match.start() - 30):match.end() + 50]
            })
    
    # 이론/독트린 패턴
    theory_pattern = r'((?:[A-Z][a-z]*\s*)*(?:이론|독트린|전략|doctrine|theory|strategy))'
    theory_matches = re.finditer(theory_pattern, analysis)
    for match in theory_matches:
        theory = match.group(1).strip()
        references.append({
            "type": "theory",
            "name": theory,
            "context": analysis[max(0, match.start() - 30):match.end() + 50]
        })
    
    return references

def create_military_citation_list(source_summaries: List[Dict[str, Any]]) -> str:
    """소스에서 군사/국제정치 관련 인용 목록 생성"""
    citations = []
    
    for source in source_summaries:
        # 소스별 분석에서 출처 정보 추출 시도
        analysis = source["analysis"]
        
        # 저자명 추출 시도
        author_match = re.search(r'저자(?:는|의|:|은)?\s*([^,.]+)', analysis)
        author = author_match.group(1) if author_match else "Unknown"
        
        # 제목 추출 시도
        title_match = re.search(r'제목(?:은|는|:|이)?\s*"?([^",.]+)"?', analysis)
        title = title_match.group(1) if title_match else f"Source #{source['index']}"
        
        # 연도 추출 시도
        year_match = re.search(r'(19|20)\d{2}년', analysis)
        year = year_match.group(0) if year_match else ""
        
        # 기관 추출 시도
        institution_match = re.search(r'(?:기관|출처|발행처|출판사)(?:는|의|:|은)?\s*([^,.]+)', analysis)
        institution = institution_match.group(1) if institution_match else ""
        
        citation = f"{author}. {title}. {institution} {year}".strip()
        if citation.endswith('.'):
            citation = citation[:-1]
        
        citations.append(citation)
    
    return "\n".join(citations)

if __name__ == "__main__":
    # 테스트 코드
    test_texts = [
        "이것은 첫 번째 테스트 텍스트입니다. 북한의 핵무기 개발 프로그램에 관한 내용이 포함되어 있습니다.",
        "이것은 두 번째 테스트 텍스트입니다. 미국과 중국의 군사적 긴장 관계에 관한 내용이 포함되어 있습니다."
    ]
    
    result = advanced_summarize_texts(
        test_texts, 
        "동북아시아 안보 정세", 
        "서론-본론-결론",
        style="military_expert"
    )
    
    print("\n최종 스크립트 일부:")
    print(result[:500] + "..." if len(result) > 500 else result)