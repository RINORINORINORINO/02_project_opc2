from openai import OpenAI
import os
import re
import json
import time
import logging
from typing import List, Dict, Union, Optional, Any, Tuple
from functools import lru_cache
import concurrent.futures
from dotenv import load_dotenv
import threading
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 전역 설정
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1  # 초 단위
MAX_WORKERS = 3  # 병렬 처리 워커 수
CACHE_DIR = "cache/media_suggestions"  # 캐시 저장 디렉토리

# API 호출 세마포어 추가
api_semaphore = threading.Semaphore(3)  # 최대 3개 동시 요청

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def api_call_with_retry(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """
    API 호출 함수를 재시도 로직으로 감싸는 유틸리티 함수
    지수 백오프 전략 사용
    
    Args:
        func: 호출할 함수
        *args, **kwargs: 함수에 전달할 인자들
        max_retries: 최대 재시도 횟수
        
    Returns:
        함수 호출 결과
    """
    # 세마포어로 동시 API 요청 제한
    with api_semaphore:
        for attempt in range(max_retries):
            try:
                # 첫 번째 시도가 아니면 약간의 지연 추가
                if attempt > 0:
                    # 지수 백오프 + 무작위성(jitter) 추가
                    base_delay = BASE_RETRY_DELAY * (2 ** attempt)
                    jitter = random.uniform(0, 0.5 * base_delay)
                    delay = base_delay + jitter
                    logger.warning(f"⚠️ API 호출 실패 ({attempt+1}/{max_retries}), {delay:.2f}초 후 재시도")
                    time.sleep(delay)
                
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    # 마지막 시도였다면 예외 발생
                    raise
                
                logger.warning(f"⚠️ API 호출 실패 ({attempt+1}/{max_retries}): {str(e)}")

def generate_media_suggestions(
    script: str, 
    topic: str,
    output_dir: str = "output_media",
    use_cache: bool = True,
    parallel_processing: bool = True
) -> str:
    """
    스크립트를 분석하여 각 구간에 필요한 미디어 요소 제안 생성
    국제관계/지정학/세계사 전문가 콘텐츠에 특화된 시각자료 제안
    
    Args:
        script: 생성된 스크립트 텍스트
        topic: 콘텐츠 주제
        output_dir: 결과물 저장 디렉토리
        use_cache: 캐시 사용 여부
        parallel_processing: 병렬 처리 사용 여부
        
    Returns:
        미디어 제안 텍스트
    """
    try:
        # API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("❌ OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정하세요.")
            return create_default_geopolitical_media_suggestions(topic)
        
        # 출력 및 캐시 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
        
        # 캐시 키 생성 (주제 + 스크립트 해시)
        if use_cache:
            cache_key = f"{topic.replace(' ', '_')[:30]}_{hash(script) % 10000000}"
            cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
            
            # 캐시 확인
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        logger.info(f"✅ 캐시에서 미디어 제안 로드: {cache_file}")
                        return cached_data['suggestions']
                except Exception as e:
                    logger.warning(f"⚠️ 캐시 로드 실패: {str(e)}")
        
        logger.info("🔍 국제관계/지정학/세계사 전문 미디어 요소 분석 중...")
        
        # 이미 스크립트에 포함된 영상 지시사항 추출
        existing_media = extract_existing_media_directions(script)
        
        # 텍스트 청크로 분할
        script_segments = split_script_to_segments(script)
        
        # 기본 미디어 제안 생성
        main_suggestions = generate_main_media_suggestions(script, topic, existing_media)
        
        if parallel_processing:
            # 병렬 처리로 추가 미디어 요소 생성
            additional_elements = generate_additional_media_parallel(script, topic)
        else:
            # 순차 처리로 추가 미디어 요소 생성
            additional_elements = generate_additional_media_sequential(script, topic)
        
        # 전체 미디어 제안 조합
        full_suggestions = f"""# 📹 국제관계/지정학/세계사 전문 미디어 요소 제안

## 📼 스크립트별 전문 시각자료 제안
{main_suggestions}

{additional_elements}

## 📝 국제관계/지정학/세계사 콘텐츠 제작 팁
- 모든 지도와 지리적 시각자료에는 정확한 국경선과 출처를 명시하세요
- 역사적 사건과 시기를 표현할 때는 정확한 연대와 맥락을 제공하세요
- 국제기구, 조약, 협정 등을 언급할 때 정확한 로고와 공식 명칭을 사용하세요
- 전문가 인용 시 소속 기관과 전문 분야를 자막으로 표시하세요
- 국가별 데이터 비교 시 객관적 지표와 최신 통계를 사용하세요
- 복잡한 국제관계 개념을 설명할 때는 간단한 도표와 인포그래픽을 활용하세요
- 지정학적 긴장 지역을 설명할 때는 중립적인 시각에서 여러 관점을 제시하세요
- 역사적 사건의 현대적 함의를 설명할 때는 명확한 연결고리를 시각화하세요
"""
        
        # 결과 저장
        result_path = os.path.join(output_dir, f"intl_media_suggestions_{int(time.time())}.txt")
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(full_suggestions)
        
        # 캐시 저장
        if use_cache:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'suggestions': full_suggestions, 'timestamp': time.time()}, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ 미디어 제안 캐시 저장: {cache_file}")
            except Exception as e:
                logger.warning(f"⚠️ 캐시 저장 실패: {str(e)}")
        
        logger.info("✅ 국제관계/지정학/세계사 전문 미디어 제안 생성 완료")
        return full_suggestions
        
    except Exception as e:
        logger.error(f"⚠️ 미디어 제안 생성 중 오류: {e}")
        # 기본 미디어 제안 반환
        return create_default_geopolitical_media_suggestions(topic)

def generate_main_media_suggestions(script: str, topic: str, existing_media: List[str]) -> str:
    """
    스크립트를 분석하여 주요 미디어 제안 생성
    
    Args:
        script: 스크립트 텍스트
        topic: 주제
        existing_media: 이미 포함된 미디어 지시사항
        
    Returns:
        미디어 제안 텍스트
    """
    # 스크립트 적절히 자르기 (너무 길면 API 한도 초과)
    max_script_length = 8000  # GPT 모델 토큰 한도 고려
    truncated_script = script[:max_script_length] if len(script) > max_script_length else script
    
    prompt = f"""
당신은 국제관계, 지정학, 세계사 전문 다큐멘터리와 교육 콘텐츠 제작의 시각화 전문가입니다.
다음 국제관계/지정학/세계사 전문가 스크립트의 각 부분에 필요한 효과적인 시각 자료와 미디어 요소를 한국어로 제안해주세요.

주제: {topic}

스크립트에는 이미 다음과 같은 미디어 지시사항이 포함되어 있습니다:
{"".join([f"- {item}\n" for item in existing_media]) if existing_media else "없음"}

국제관계/지정학/세계사 전문 콘텐츠를 위한 다음 미디어 요소들을 제안해주세요:

1. 글로벌/지역 지도와 지정학적 시각화 (국경, 분쟁 지역, 자원 분포 등)
2. 역사적 사건과 조약의 타임라인
3. 국가 간 관계와 동맹 구조 다이어그램
4. 국제기구와 관련 협정의 구조도
5. 국가별 주요 지표 비교 차트 (GDP, 군사력, 외교 관계망 등)
6. 역사적 인물 및 현대 지도자의 인용구 및 정책
7. 이론적 모델과 분석 프레임워크 시각화
8. 역사적 사건 아카이브 영상 또는 이미지
9. 주요 외교/역사적 문서와 조약 텍스트

각 섹션별로 명확하고 상세한 시각화 제안을 제공해주세요. 특히 복잡한 국제관계와 지정학적 개념을 쉽게 이해할 수 있도록 하는 시각화에 중점을 두세요.

스크립트의 주요 섹션을 분석하고, 각 섹션마다 적절한 미디어 요소를 제안하는 형태로 답변해주세요.
섹션 구분이 명확하지 않은 경우 내용의 흐름에 따라 주요 토픽별로 미디어 요소를 제안해주세요.

스크립트:
{truncated_script}
"""

    try:
        def make_api_call():
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        
        # 재시도 로직으로 API 호출
        suggestions = api_call_with_retry(make_api_call)
        return suggestions
    
    except Exception as e:
        logger.error(f"⚠️ 주요 미디어 제안 생성 실패: {str(e)}")
        return """
        
### 도입부 미디어 제안
- 전 세계 지도 위에 주요 국가 및 지역 하이라이트
- 주제와 관련된 주요 인물이나 사건의 이미지
- 시청자의 관심을 끌 수 있는 역사적 사건 영상 클립
- 주제에 관한 중요 통계나 데이터를 보여주는 간략한 인포그래픽

### 본론 1 섹션 미디어 제안
- 특정 지역에 초점을 맞춘 상세 지도
- 국가 간 관계를 보여주는 네트워크 다이어그램
- 주요 사건들의 타임라인
- 관련된 역사적 문서나 조약의 이미지

### 본론 2 섹션 미디어 제안
- 주요 국가들의 지표를 비교하는 바 차트나 레이더 차트
- 변화 추세를 보여주는 라인 그래프
- 전문가 인터뷰 영상이나 인용문
- 관련 국제기구나 협정의 로고와 설명

### 결론 미디어 제안
- 미래 시나리오를 시각화한 다이어그램
- 논의된 모든 요소를 통합한 개념 맵
- 추가 학습 리소스에 대한 정보 그래픽
- 주제에 관한 핵심 통찰을 강조하는 인용문이나 통계
"""

def generate_additional_media_parallel(script: str, topic: str) -> str:
    """
    병렬 처리로 추가 미디어 요소 생성
    
    Args:
        script: 스크립트 텍스트
        topic: 주제
        
    Returns:
        추가 미디어 요소 텍스트
    """
    logger.info("🔄 추가 미디어 요소 병렬 생성 중...")
    
    # 생성할 미디어 요소 리스트
    media_elements = [
        ("stock_keywords", generate_military_stock_footage_keywords),
        ("music_suggestions", suggest_military_background_music),
        ("data_viz_suggestions", suggest_military_data_visualizations),
        ("expert_citations", suggest_expert_citations)
    ]
    
    results = {}
    
    # 작업량에 따라 워커 수 동적 조정
    worker_count = min(MAX_WORKERS, len(media_elements))

    # 병렬 처리 함수
    def process_element(element_data):
        name, func = element_data
        try:
            logger.info(f"🔄 {name} 생성 중...")
            result = func(script, topic)
            logger.info(f"✅ {name} 생성 완료")
            return name, result, True
        except Exception as e:
            logger.error(f"❌ {name} 생성 실패: {str(e)}")
            return name, "", False
    
    # 병렬 처리 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(media_elements)) as executor:

        # 일괄 제출 대신 하나씩 제출하고 완료될 때마다 다음 작업 제출
        future_to_item = {}
        remaining_items = list(media_elements)
        
        # 첫 번째 배치 제출 (워커 수만큼)
        initial_batch = remaining_items[:worker_count]
        remaining_items = remaining_items[worker_count:]
        
        for item in initial_batch:
            future = executor.submit(process_element, item)
            future_to_item[future] = item[0]  # 요소 이름 저장
        
        # 완료된 작업 처리 및 새 작업 제출
        while future_to_item:
            # 완료된 작업 하나 가져오기
            done, _ = concurrent.futures.wait(
                future_to_item, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                try:
                    name, result, success = future.result()
                    if success:
                        results[name] = result
                    
                    # 새 작업 제출 (남은 항목이 있는 경우)
                    if remaining_items:
                        new_item = remaining_items.pop(0)
                        new_future = executor.submit(process_element, new_item)
                        future_to_item[new_future] = new_item[0]
                    
                except Exception as e:
                    logger.error(f"❌ 미디어 요소 처리 중 예외 발생: {str(e)}")
                
                # 처리된 future 삭제
                del future_to_item[future]
    
    # 결과 조합
    combined_results = ""
    
    # 스톡 푸티지 키워드
    if "stock_keywords" in results:
        combined_results += f"""## 🔑 전문 영상/이미지 검색 키워드
{results["stock_keywords"]}

"""
    
    # 배경음악 제안
    if "music_suggestions" in results:
        combined_results += f"""## 🎵 배경음악 제안
{results["music_suggestions"]}

"""
    
    # 데이터 시각화 제안
    if "data_viz_suggestions" in results:
        combined_results += f"""## 📊 군사/국제정치 데이터 시각화 제안
{results["data_viz_suggestions"]}

"""
    
    # 전문가 인용 및 출처 표시 제안
    if "expert_citations" in results:
        combined_results += f"""## 📚 전문가 인용 및 출처 표시 제안
{results["expert_citations"]}

"""
    
    # 결과 없는 경우 기본값 제공
    if not combined_results:
        combined_results = create_default_additional_elements()
    
    return combined_results

def generate_additional_media_sequential(script: str, topic: str) -> str:
    """
    순차 처리로 추가 미디어 요소 생성
    
    Args:
        script: 스크립트 텍스트
        topic: 주제
        
    Returns:
        추가 미디어 요소 텍스트
    """
    logger.info("🔄 추가 미디어 요소 순차 생성 중...")
    
    combined_results = ""
    
    # 스톡 푸티지 키워드
    try:
        logger.info("🔄 전문 영상/이미지 검색 키워드 생성 중...")
        stock_keywords = generate_military_stock_footage_keywords(script, topic)
        combined_results += f"""## 🔑 전문 영상/이미지 검색 키워드
{stock_keywords}

"""
        logger.info("✅ 전문 영상/이미지 검색 키워드 생성 완료")
    except Exception as e:
        logger.error(f"❌ 전문 영상/이미지 검색 키워드 생성 실패: {str(e)}")
    
    # 배경음악 제안
    try:
        logger.info("🔄 배경음악 제안 생성 중...")
        music_suggestions = suggest_military_background_music(script, topic)
        combined_results += f"""## 🎵 배경음악 제안
{music_suggestions}

"""
        logger.info("✅ 배경음악 제안 생성 완료")
    except Exception as e:
        logger.error(f"❌ 배경음악 제안 생성 실패: {str(e)}")
    
    # 데이터 시각화 제안
    try:
        logger.info("🔄 데이터 시각화 제안 생성 중...")
        data_viz_suggestions = suggest_military_data_visualizations(script, topic)
        combined_results += f"""## 📊 군사/국제정치 데이터 시각화 제안
{data_viz_suggestions}

"""
        logger.info("✅ 데이터 시각화 제안 생성 완료")
    except Exception as e:
        logger.error(f"❌ 데이터 시각화 제안 생성 실패: {str(e)}")
    
    # 전문가 인용 및 출처 표시 제안
    try:
        logger.info("🔄 전문가 인용 및 출처 표시 제안 생성 중...")
        expert_citations = suggest_expert_citations(script, topic)
        combined_results += f"""## 📚 전문가 인용 및 출처 표시 제안
{expert_citations}

"""
        logger.info("✅ 전문가 인용 및 출처 표시 제안 생성 완료")
    except Exception as e:
        logger.error(f"❌ 전문가 인용 및 출처 표시 제안 생성 실패: {str(e)}")
    
    # 결과 없는 경우 기본값 제공
    if not combined_results:
        combined_results = create_default_additional_elements()
    
    return combined_results

def extract_existing_media_directions(script: str) -> List[str]:
    """
    스크립트에 이미 포함된 미디어 지시사항 추출
    
    Args:
        script: 스크립트 텍스트
        
    Returns:
        미디어 지시사항 리스트
    """
    # 영상/비디오 지시사항 패턴
    patterns = [
        r'\[영상:(.*?)\]',
        r'\[Video:(.*?)\]',
        r'\[영상\s*:\s*(.*?)\]',
        r'\[Video\s*:\s*(.*?)\]'
    ]
    
    all_matches = []
    for pattern in patterns:
        matches = re.findall(pattern, script)
        all_matches.extend(matches)
    
    # 중복 제거 및 정리
    unique_matches = set()
    for match in all_matches:
        clean_match = match.strip()
        if clean_match:
            unique_matches.add(clean_match)
    
    return list(unique_matches)

def split_script_to_segments(script: str, max_segments: int = 10) -> List[str]:
    """
    스크립트를 여러 세그먼트로 분할
    
    Args:
        script: 스크립트 텍스트
        max_segments: 최대 세그먼트 수
        
    Returns:
        세그먼트 리스트
    """
    # 섹션 헤더로 구분 시도
    sections = re.split(r'\n##\s+.*?\s+##\n', script)
    
    if len(sections) > 1 and len(sections) <= max_segments:
        return sections
    
    # 문단으로 구분 시도
    paragraphs = re.split(r'\n\n+', script)
    
    if len(paragraphs) <= max_segments:
        return paragraphs
    
    # 너무 많은 문단이 있는 경우 병합
    segments = []
    current_segment = ""
    
    for para in paragraphs:
        if len(current_segment) + len(para) < 1000:
            if current_segment:
                current_segment += "\n\n" + para
            else:
                current_segment = para
        else:
            segments.append(current_segment)
            current_segment = para
    
    if current_segment:
        segments.append(current_segment)
    
    # 여전히 너무 많은 세그먼트가 있는 경우 제한
    if len(segments) > max_segments:
        # 평균 길이 계산
        avg_length = len(script) // max_segments
        
        # 세그먼트 병합
        merged_segments = []
        current_segment = ""
        
        for segment in segments:
            if len(current_segment) + len(segment) < avg_length * 1.5:
                if current_segment:
                    current_segment += "\n\n" + segment
                else:
                    current_segment = segment
            else:
                merged_segments.append(current_segment)
                current_segment = segment
        
        if current_segment:
            merged_segments.append(current_segment)
        
        segments = merged_segments
    
    return segments

def generate_international_stock_footage_keywords(script: str, topic: str) -> str:
    """
    주제와 스크립트를 분석하여 국제관계/지정학 관련 스톡 영상/이미지 검색에 필요한 키워드 추천
    
    Args:
        script: 스크립트 텍스트
        topic: 주제
        
    Returns:
        키워드 추천 텍스트
    """
    # 스크립트 축약 (API 토큰 제한 고려)
    script_excerpt = script[:1500] if len(script) > 1500 else script
    
    prompt = f"""
"{topic}" 주제의 국제관계/지정학/세계사 전문 영상 제작을 위한 스톡 영상/이미지 검색 키워드를 15개 추천해주세요.
각 키워드는 영어와 한국어로 제공하고, 국제관계/지정학/세계사 콘텐츠에 특화된 검색 필터링 옵션도 제안해주세요.

다음과 같은 국제관계/지정학/세계사 관련 키워드를 포함하세요:
- 국제 정상회담/외교적 만남
- 역사적 조약 체결 장면
- 국제기구 회의/총회
- 유명한 국제관계 이론가/학자
- 지정학적 중요 지역
- 정치적 지도자/외교관
- 역사적 국제 갈등/협력 사례
- 국경/분쟁 지역
- 문화적 외교/교류

스크립트의 주요 내용을 반영한 키워드를 생성하세요.
각 키워드는 번호를 매겨 목록으로 제시하고, 영어 키워드와 그 한국어 번역을 함께 제공하세요.

스크립트 일부:
{script_excerpt}
"""

    try:
        def make_api_call():
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        
        # 재시도 로직으로 API 호출
        return api_call_with_retry(make_api_call)
    except Exception as e:
        logger.error(f"⚠️ 스톡 영상 키워드 생성 실패: {str(e)}")
        return """
1. 국제 정상회담 (영어: international summit)
2. 유엔 총회 (영어: UN General Assembly)
3. 외교 협상 테이블 (영어: diplomatic negotiation table)
4. 역사적 조약 서명 (영어: historic treaty signing)
5. 국경 지역 항공 촬영 (영어: border region aerial view)
6. 세계 지도 애니메이션 (영어: world map animation)
7. 유명 정치 지도자 연설 (영어: political leader speech)
8. 국제기구 본부 (영어: international organization headquarters)
9. 국제 분쟁 지역 (영어: international conflict zone)
10. 역사적 외교 문서 (영어: historic diplomatic document)

검색 필터링 옵션:
- 출처: UN, EU, 외교부, 주요 대학 및 연구기관
- 날짜: 최근 5년 이내 자료 (시사성), 또는 특정 역사적 시기
- 라이센스: 상업적 사용 가능 콘텐츠
- 지역별 필터링: 관련 국가나 지역으로 검색 범위 제한
"""

def suggest_military_background_music(script: str, topic: str) -> str:
    """
    군사/국제정치 콘텐츠에 적합한 배경음악 제안
    
    Args:
        script: 스크립트 텍스트
        topic: 주제
        
    Returns:
        배경음악 제안 텍스트
    """
    # 스크립트 축약 (API 토큰 제한 고려)
    script_excerpt = script[:1000] if len(script) > 1000 else script
    
    prompt = f"""
다음 군사/국제정치 전문가 스크립트의 분위기와 내용을 분석하여 적절한 배경음악 스타일을 제안해주세요.
스크립트의 서로 다른 부분(서론, 본론, 결론 등)에 맞게 3-5가지 다른 배경음악을 제안하고,
로열티 프리 음악을 찾을 수 있는 사이트도 추천해주세요.

특히 다음과 같은 군사/국제정치 콘텐츠에 적합한 음악 스타일을 고려하세요:
- 심각한 전략적 분석 부분용 음악
- 역사적 군사 사건 설명용 드라마틱한 음악
- 지정학적 긴장 묘사를 위한 긴장감 있는 음악
- 국제 협력/외교 성과 설명용 희망적 음악

각 음악 제안에는 다음 정보를 포함하세요:
1. 스크립트의 어느 부분에 적합한지
2. 음악의 분위기와 스타일
3. 적합한 악기 구성
4. 템포와 다이나믹스 특성

주제: {topic}

스크립트 일부:
{script_excerpt}
"""

    try:
        def make_api_call():
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        
        # 재시도 로직으로 API 호출
        return api_call_with_retry(make_api_call)
    except Exception as e:
        logger.error(f"⚠️ 배경음악 제안 생성 실패: {str(e)}")
        return """
1. 서론: 깊고 무게감 있는 현악기와 저음 브라스
   - 분위기: 권위 있고 전문적인 톤 설정
   - 악기 구성: 첼로, 더블 베이스, 호른, 탐페니
   - 템포: 느리고 안정적인 리듬, 점진적 빌드업

2. 역사적 배경/사례: 미니멀한 피아노와 타악기
   - 분위기: 사실 전달에 집중하면서도 역사적 무게감 부여
   - 악기 구성: 피아노, 스트링 앙상블, 미니멀 타악기
   - 템포: 중간 템포, 반복적 패턴

3. 전략 분석 부분: 일정한 리듬의 전자 요소
   - 분위기: 분석적 사고와 현대적 접근 강조
   - 악기 구성: 전자 베이스, 심플한 신스, 아날로그 신디사이저
   - 템포: 중간~빠른 템포, 리드미컬한 펄스

4. 결론/함의: 브라스와 현악기의 점진적 상승
   - 분위기: 전략적 중요성과 미래 전망 강조
   - 악기 구성: 풀 오케스트라, 브라스 섹션 강조
   - 템포: 중간 템포, 다이나믹한 빌드업

군사/국제정치 콘텐츠에 적합한 로열티 프리 음악 사이트:
- Epidemic Sound: 다큐멘터리/드라마 섹션
- PremiumBeat: '긴장감' 및 '드라마' 카테고리
- AudioJungle: '기업/다큐멘터리' 컬렉션
- Artlist.io: 'Epic/Dramatic' 섹션
"""

def suggest_military_data_visualizations(script: str, topic: str = None) -> str:
    """
    군사/국제정치 스크립트에서 데이터 시각화 제안
    
    Args:
        script: 스크립트 텍스트
        topic: 주제 (선택사항)
        
    Returns:
        데이터 시각화 제안 텍스트
    """
    # 군사/국제정치 관련 숫자 패턴 찾기
    military_numbers = re.findall(r'(\d+(?:\.\d+)?(?:\s*(?:%|퍼센트|percent|명|개|원|달러|위|등|년|척|대|기|문|발사대|킬로미터|km|마일|해리)))', script)
    military_budget = re.findall(r'((?:국방비|방위비|군비|예산|지출)\s*\d+(?:\.\d+)?(?:\s*(?:억|조|만|천|달러|원)))', script)
    military_capability = re.findall(r'((?:병력|탄두|미사일|함정|항공기|전차|포|장갑차|잠수함|전투기)\s*\d+(?:\.\d+)?(?:\s*(?:기|문|대|척|문|기|문)))', script)
    
    # 키워드 분석 (통계/데이터 관련)
    data_keywords = [
        "비교", "증가", "감소", "추세", "통계", "데이터", "수치", "지표", "분석",
        "비율", "예측", "추산", "평가", "순위", "교전", "손실", "전과", "포획", "파괴"
    ]
    
    # 키워드 매치 확인
    keyword_matches = []
    for keyword in data_keywords:
        if re.search(r'\b' + keyword + r'\b', script):
            keyword_matches.append(keyword)
    
    all_stats = military_numbers + military_budget + military_capability
    
    # 추출된 데이터 포인트가 있는지 확인
    if all_stats or keyword_matches:
        prompt = f"""
당신은 군사 및 국제정치 데이터 시각화 전문가입니다.
다음 스크립트에서 추출한 군사/국제정치 관련 데이터를 가장 효과적으로 시각화할 수 있는 방법을 제안해주세요.

각 데이터 포인트에 대해 적합한 차트/그래프/다이어그램 유형을 추천하고, 
군사/국제정치 전문 콘텐츠에 적합한 시각화 디자인에 대한 조언을 제공해주세요.

추출된 군사/국제정치 데이터:
{', '.join(all_stats[:15]) if all_stats else '명시적 데이터 포인트 없음'}

발견된 데이터 관련 키워드:
{', '.join(keyword_matches) if keyword_matches else '없음'}

주제: {topic if topic else '군사/국제정치 분석'}

다음과 같은 군사/국제정치 전문 시각화를 구체적으로 제안하세요:
1. 각 데이터 유형에 가장 적합한 차트/그래프 유형
2. 권장 색상 팔레트 및 디자인 가이드라인
3. 레이블링 및 주석 처리 방법
4. 정보의 계층화 방법
5. 사용할 수 있는 시각화 도구나 소프트웨어

군사/국제정치 콘텐츠에 적합한 5-8개의 구체적인 시각화 제안과 함께, 각 시각화의 구현 방법에 대한 
간략한 설명을 제공해주세요.
"""

        try:
            def make_api_call():
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            
            # 재시도 로직으로 API 호출
            return api_call_with_retry(make_api_call)
        except Exception as e:
            logger.error(f"⚠️ 데이터 시각화 제안 실패: {str(e)}")
    
    # 명시적인 데이터가 없는 경우
    return """
스크립트에서 시각화할 명확한 군사/국제정치 관련 데이터가 제한적이지만, 다음과 같은 전문 시각화를 권장합니다:

1. 군사력 비교 레이더 차트
   - 설명: 여러 국가의 군사력 요소(병력 수, 탱크, 항공기, 함정, 국방비 등)를 다차원적으로 비교
   - 색상: 국가별 구분 색상, 높은 대비와 명확한 구분
   - 도구: D3.js, Tableau

2. 지역별 군사 자산 배치 지도
   - 설명: 위성 지도에 군사 자산을 아이콘으로 표시, 군사력 집중도를 히트맵으로 표현
   - 색상: 블루(해군), 그린(육군), 그레이(공군) 계열의 전문적 색상
   - 도구: ArcGIS, CARTO

3. 국방비 시계열 그래프
   - 설명: 주요국 국방비 변화 추이를 연도별로 시각화
   - 색상: 네이비 블루, 버건디 레드, 다크 그린 등 전통적 군사 색상
   - 도구: Google Data Studio, Excel

4. 동맹 관계 네트워크 다이어그램
   - 설명: 국가 간 군사 동맹과 협력 관계를 노드와 엣지로 시각화
   - 색상: 동맹 유형별 색상 코드, 관계 강도에 따른 선 굵기 변화
   - 도구: Gephi, NodeXL

5. 전략적 의사결정 트리
   - 설명: 군사적 의사결정 과정과 가능한 결과를 계층적으로 시각화
   - 색상: 위험도에 따른 색상 구분(적색-높음, 황색-중간, 녹색-낮음)
   - 도구: Lucidchart, Microsoft Visio

시각화 디자인 팁:
- 한 화면에 너무 많은 정보를 넣지 않고 계층적으로 정보 전달
- 군사 전통에 맞는 색상과 아이콘 사용 (카무플라주 패턴, 군사 장비 실루엣 등)
- 모든 차트에 출처와 데이터 기준일 명시
- 주요 시점에는 타임스탬프나 이벤트 마커 추가
"""

def suggest_expert_citations(script: str, topic: str = None) -> str:
    """
    군사/국제정치 스크립트에서 전문가 인용 및 출처 표시 제안
    
    Args:
        script: 스크립트 텍스트
        topic: 주제 (선택사항)
        
    Returns:
        전문가 인용 및 출처 표시 제안 텍스트
    """
    # 전문가/기관 이름 찾기
    expert_pattern = r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)(?=\s*(?:에 따르면|의 연구|의 분석|박사|교수|연구원|소장|전 장성|전략가|분석가|이론가|states|argues|claims|suggests|according to))'
    institution_pattern = r'((?:[A-Z][a-zA-Z]*\s*)+(?:Institute|Center|Council|대학교|연구소|연구원|센터|기관|연맹|협회|University|College|Foundation|Agency|Organization))'
    
    experts = re.findall(expert_pattern, script)
    institutions = re.findall(institution_pattern, script)
    
    # 중복 제거 및 정리
    experts = list(set([e.strip() for e in experts if len(e.strip()) > 5]))
    institutions = list(set([i.strip() for i in institutions if len(i.strip()) > 5]))
    
    # 발견된 전문가나 기관이 있는 경우
    if experts or institutions or topic:
        prompt = f"""
당신은 군사 및 국제정치 전문 영상 콘텐츠의 인용 및 출처 디자인 전문가입니다.
다음 스크립트에서 발견된 전문가와 기관을 기반으로, 효과적인 인용 표시 방법과 출처 제안을 제공해주세요.

발견된 전문가 (있는 경우):
{', '.join(experts[:10]) if experts else '없음'}

발견된 기관 (있는 경우):
{', '.join(institutions[:10]) if institutions else '없음'}

주제: {topic if topic else '군사/국제정치 분석'}

다음 내용을 포함한 군사/국제정치 전문 콘텐츠를 위한 인용 및 출처 표시 가이드를 제공해주세요:

1. 전문가 인용문 표시 방법 (텍스트 디자인, 위치, 지속 시간 등)
2. 연구기관 데이터 출처 표시 방법 (신뢰성을 높이는 시각적 요소)
3. 주제에 적합한 추가 권위 있는 군사/국제정치 전문가 및 기관 추천 (3-5개)
4. 인용/출처 표시의 시각적 일관성을 위한 디자인 템플릿
5. 인용 및 출처에 대한 구체적인 애니메이션 및 트랜지션 제안

실제 영상에서 사용할 수 있는 구체적인 시각 요소(색상, 폰트, 레이아웃, 애니메이션 등)를 명시해주세요.
"""

        try:
            def make_api_call():
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            
            # 재시도 로직으로 API 호출
            return api_call_with_retry(make_api_call)
        except Exception as e:
            logger.error(f"⚠️ 전문가 인용 제안 생성 실패: {str(e)}")
    
    # 전문가나 기관이 발견되지 않은 경우
    return """
스크립트에서 구체적인 전문가나 기관 인용이 발견되지 않았습니다. 다음과 같은 권위 있는 군사/국제정치 전문가와 기관을 인용에 활용하세요:

## 권위 있는 군사/국제정치 전문가 인용 추천
1. 국제전략연구소(IISS) 소속 연구원들
2. 스톡홀름 국제평화연구소(SIPRI) 군축 전문가
3. 각국 전직 국방장관/국방 관료
4. 저명한 군사사/국제관계학 교수
5. 전직 고위 장성 및 군사 전략가

## 인용문 시각화 템플릿
1. 전문가 인용문 디자인:
   - 폰트: Oswald(인용문), Roboto(출처 정보)
   - 색상: 네이비 블루(#1C3144) 배경, 화이트(#FFFFFF) 텍스트
   - 레이아웃: 화면 하단 1/3에 반투명 패널, 왼쪽에 전문가 실루엣 또는 사진
   - 애니메이션: 좌측에서 슬라이드 인, 3-5초 유지, 페이드 아웃
   - 추가 요소: 인용부호 아이콘, 기관 로고(있는 경우)

2. 데이터 출처 표시:
   - 폰트: Roboto Condensed, 12-14pt
   - 위치: 차트/그래프 우하단
   - 형식: "출처: [기관명], [연도]"
   - 색상: 차트 배경과 대비되는 중간 톤의 그레이

3. 연구 인용 형식:
   - 저자 이름(굵게), 제목, 기관, 연도
   - 예: "John Smith, 'Strategic Balance in Asia', RAND Corporation, 2023"
   - 위치: 화면 하단 1/4 영역, 중앙 정렬
   - 지속 시간: 주요 포인트 언급 시 3-4초

## 일관된 디자인을 위한 가이드라인
- 색상 팔레트: 네이비 블루(#1C3144), 버건디 레드(#990000), 다크 그레이(#333333), 라이트 그레이(#CCCCCC), 화이트(#FFFFFF)
- 폰트 조합: 제목/강조(Oswald), 본문(Roboto)
- 애니메이션: 모든 인용 요소에 일관된 트랜지션(페이드, 슬라이드) 사용
- 배치: 화면을 방해하지 않도록 중요 시각 요소와 겹치지 않는 위치 선정
- 지속 시간: 글자 수에 따라 조정(평균 5-7초)

인용문 사용 시, 영상의 흐름을 방해하지 않도록 간결하고 핵심적인 내용만 선별하여 표시하세요.
"""

def create_default_geopolitical_media_suggestions(topic: str) -> str:
    """
    기본 국제관계/지정학 미디어 제안 생성
    
    Args:
        topic: 주제
        
    Returns:
        기본 미디어 제안 텍스트
    """
    return f"""# 📹 {topic} - 국제관계/지정학/세계사 전문 미디어 요소 제안

## 📼 스크립트별 전문 시각자료 제안
### 도입부
- 글로벌 지도 오프닝 (주요 국가와 지역 강조)
- 주제 관련 역사적 중요 사건 아카이브 영상 짧은 몽타주
- 제목과 부제목이 있는 세계지도 또는 지구본 모티프의 타이틀 카드
- 관련 국가의 국기 및 지도자 시각 요소

### 본론
- 국가/지역별 데이터 비교 차트 (GDP, 군사력, 인구 등)
- 관련 국제기구 및 협정의 로고와 설명
- 중요 지정학적 지역의 상세 지도
- 국가 간 관계도 및 동맹 구조
- 역사적 사건의 연대표
- 주요 국제관계/지정학 전문가 인용구 텍스트 오버레이
- 관련 국제 조약/협정 핵심 조항 텍스트 카드
- 이론적 개념 설명을 위한 인포그래픽

### 결론
- 미래 시나리오 시각화
- 핵심 요점을 강조하는 지정학적 요약 다이어그램
- 추가 정보를 위한 권위 있는 국제관계/지정학 리소스 목록

{create_default_additional_elements()}

## 📝 국제관계/지정학/세계사 콘텐츠 제작 팁
- 모든 지도와 지리적 시각자료에는 정확한 국경선과 출처를 명시하세요
- 역사적 사건과 시기를 표현할 때는 정확한 연대와 맥락을 제공하세요
- 국제기구, 조약, 협정 등을 언급할 때 정확한 로고와 공식 명칭을 사용하세요
- 전문가 인용 시 소속 기관과 전문 분야를 자막으로 표시하세요
- 국가별 데이터 비교 시 객관적 지표와 최신 통계를 사용하세요
- 복잡한 국제관계 개념을 설명할 때는 간단한 도표와 인포그래픽을 활용하세요
- 지정학적 긴장 지역을 설명할 때는 중립적인 시각에서 여러 관점을 제시하세요
- 역사적 사건의 현대적 함의를 설명할 때는 명확한 연결고리를 시각화하세요
"""

def create_default_additional_elements() -> str:
    """
    기본 추가 미디어 요소 텍스트 생성
    
    Returns:
        기본 추가 미디어 요소 텍스트
    """
    return """## 🔑 국제관계/지정학 스톡 영상/이미지 검색 키워드
1. 국제 정상회담 (영어: international summit)
2. 세계 지도 애니메이션 (영어: world map animation)
3. 역사적 조약 서명 (영어: historic treaty signing)
4. 유엔 안보리 회의 (영어: UN Security Council meeting)
5. 지정학적 분쟁 지역 (영어: geopolitical hotspot)
6. 국제기구 본부 (영어: international organization headquarters)
7. 외교 협상 장면 (영어: diplomatic negotiations)
8. 국경 지역 항공 촬영 (영어: border region aerial view)
9. 역사적 전환점 아카이브 (영어: historic turning point archive)
10. 글로벌 경제 포럼 (영어: global economic forum)

## 🎵 국제관계/지정학 콘텐츠 배경음악 제안
1. 서론: 웅장하고 무게감 있는 오케스트라 - 국제적 긴장감과 중요성 조성
2. 역사적 배경: 클래식한 현악기와 피아노 - 역사적 맥락과 시간의 흐름 표현
3. 지정학적 분석: 현대적인 미니멀 사운드 - 분석적 관점과 객관성 강조
4. 결론부: 점진적으로 빌드업되는 오케스트라 - 미래 전망과 함의 강조

로열티 프리 음악 사이트:
- Epidemic Sound: 다큐멘터리/드라마 섹션
- PremiumBeat: 기업/다큐멘터리 카테고리
- AudioJungle: 뉴스/다큐멘터리 컬렉션
- Artlist.io: 시네마틱/드라마틱 섹션

## 📊 국제관계/지정학 데이터 시각화 제안
1. 레이더 차트 - 국가별 다차원 지표 비교 (GDP, 군사력, 외교 영향력 등)
2. 지리적 히트맵 - 분쟁 발생 빈도나 외교적 중요성 강조
3. 타임라인 - 주요 국제 협약 및 갈등의 역사적 진행
4. 관계도 - 국가 간 동맹, 갈등, 무역 관계를 네트워크로 시각화
5. 스택 바 차트 - 각국의 시간에 따른 상대적 영향력 변화
6. 도넛 차트 - 국제기구 의결권이나 예산 배분 구조 표시

## 📚 전문가 인용 및 출처 표시 제안
1. 전문가 인용문 표시:
   - 큰 따옴표와 함께 화면 중앙에 핵심 인용구 표시
   - 인용구 아래에 전문가 이름, 소속기관, 직위 표기
   - 소속 기관 로고를 작게 함께 표시

2. 추천 권위 있는 출처:
   - 외교안보연구소, 국제관계연구원 등 국내 유명 연구기관
   - 카네기 국제평화재단, 브루킹스 연구소 등 글로벌 싱크탱크
   - 국제관계학회지, Foreign Affairs, Foreign Policy 등 전문 저널
   - 유엔, 세계은행, IMF 등 국제기구 공식 보고서

3. 디자인 가이드:
   - 글꼴: 타이틀 - Noto Sans, 본문 - Noto Serif
   - 색상: 딥 블루, 버건디 레드, 차콜 그레이의 전문적 색상 팔레트
   - 인용구 배경: 반투명 패널 또는 얇은 테두리 사용
"""

def batch_generate_media_suggestions(
    scripts: List[str], 
    topics: List[str],
    output_dir: str = "output_media",
    use_cache: bool = True
) -> List[str]:
    """
    여러 스크립트에 대한 미디어 제안을 병렬로 생성
    
    Args:
        scripts: 스크립트 리스트
        topics: 주제 리스트
        output_dir: 결과물 저장 디렉토리
        use_cache: 캐시 사용 여부
        
    Returns:
        생성된 미디어 제안 경로 리스트
    """
    if len(scripts) != len(topics):
        logger.error(f"❌ 스크립트 수 ({len(scripts)})와 주제 수 ({len(topics)})가 일치하지 않습니다.")
        return []
    
    if not scripts:
        logger.warning("⚠️ 처리할 스크립트가 없습니다.")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_scripts = len(scripts)
    logger.info(f"🔄 {total_scripts}개 스크립트의 미디어 제안 생성 시작")

    # 작업량에 따라 워커 수 동적 조정
    worker_count = min(MAX_WORKERS, total_scripts)
    
    # 병렬 처리 함수
    def process_script(script_data):
        idx, (script, topic) = script_data
        try:
            logger.info(f"[{idx+1}/{total_scripts}] '{topic}' 미디어 제안 생성 중...")
            
            # 각 스크립트마다 고유한 출력 경로 생성
            script_output_dir = os.path.join(output_dir, f"script_{idx+1}")
            
            suggestions = generate_media_suggestions(
                script, 
                topic, 
                output_dir=script_output_dir,
                use_cache=use_cache
            )
            
            # 결과 저장
            result_path = os.path.join(output_dir, f"media_suggestions_{idx+1}.txt")
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(suggestions)
            
            logger.info(f"✅ [{idx+1}/{total_scripts}] '{topic}' 미디어 제안 생성 완료")
            return idx, result_path, True
        except Exception as e:
            logger.error(f"❌ [{idx+1}/{total_scripts}] 미디어 제안 생성 중 오류: {str(e)}")
            return idx, "", False
    
    # 병렬 처리 실행
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        
        # 일괄 제출 대신 하나씩 제출하고 완료될 때마다 다음 작업 제출
        future_to_idx = {}
        remaining_items = list(enumerate(zip(scripts, topics)))
        
        # 첫 번째 배치 제출 (워커 수만큼)
        initial_batch = remaining_items[:worker_count]
        remaining_items = remaining_items[worker_count:]
        
        for item in initial_batch:
            future = executor.submit(process_script, item)
            future_to_idx[future] = item[0]
        
        # 완료된 작업 처리 및 새 작업 제출
        while future_to_idx:
            done, _ = concurrent.futures.wait(
                future_to_idx, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 새 작업 제출 (남은 항목이 있는 경우)
                    if remaining_items:
                        new_item = remaining_items.pop(0)
                        new_future = executor.submit(process_script, new_item)
                        future_to_idx[new_future] = new_item[0]
                    
                except Exception as e:
                    logger.error(f"❌ 미디어 제안 생성 중 예외 발생: {str(e)}")
                    results.append((future_to_idx[future], "", False))
                
                # 처리된 future 삭제
                del future_to_idx[future]
    
    # 결과 정렬 (원래 순서대로)
    results.sort(key=lambda x: x[0])
    
    # 성공한 경로만 필터링
    successful_paths = [res[1] for res in results if res[2]]
    
    success_count = len(successful_paths)
    logger.info(f"🏁 미디어 제안 생성 완료: 성공 {success_count}개, 실패 {total_scripts - success_count}개")
    
    return successful_paths

if __name__ == "__main__":
    # 테스트 코드
    test_script = """
    북한의 핵개발 프로그램은 지난 20년간 동북아시아 안보 구조에 중대한 변화를 가져왔습니다. [영상: 북한 핵실험 지역 위성 이미지]
    
    랜드연구소의 브루스 베넷 박사는 "북한의 핵무기는 단순한 억제력을 넘어 지역 패권을 위한 전략적 지렛대로 활용되고 있다"고 분석했습니다.
    
    특히 ICBM 기술의 발전은 미국 본토를 사정권에 두는 게임 체인저로 작용하고 있습니다. 이는 북한의 국방비 증가와 함께 지역 군사 균형에 영향을 미치고 있으며, 한국과 일본의 MD(미사일 방어) 체계 구축을 가속화시키는 요인이 되고 있습니다.
    """
    
    test_topic = "북한 핵 위협과 동북아 안보"
    
    print("🔄 미디어 제안 생성 테스트 실행 중...")
    suggestions = generate_media_suggestions(test_script, test_topic)
    print("\n=== 생성된 미디어 제안 일부 ===")
    print(suggestions[:500] + "..." if len(suggestions) > 500 else suggestions)
    
    # 배치 처리 테스트 (선택적)
    if "--batch-test" in sys.argv:
        print("\n🔄 배치 처리 테스트 실행 중...")
        batch_results = batch_generate_media_suggestions(
            [test_script, test_script], 
            [test_topic, "북한의 미사일 프로그램"]
        )
        print(f"✅ 배치 처리 결과: {len(batch_results)}개 생성 완료")