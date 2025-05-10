import os
import sys
import re
import json
import logging
from typing import List, Dict, Union, Optional, Tuple, Any
from pathlib import Path
import validators
from urllib.parse import urlparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 타입 힌트를 위한 정의
SourceType = Union[str, Dict[str, str]]  # str(URL) 또는 {'type': 'pdf', 'path': '...'} 형태
UserConfig = Dict[str, Any]  # 사용자 설정을 위한 타입

# 글로벌 설정
DEFAULT_CONFIG_PATH = "config.json"
MAX_SOURCES = 30  # 최대 소스 개수
SUPPORTED_FILE_TYPES = ['.pdf', '.docx', '.txt']  # 지원하는 파일 형식

def get_user_input(config_path: str = DEFAULT_CONFIG_PATH, force_input: bool = False) -> Dict:
    """
    사용자로부터 필요한 입력을 받는 단순화된 함수
    
    Args:
        config_path: 설정 파일 경로
        force_input: 강제로 새 입력 요청 (기본값: False)
        
    Returns:
        사용자 입력 및 구성 정보 딕셔너리
    """
    # 이전 설정 불러오기 시도
    previous_config = {}
    if not force_input and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                previous_config = json.load(f)
            print(f"✅ 이전 설정 파일 로드: {config_path}")
        except Exception as e:
            print(f"⚠️ 설정 파일 로드 실패: {str(e)}")
    
    print("\n" + "="*50)
    print("🎬 국제관계/지정학/세계사 전문 한국어 유튜브 콘텐츠 자동 생성")
    print("="*50)
    
    # 주제 입력
    topic = get_topic_input(previous_config)
    
    # 소스 입력
    sources = get_sources_input(previous_config)
    
    # 구조는 기본값 사용
    structure = "서론-본론-결론"
    if "structure" in previous_config:
        structure = previous_config["structure"]
    
    # 기본 설정 값
    result = {
        "topic": topic,
        "sources": sources,
        "structure": structure,
        "style": "international_relations_expert",
        "voice": "echo",
        "parallel_workers": 3,
        "use_whisper": True,
        "optimize_tts": True,
        "additional_instructions": "",
        "content_types": ["longform", "shortform1", "shortform2"]
    }
    
    # 이전 설정에서 기본값이 아닌 값 복원
    for key in ["voice", "parallel_workers", "use_whisper", "optimize_tts", 
                "additional_instructions", "content_types"]:
        if key in previous_config:
            result[key] = previous_config[key]
    
    # 설정 저장
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"✅ 설정 저장 완료: {config_path}")
    except Exception as e:
        print(f"⚠️ 설정 저장 실패: {str(e)}")
    
    # 입력 요약 확인
    show_input_summary(result)
    
    return result

def load_previous_config(config_path: str) -> Dict:
    """
    이전 설정 파일 불러오기
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리 (파일이 없으면 빈 딕셔너리)
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ 이전 설정 파일 로드: {config_path}")
            return config
        else:
            logger.debug(f"⚠️ 이전 설정 파일 없음: {config_path}")
            return {}
    except Exception as e:
        logger.warning(f"⚠️ 설정 파일 로드 실패: {str(e)}")
        return {}

def save_config(config: Dict, config_path: str) -> bool:
    """
    현재 설정을 파일로 저장
    
    Args:
        config: 저장할 설정 딕셔너리
        config_path: 저장할 파일 경로
        
    Returns:
        성공 여부
    """
    try:
        # 설정에서 저장하지 않을 항목 제거 (선택적)
        save_config = {k: v for k, v in config.items() if k != 'temp_data'}
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(save_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 설정 저장 완료: {config_path}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ 설정 저장 실패: {str(e)}")
        return False

def get_topic_input(previous_config: Dict) -> str:
    """
    주제 입력 받기
    
    Args:
        previous_config: 이전 설정 정보
        
    Returns:
        입력된 주제
    """
    previous_topic = previous_config.get("topic", "")
    
    if previous_topic:
        print(f"\n🎯 주제를 입력하세요 (이전: '{previous_topic}'):")
        print("   (기본값으로 이전 주제를 사용하려면 엔터)")
        topic = input("> ").strip()
        
        if not topic:
            topic = previous_topic
            print(f"   이전 주제 '{topic}'을 사용합니다.")
    else:
        print("\n🎯 주제를 입력하세요:")
        topic = input("> ").strip()
    
    if not topic:
        topic = "제목 없음"
        print("   주제가 입력되지 않아 '제목 없음'으로 설정됩니다.")
    
    return topic

def get_sources_input(previous_config: Dict) -> List[SourceType]:
    """
    소스 입력 받기
    
    Args:
        previous_config: 이전 설정 정보
        
    Returns:
        입력된 소스 리스트
    """
    previous_sources = previous_config.get("sources", [])
    
    print(f"\n🔗 소스를 입력하세요 (최대 {MAX_SOURCES}개):")
    print("- 기사/블로그/논문 URL (https://...)")
    print("- 유튜브 영상 URL (https://youtube.com/... 또는 https://youtu.be/...)")
    print("- 파일 경로 (PDF, DOCX, TXT 지원)")
    
    if previous_sources:
        print("\n📋 이전 소스 목록:")
        for i, src in enumerate(previous_sources[:5]):  # 처음 5개만 표시
            if isinstance(src, str):
                print(f"   {i+1}. {src[:60]}{'...' if len(src) > 60 else ''}")
            else:
                print(f"   {i+1}. 파일: {src.get('path', '알 수 없음')}")
        
        if len(previous_sources) > 5:
            print(f"   ... 외 {len(previous_sources)-5}개")
        
        print("\n이전 소스를 재사용하려면 'prev' 입력")
    
    sources = []
    while len(sources) < MAX_SOURCES:
        src_input = input(f"[{len(sources)+1}] 입력 (종료하려면 엔터): ").strip()
        
        if not src_input:
            break
            
        if src_input.lower() == 'prev' and previous_sources:
            print(f"✅ 이전 소스 {len(previous_sources)}개를 재사용합니다.")
            return previous_sources
        
        # 소스 입력 처리
        processed_source = process_source_input(src_input)
        if processed_source:
            sources.append(processed_source)
        # 오류는 process_source_input 내에서 출력
    
    if not sources:
        if previous_sources:
            print("⚠️ 소스가 입력되지 않았습니다. 이전 소스를 사용합니다.")
            return previous_sources
        else:
            print("⚠️ 소스가 입력되지 않았습니다. 계속하려면 적어도 하나의 소스가 필요합니다.")
            return get_sources_input(previous_config)  # 재귀적으로 다시 입력 받기
    
    return sources

# input_handler_updated.py 파일에 추가 또는 수정할 부분

def process_source_input(src_input: str) -> Optional[SourceType]:
    """
    소스 입력을 처리하고 유효성 검사
    
    Args:
        src_input: 사용자가 입력한 소스
        
    Returns:
        처리된 소스 또는 None (유효하지 않은 경우)
    """
    # URL 여부 확인
    if src_input.startswith(('http://', 'https://')):
        # URL 유효성 검사
        if not validators.url(src_input):
            print(f"❌ 유효하지 않은 URL 형식입니다: {src_input}")
            return None
            
        # YouTube URL 확인
        if is_youtube_url(src_input):
            print("✅ 유튜브 영상 URL이 감지되었습니다.")
        else:
            print("✅ 웹 URL이 감지되었습니다.")
        
        return src_input
    
    # 폴더 여부 확인 - 추가된 부분
    elif os.path.isdir(src_input):
        print(f"✅ 폴더가 감지되었습니다: {src_input}")
        # 이미지 파일 확장자
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
        
        # 폴더 내 이미지 파일 확인
        image_files = []
        for file in os.listdir(src_input):
            file_path = os.path.join(src_input, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)
        
        if not image_files:
            print(f"❌ 폴더에 지원되는 이미지 파일이 없습니다: {src_input}")
            return None
        
        print(f"✅ {len(image_files)}개의 이미지 파일이 감지되었습니다.")
        
        # OCR 엔진 선택 프롬프트
        print("\n🔍 사용할 OCR 엔진을 선택하세요:")
        print("1. Google Cloud Vision (기본값)")
        print("2. AWS Textract")
        print("3. Naver CLOVA OCR")
        print("4. Azure Document Intelligence (현재 사용 불가)")
        
        choice = input("> ").strip()
        
        # 선택에 따른 엔진 설정
        if choice == "2":
            engine = "aws"
        elif choice == "3":
            engine = "naver"
        elif choice == "4":
            print("⚠️ Azure는 현재 사용할 수 없습니다. Google Vision을 대신 사용합니다.")
            engine = "google"
        else:
            # 기본값 또는 잘못된 입력
            engine = "google"
        
        print(f"✅ 선택된 OCR 엔진: {engine}")
        
        # 폴더를 특별한 타입으로 반환
        return {"type": "image_folder", "path": src_input, "files": image_files, "ocr_engine": engine}
    
    # 파일 경로 확인 - 기존 코드
    elif os.path.exists(src_input):
        ext = os.path.splitext(src_input)[1].lower()
        if ext in SUPPORTED_FILE_TYPES:
            print(f"✅ {ext[1:].upper()} 파일이 감지되었습니다: {os.path.basename(src_input)}")
            
            # 이미지 파일인 경우 OCR 엔진 선택 프롬프트 추가
            if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                print("\n🔍 사용할 OCR 엔진을 선택하세요:")
                print("1. Google Cloud Vision (기본값)")
                print("2. AWS Textract")
                print("3. Naver CLOVA OCR")
                print("4. Azure Document Intelligence (현재 사용 불가)")
                
                choice = input("> ").strip()
                
                # 선택에 따른 엔진 설정
                if choice == "2":
                    engine = "aws"
                elif choice == "3":
                    engine = "naver"
                elif choice == "4":
                    print("⚠️ Azure는 현재 사용할 수 없습니다. Google Vision을 대신 사용합니다.")
                    engine = "google"
                else:
                    # 기본값 또는 잘못된 입력
                    engine = "google"
                
                print(f"✅ 선택된 OCR 엔진: {engine}")
                return {"type": ext[1:], "path": os.path.abspath(src_input), "ocr_engine": engine}
            
            return {"type": ext[1:], "path": os.path.abspath(src_input)}
        else:
            print(f"❌ 지원하지 않는 파일 형식입니다: {ext}")
            print(f"   지원 형식: {', '.join(SUPPORTED_FILE_TYPES)}")
            return None
    
    # 나머지 코드는 그대로 유지...
    
    # 상대 경로 시도
    else:
        # 현재 디렉토리 기준 상대 경로 확인
        relative_path = os.path.join(os.getcwd(), src_input)
        if os.path.exists(relative_path):
            ext = os.path.splitext(relative_path)[1].lower()
            if ext in SUPPORTED_FILE_TYPES:
                print(f"✅ {ext[1:].upper()} 파일이 감지되었습니다: {os.path.basename(relative_path)}")
                return {"type": ext[1:], "path": os.path.abspath(relative_path)}
        
        print("❌ 유효한 URL 또는 파일 경로를 입력해주세요.")
        return None

def get_structure_input(previous_config: Dict) -> str:
    """
    논리 구조 입력 받기
    
    Args:
        previous_config: 이전 설정 정보
        
    Returns:
        입력된 구조
    """
    previous_structure = previous_config.get("structure", "")
    
    structure_options = [
        "서론-본론-결론",
        "도입-전개-마무리",
        "문제-분석-해결책",
        "배경-현황-전망",
        "Introduction-Body-Conclusion"
    ]
    
    print("\n🧠 논리 흐름을 선택하거나 입력하세요:")
    for i, option in enumerate(structure_options):
        is_default = (not previous_structure and i == 0) or option == previous_structure
        print(f"   {i+1}. {option}{' (기본)' if is_default else ''}")
    
    print("   0. 직접 입력")
    
    choice = input("> ").strip()
    
    # 숫자 선택 처리
    if choice.isdigit():
        idx = int(choice)
        if idx == 0:
            print("직접 입력:")
            structure = input("> ").strip()
        elif 1 <= idx <= len(structure_options):
            structure = structure_options[idx-1]
        else:
            print(f"⚠️ 유효하지 않은 선택입니다. 기본값을 사용합니다.")
            structure = previous_structure if previous_structure else structure_options[0]
    # 직접 텍스트 입력
    else:
        structure = choice if choice else (previous_structure if previous_structure else structure_options[0])
    
    if not structure:
        structure = structure_options[0]
        print(f"기본 '{structure}' 구조로 설정됩니다.")
    else:
        print(f"✅ '{structure}' 구조로 설정됩니다.")
    
    return structure

def get_advanced_settings(previous_config: Dict) -> Dict:
    """
    고급 설정 입력 받기
    
    Args:
        previous_config: 이전 설정 정보
        
    Returns:
        고급 설정 딕셔너리
    """
    previous_advanced = {
        "voice": previous_config.get("voice", "Wyatt"),
        "parallel_workers": previous_config.get("parallel_workers", 3),
        "use_whisper": previous_config.get("use_whisper", True),
        "optimize_tts": previous_config.get("optimize_tts", True),
        "additional_instructions": previous_config.get("additional_instructions", ""),
        "content_types": previous_config.get("content_types", ["longform", "shortform1", "shortform2"])
    }
    
    print("\n⚙️ 고급 설정을 변경하시겠습니까? (y/n)")
    if input("> ").strip().lower() not in ['y', 'yes']:
        return previous_advanced
    
    advanced_settings = {}
    
    # 음성 설정
    voice_options = [
        ("Wyatt", "Wise Rustic Cowboy 목소리 (Wyatt)"),
        ("James", "Husky & Engaging 목소리 (James)"),
        ("Brian", "권위 있는 남성 목소리 (Brian)")
    ]
    
    print("\n🔊 TTS 음성 선택:")
    for i, (voice_id, desc) in enumerate(voice_options):
        is_default = voice_id == previous_advanced["voice"]
        print(f"   {i+1}. {desc}{' (현재)' if is_default else ''}")
    
    voice_choice = input("> ").strip()
    if voice_choice.isdigit() and 1 <= int(voice_choice) <= len(voice_options):
        advanced_settings["voice"] = voice_options[int(voice_choice)-1][0]
    else:
        advanced_settings["voice"] = previous_advanced["voice"]
        print(f"   기존 음성을 유지합니다: {advanced_settings['voice']}")
    
    # 콘텐츠 유형 선택 추가
    print("\n📋 생성할 콘텐츠 유형 선택:")
    print("   1. 롱폼 + 숏폼 2개 (기본)")
    print("   2. 롱폼만")
    print("   3. 숏폼 2개만")
    print("   4. 숏폼 3개만")
    print("   5. 롱폼 + 숏폼 1개")
    print("   6. 롱폼 + 숏폼 3개")
    print("   7. 숏폼 1개만")
    print("   8. 직접 선택")
    
    content_choice = input("> ").strip()
    
    content_type_presets = {
        "1": ["longform", "shortform1", "shortform2"],
        "2": ["longform"],
        "3": ["shortform1", "shortform2"],
        "4": ["shortform1", "shortform2", "shortform3"],
        "5": ["longform", "shortform1"],
        "6": ["longform", "shortform1", "shortform2", "shortform3"],
        "7": ["shortform1"]
    }
    
    if content_choice in content_type_presets:
        advanced_settings["content_types"] = content_type_presets[content_choice]
    elif content_choice == "8":
        selected_types = []
        
        print("\n생성할 콘텐츠 유형을 선택하세요 (각 항목에 y/n로 응답):")
        
        # 롱폼 선택 (최대 1개)
        print("롱폼 스크립트 생성? (y/n)")
        if input("> ").strip().lower() in ['y', 'yes']:
            selected_types.append("longform")
            
        # 숏폼 선택 (최대 3개)
        for i in range(1, 4):
            print(f"숏폼 #{i} 스크립트 생성? (y/n)")
            if input("> ").strip().lower() in ['y', 'yes']:
                selected_types.append(f"shortform{i}")
        
        if not selected_types:
            print("⚠️ 최소한 하나의 콘텐츠 유형을 선택해야 합니다. 기본값을 사용합니다.")
            advanced_settings["content_types"] = previous_advanced["content_types"]
        else:
            advanced_settings["content_types"] = selected_types
    else:
        advanced_settings["content_types"] = previous_advanced["content_types"]
        print(f"   기존 콘텐츠 유형을 유지합니다.")
    
    # 나머지 설정들...
    # 병렬 처리 워커 수
    print(f"\n🧮 병렬 처리 워커 수 (1-8, 현재: {previous_advanced['parallel_workers']}):")
    worker_input = input("> ").strip()
    if worker_input.isdigit() and 1 <= int(worker_input) <= 8:
        advanced_settings["parallel_workers"] = int(worker_input)
    else:
        advanced_settings["parallel_workers"] = previous_advanced["parallel_workers"]
        print(f"   기존 설정을 유지합니다: {advanced_settings['parallel_workers']}")
    
    # Whisper 사용 여부
    print(f"\n🎤 자막 생성에 Whisper 모델 사용 (현재: {'켜짐' if previous_advanced['use_whisper'] else '꺼짐'}):")
    print("   1. 켜기 (높은 정확도, 느림)")
    print("   2. 끄기 (낮은 정확도, 빠름)")
    whisper_choice = input("> ").strip()
    if whisper_choice == "1":
        advanced_settings["use_whisper"] = True
    elif whisper_choice == "2":
        advanced_settings["use_whisper"] = False
    else:
        advanced_settings["use_whisper"] = previous_advanced["use_whisper"]
        print(f"   기존 설정을 유지합니다: {'켜짐' if advanced_settings['use_whisper'] else '꺼짐'}")
    
    # TTS 최적화
    print(f"\n🔧 TTS 최적화 (현재: {'켜짐' if previous_advanced['optimize_tts'] else '꺼짐'}):")
    print("   1. 켜기 (높은 품질, 느림)")
    print("   2. 끄기 (낮은 품질, 빠름)")
    tts_choice = input("> ").strip()
    if tts_choice == "1":
        advanced_settings["optimize_tts"] = True
    elif tts_choice == "2":
        advanced_settings["optimize_tts"] = False
    else:
        advanced_settings["optimize_tts"] = previous_advanced["optimize_tts"]
        print(f"   기존 설정을 유지합니다: {'켜짐' if advanced_settings['optimize_tts'] else '꺼짐'}")
    
    # 추가 지시사항
    print("\n📝 추가 지시사항 (선택사항):")
    if previous_advanced["additional_instructions"]:
        print(f"   현재: {previous_advanced['additional_instructions'][:50]}...")
        print("   변경하려면 입력하세요. 유지하려면 엔터:")
    additional = input("> ").strip()
    if additional:
        advanced_settings["additional_instructions"] = additional
    else:
        advanced_settings["additional_instructions"] = previous_advanced["additional_instructions"]
    
    return advanced_settings

def show_input_summary(data: Dict) -> None:
    """
    입력 정보 요약 표시
    
    Args:
        data: 입력 데이터 딕셔너리
    """
    print("\n" + "="*50)
    print("✅ 입력 정보 요약:")
    print("="*50)
    print(f"- 주제: {data['topic']}")
    print(f"- 소스 수: {len(data['sources'])}개")
    print(f"- 구조: {data['structure']}")
    print(f"- 스타일: 국제관계/지정학/세계사 전문가")
    print(f"- TTS 음성: {data['voice']}")
    print(f"- 병렬 처리 워커 수: {data['parallel_workers']}")
    print(f"- Whisper 사용: {'예' if data['use_whisper'] else '아니오'}")
    print(f"- TTS 최적화: {'예' if data['optimize_tts'] else '아니오'}")
    
    # 콘텐츠 유형 출력
    content_types = data.get('content_types', ["longform", "shortform1", "shortform2"])
    content_type_desc = []
    if "longform" in content_types:
        content_type_desc.append("롱폼")
    shortform_count = sum(1 for ct in content_types if ct.startswith("shortform"))
    if shortform_count > 0:
        content_type_desc.append(f"숏폼 {shortform_count}개")
    print(f"- 생성할 콘텐츠: {' + '.join(content_type_desc)}")
    
    # 소스 목록
    if len(data['sources']) > 0:
        print("\n📋 소스 목록:")
        for i, src in enumerate(data['sources']):
            if isinstance(src, str):
                if is_youtube_url(src):
                    print(f"   {i+1}. YouTube: {src[:60]}{'...' if len(src) > 60 else ''}")
                else:
                    print(f"   {i+1}. URL: {src[:60]}{'...' if len(src) > 60 else ''}")
            else:
                print(f"   {i+1}. 파일: {os.path.basename(src.get('path', ''))} (타입: {src.get('type', '알 수 없음')})")
    
    print("\n계속하려면 Enter 키를 누르세요...")
    input()

def save_user_inputs(data: Dict, output_dir: str) -> str:
    """
    사용자 입력 데이터를 파일로 저장
    
    Args:
        data: 입력 데이터 딕셔너리
        output_dir: 저장할 디렉토리
        
    Returns:
        저장된 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "user_inputs.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"주제: {data['topic']}\n\n")
        f.write(f"구조: {data['structure']}\n\n")
        f.write(f"스타일: {data['style']}\n\n")
        f.write(f"TTS 음성: {data['voice']}\n\n")
        f.write(f"병렬 처리 워커 수: {data['parallel_workers']}\n\n")
        f.write(f"Whisper 사용: {'예' if data['use_whisper'] else '아니오'}\n\n")
        f.write(f"TTS 최적화: {'예' if data['optimize_tts'] else '아니오'}\n\n")
        
        if data.get('additional_instructions'):
            f.write(f"추가 지시사항:\n{data['additional_instructions']}\n\n")
        
        f.write("소스 목록:\n")
        for i, src in enumerate(data['sources']):
            if isinstance(src, str):
                if is_youtube_url(src):
                    f.write(f"{i+1}. YouTube: {src}\n")
                else:
                    f.write(f"{i+1}. URL: {src}\n")
            else:
                f.write(f"{i+1}. 파일: {src.get('path')} (타입: {src.get('type')})\n")
    
    return output_path

def is_youtube_url(url: str) -> bool:
    """
    URL이 유튜브 URL인지 확인
    
    Args:
        url: 확인할 URL
        
    Returns:
        유튜브 URL 여부
    """
    youtube_patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)[a-zA-Z0-9_-]+',
        r'youtube\.com\/embed\/[a-zA-Z0-9_-]+',
        r'youtube\.com\/shorts\/[a-zA-Z0-9_-]+'
    ]
    
    return any(re.search(pattern, url) for pattern in youtube_patterns)

def validate_file_path(file_path: str) -> Tuple[bool, str]:
    """
    파일 경로 유효성 검사
    
    Args:
        file_path: 검사할 파일 경로
        
    Returns:
        (유효 여부, 오류 메시지)
    """
    if not file_path:
        return False, "파일 경로가 비어 있습니다."
    
    if not os.path.exists(file_path):
        # 상대 경로 시도
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            return False, f"파일을 찾을 수 없습니다: {file_path}"
    
    if not os.path.isfile(file_path):
        return False, f"디렉토리가 아닌 파일이어야 합니다: {file_path}"
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_FILE_TYPES:
        return False, f"지원하지 않는 파일 형식입니다: {ext}"
    
    return True, ""

def validate_url(url: str) -> Tuple[bool, str]:
    """
    URL 유효성 검사
    
    Args:
        url: 검사할 URL
        
    Returns:
        (유효 여부, 오류 메시지)
    """
    if not url:
        return False, "URL이 비어 있습니다."
    
    if not url.startswith(('http://', 'https://')):
        return False, "URL은 http:// 또는 https://로 시작해야 합니다."
    
    # 기본 URL 구조 검사
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return False, "유효하지 않은 URL 구조입니다."
    except:
        return False, "URL 파싱 중 오류가 발생했습니다."
    
    # validators 라이브러리 사용 (설치된 경우)
    try:
        if not validators.url(url):
            return False, "유효하지 않은 URL 형식입니다."
    except:
        # validators 없이 기본 검사만
        if not re.match(r'^https?://[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,}(:[0-9]{1,5})?(\/.*)?$', url):
            return False, "URL 형식이 올바르지 않습니다."
    
    return True, ""

def create_config_file(config_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """
    기본 설정 파일 생성
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        성공 여부
    """
    default_config = {
        "topic": "",
        "structure": "서론-본론-결론",
        "style": "military_expert",
        "voice": "echo",
        "parallel_workers": 3,
        "use_whisper": True,
        "optimize_tts": True,
        "sources": []
    }
    
    try:
        # 디렉토리 생성
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 기본 설정 파일 생성: {config_path}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ 설정 파일 생성 실패: {str(e)}")
        return False

if __name__ == "__main__":
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-config":
            create_config_file()
            print("✅ 기본 설정 파일이 생성되었습니다.")
            sys.exit(0)
    
    # 테스트 코드
    user_data = get_user_input()
    print("\n✅ 입력 완료!\n")
    
    # 저장 테스트
    save_path = save_user_inputs(user_data, "output_test")
    print(f"입력 데이터 저장 완료: {save_path}")

