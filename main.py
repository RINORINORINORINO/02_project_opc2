print("스크립트 시작")
import os
import sys
print("모듈 임포트 완료")
import time
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import concurrent.futures
from pathlib import Path
import multiprocessing

# main.py 파일에 추가
from dotenv import load_dotenv
load_dotenv()

print("OpenAI API 키:", os.getenv("OPENAI_API_KEY")[:5] + "..." if os.getenv("OPENAI_API_KEY") else "없음")
print("ElevenLabs API 키:", os.getenv("ELEVENLABS_API_KEY")[:5] + "..." if os.getenv("ELEVENLABS_API_KEY") else "없음")

# 개선된 모듈들 임포트
from input_handler_updated import get_user_input, save_user_inputs
from source_parser_updated import parse_sources
from advanced_summarizer_updated import advanced_summarize_texts
from subtitle_generator import generate_srt, batch_generate_srt
from media_suggester_updated import generate_media_suggestions
# TTS 엔진 모두 임포트
from openai_tts_generator import generate_tts_openai, list_available_voices, get_audio_info
from tts_generator import generate_tts_elevenlabs, list_recommended_voices, resolve_voice_id



# 로깅 레벨을 환경 변수나 명령행 인자로 설정할 수 있게 함
log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# 전역 설정
DEFAULT_CONFIG_PATH = "config.json"
MAX_PARALLEL_WORKERS = min(multiprocessing.cpu_count(), 4)  # 최대 4개 제한

def check_dependencies() -> bool:
    """
    필요한 패키지가 설치되어 있는지 확인하고 결과를 보고합니다.
    
    Returns:
        bool: 모든 필수 패키지가 설치되어 있으면 True, 아니면 False
    """
    # 의존성 정의 (모듈명, 패키지명, 필수여부)
    dependencies = [
        # 핵심 의존성 (필수)
        ("requests", "requests", True),
        ("numpy", "numpy", True),
        ("PIL", "Pillow", True),
        ("cv2", "opencv-python", True),
        
        # OCR 관련 의존성 (선택)
        ("google.cloud.vision", "google-cloud-vision", False),
        ("boto3", "boto3", False),
        ("azure.ai.formrecognizer", "azure-ai-formrecognizer", False),
        
        # 문서 처리 관련 의존성 (선택)
        ("docx", "python-docx", False),
        ("fitz", "PyMuPDF", False),
        
        # 오디오 처리 관련 의존성 (선택)
        ("pydub", "pydub", False),
        ("whisper", "openai-whisper", False),
        
        # YouTube 관련 의존성 (선택)
        ("pytube", "pytube", False),
        ("youtube_transcript_api", "youtube-transcript-api", False),
    ]
    
    missing_required = []
    missing_optional = []
    
    # 각 의존성 확인
    for module_name, package_name, required in dependencies:
        try:
            __import__(module_name)
        except ImportError:
            if required:
                missing_required.append((module_name, package_name))
            else:
                missing_optional.append((module_name, package_name))
    
    # 결과 보고
    if not missing_required and not missing_optional:
        logger.info("✅ 모든 패키지가 설치되어 있습니다.")
        return True
    
    if missing_required:
        logger.error("❌ 다음 필수 패키지가 설치되지 않았습니다:")
        for module, package in missing_required:
            logger.error(f"  - {module} (설치: pip install {package})")
    
    if missing_optional:
        logger.warning("⚠️ 다음 선택적 패키지가 설치되지 않았습니다 (일부 기능이 제한될 수 있습니다):")
        for module, package in missing_optional:
            logger.warning(f"  - {module} (설치: pip install {package})")
    
    if missing_required:
        logger.info("\n모든 패키지를 한 번에 설치하려면: pip install -r requirements.txt")
        return False
    
    return True

def main(args: Optional[Dict[str, Any]] = None):
    print("메인 함수 시작!")
    """
    군사/국제정치 전문 영어 유튜브 콘텐츠 자동 생성 메인 함수
    
    Args:
        args: 명령행 인자 또는 직접 입력한 설정
    """
    start_time = time.time()
    
    print("\n" + "="*60)
    print("🎬 국제관계/지정학/세계사 전문 한국어 유튜브 콘텐츠 자동 생성 시작!")
    print("="*60 + "\n")
    
    # 의존성 확인
    if not check_dependencies():
        print("❌ 필수 패키지가 설치되지 않아 프로그램을 종료합니다.")
        print("📦 'pip install -r requirements.txt' 명령으로 필요한 패키지를 설치하세요.")
        sys.exit(1)
    
    # 1. 명령행 인자 처리 또는 사용자 입력 받기
    if args is None:
        # 명령행 인자 파싱
        args = parse_arguments()
        
        # 구성 파일 처리
        config_file = args.config or DEFAULT_CONFIG_PATH
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"✅ 구성 파일 로드: {config_file}")
                    
                    # 명령행에서 명시적으로 지정되지 않은 값만 구성 파일에서 가져옴
                    for key, value in config.items():
                        if not hasattr(args, key) or getattr(args, key) is None:
                            setattr(args, key, value)
            except json.JSONDecodeError:
                logger.error(f"❌ 구성 파일 형식 오류: {config_file}")
            except Exception as e:
                logger.error(f"❌ 구성 파일 로드 실패: {str(e)}")
        
        # 키 값이 충분하지 않으면 사용자 입력 받기
        if not hasattr(args, 'topic') or not args.topic or not hasattr(args, 'sources') or not args.sources:
            user_data = get_user_input(args.config or DEFAULT_CONFIG_PATH)
            
            # 사용자 입력 데이터를 args에 병합
            for key, value in user_data.items():
                setattr(args, key, value)
    
    # 작업 폴더 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = args.topic.replace(' ', '_').replace('/', '_')[:20]
    project_folder = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else f"output_{timestamp}_{safe_topic}"
    os.makedirs(project_folder, exist_ok=True)
    
    # 로그 파일 설정
    log_file = os.path.join(project_folder, "process.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"🚀 프로젝트 시작: {args.topic}")
    logger.info(f"📂 프로젝트 폴더: {os.path.abspath(project_folder)}")
    
    try:
        # 2. 소스 텍스트 파싱 (유튜브 포함)
        source_texts = parse_source_content(args.sources, project_folder, args.parallel_workers)
        
        if not source_texts:
            logger.error("❌ 모든 소스 파싱 실패. 최소한 하나의 유효한 소스가 필요합니다.")
            return
        
        # 3. 스크립트 생성 (롱폼 및 숏폼)
        script_paths = generate_script(
            source_texts, 
            args.topic, 
            args.structure, 
            project_folder,
            args.style,
            args.additional_instructions,
            args.content_types  # 콘텐츠 유형 전달
        )
        
        if not script_paths or "longform" not in script_paths:
            logger.error("❌ 스크립트 생성 실패. 프로세스를 중단합니다.")
            return
        
        # 병렬 처리를 위한 작업 정의
        tasks = []
        
        # 4. 미디어 제안 생성 (비동기 처리)
        media_task = ('media', generate_media_content, (script_paths.get("longform", ""), args.topic, project_folder))
        tasks.append(media_task)
        
        # 5. TTS 생성 (비동기 처리)
        tts_task = ('tts', generate_tts_content, (script_paths, args.voice, project_folder, args.optimize_tts, args.tts_engine))
        tasks.append(tts_task)
        
        # 병렬 처리 실행
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), MAX_PARALLEL_WORKERS)) as executor:
            # 작업 제출
            future_to_task = {executor.submit(task[1], *task[2]): task[0] for task in tasks}
            
            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    task_result = future.result()
                    results[task_name] = task_result
                    logger.info(f"✅ {task_name.upper()} 작업 완료")
                except Exception as e:
                    logger.error(f"❌ {task_name.upper()} 작업 실패: {str(e)}")
                    results[task_name] = None
        """
        # 6. 자막 생성 (TTS 결과에 의존)
        audio_paths = results.get('tts', {})
        if audio_paths:
            subtitle_paths = generate_subtitle_content(
                script_paths, 
                audio_paths, 
                project_folder, 
                args.use_whisper
            )
            results['subtitle'] = subtitle_paths
        """
        
        # 7. 프로젝트 요약 생성
        summary_path = generate_project_summary(
            args, 
            source_texts, 
            script_paths, 
            results, 
            project_folder, 
            start_time
        )
        
        # 완료 보고
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        
        print("\n" + "="*60)
        print("🎉 전체 작업 완료!")
        print("="*60)
        print(f"🕒 소요 시간: {minutes}분 {seconds}초")
        print(f"🗂️ 프로젝트 폴더: {os.path.abspath(project_folder)}")
        print("\n생성된 파일:")
        
        # 롱폼 결과
        if "longform" in script_paths:
            print(f"- 📜 롱폼 스크립트: {os.path.basename(script_paths['longform'])}")
        
        # 숏폼 결과
        for i in range(1, 3):
            shortform_key = f"shortform{i}"
            if shortform_key in script_paths:
                print(f"- 📋 숏폼 #{i} 스크립트: {os.path.basename(script_paths[shortform_key])}")
        
        # TTS 결과
        if 'tts' in results and results['tts']:
            for content_type, audio_path in results['tts'].items():
                print(f"- 🔊 {content_type} 오디오: {os.path.basename(audio_path)}")
        
        # 자막 결과
        if 'subtitle' in results and results['subtitle']:
            for content_type, subtitle_path in results['subtitle'].items():
                print(f"- 📃 {content_type} 자막: {os.path.basename(subtitle_path)}")
        
        # 미디어 제안
        if 'media' in results and results['media']:
            print(f"- 🎨 미디어 제안: {os.path.basename(results['media'])}")
        
        print(f"- 📋 프로젝트 요약: {os.path.basename(summary_path)}")
        print(f"\n📝 로그 파일: {os.path.basename(log_file)}")
    
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자가 프로세스를 중단했습니다.")
        print("\n⚠️ 프로세스가 중단되었습니다. 부분적으로 생성된 파일은 유지됩니다.")
    except FileNotFoundError as e:
        logger.error(f"❌ 파일을 찾을 수 없음: {str(e)}", exc_info=True)
        print(f"\n❌ 파일을 찾을 수 없습니다: {str(e)}")
    except PermissionError as e:
        logger.error(f"❌ 파일 접근 권한 오류: {str(e)}", exc_info=True)
        print(f"\n❌ 파일 접근 권한이 없습니다: {str(e)}")
    except (requests.RequestException, ConnectionError) as e:
        logger.error(f"❌ 네트워크 오류: {str(e)}", exc_info=True)
        print(f"\n❌ 네트워크 연결 문제가 발생했습니다: {str(e)}")
    except Exception as e:
        logger.error(f"❌ 실행 중 오류 발생: {str(e)}", exc_info=True)
        print(f"\n❌ 오류 발생: {str(e)}")
        print(f"자세한 내용은 로그 파일을 확인하세요: {log_file}")
    
    finally:
        # 로그 핸들러 닫기
        file_handler.close()
        logger.removeHandler(file_handler)

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='국제관계/지정학/세계사 전문 한국어 유튜브 콘텐츠 자동 생성')
    
    # 필수 입력
    parser.add_argument('--topic', type=str, help='콘텐츠 주제')
    parser.add_argument('--sources', type=str, nargs='+', help='소스 URL 또는 파일 경로 목록')
    
    # 선택적 입력
    parser.add_argument('--structure', type=str, default='서론-본론-결론', 
                      help='콘텐츠 구조 (기본값: 서론-본론-결론)')
    parser.add_argument('--style', type=str, default='international_relations_expert',
                      help='생성 스타일 (기본값: international_relations_expert)')
    parser.add_argument('--voice', type=str, default='Wyatt',
                      help='TTS 음성 (기본값: Wyatt)')
    parser.add_argument('--config', type=str, help='구성 파일 경로')
    parser.add_argument('--output-dir', type=str, help='출력 디렉토리')
    parser.add_argument('--parallel-workers', type=int, default=3,
                      help='병렬 처리 워커 수 (기본값: 3)')
    parser.add_argument('--use-whisper', action='store_true', default=True,
                      help='자막 생성에 Whisper 모델 사용')
    parser.add_argument('--optimize-tts', action='store_true', default=True,
                      help='TTS 최적화 사용')
    parser.add_argument('--additional-instructions', type=str, default='',
                      help='스크립트 생성을 위한 추가 지시사항')
    parser.add_argument('--content-types', type=str, nargs='+', 
                      default=['longform', 'shortform1', 'shortform2'],
                      help='생성할 콘텐츠 유형 (예: longform shortform1)')
    parser.add_argument('--tts-engine', type=str, default='elevenlabs',
                  choices=['elevenlabs', 'openai'],
                  help='TTS 엔진 선택 (elevenlabs/openai, 기본값: elevenlabs)')
    
    return parser.parse_args()

def parse_source_content(sources: List[Any], project_folder: str, parallel_workers: int = 3) -> List[str]:
    """
    소스 텍스트 파싱 (URL, 파일, YouTube 등)
    
    Args:
        sources: 소스 목록 (URL 또는 파일 경로)
        project_folder: 프로젝트 폴더 경로
        parallel_workers: 병렬 처리 워커 수
        
    Returns:
        파싱된 텍스트 리스트
    """
    logger.info(f"📡 {len(sources)}개 소스 파싱 시작")
    
    # 소스 폴더 생성
    sources_dir = os.path.join(project_folder, "sources")
    os.makedirs(sources_dir, exist_ok=True)
    
    # 소스 파싱
    parsed_texts = parse_sources(sources, max_workers=parallel_workers)
    
    # 유효성 검사
    valid_texts = [text for text in parsed_texts if text and len(text.strip()) > 100]
    
    if not valid_texts:
        logger.error("❌ 유효한 텍스트가 파싱되지 않았습니다.")
        if parsed_texts:
            logger.warning(f"⚠️ 파싱된 텍스트가 있지만 너무 짧거나 비어 있습니다. ({len(parsed_texts)}개)")
        return []
    
    # 파싱된 텍스트 저장
    for i, text in enumerate(valid_texts):
        source_path = os.path.join(sources_dir, f"source_{i+1}.txt")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(text)
    
    logger.info(f"✅ {len(valid_texts)}/{len(sources)}개 소스 파싱 완료")
    return valid_texts

def generate_script(
    source_texts: List[str], 
    topic: str, 
    structure: str, 
    project_folder: str,
    style: str = "international_relations_expert",
    additional_instructions: str = "",
    content_types: List[str] = ["longform", "shortform1", "shortform2"]
) -> Dict[str, str]:
    """
    소스 텍스트를 분석하여 롱폼 및 숏폼 스크립트 생성
    
    Args:
        source_texts: 파싱된 소스 텍스트 리스트
        topic: 콘텐츠 주제
        structure: 논리 구조
        project_folder: 프로젝트 폴더 경로
        style: 생성 스타일
        additional_instructions: 추가 지시사항
        content_types: 생성할 콘텐츠 유형 리스트
        
    Returns:
        생성된 스크립트 파일 경로 딕셔너리
    """
    logger.info(f"✍️ 스크립트 생성 시작: 주제 '{topic}', 구조 '{structure}'")
    
    # 한국어 스크립트 생성을 위한 추가 지시사항
    korean_instruction = """
    이 스크립트는 한국어권 시청자를 위한 콘텐츠입니다. 다음 사항에 특별히 주의하세요:
    
    1. 모든 내용은 자연스러운 한국어로 작성되어야 합니다. 외국어 표현이 필요한 경우 적절히 한글로 표기하고 괄호 안에 원어를 병기하세요.
    2. 한국인 시청자들이 관심을 가질만한 관점과 사례를 포함하세요. 가능하면 한국과 관련된 맥락도 언급하세요.
    3. 국제관계/지정학 용어는 한국에서 통용되는 번역어를 사용하되, 필요 시 원어 병기를 활용하세요.
    4. 전문성을 유지하면서도 대중적으로 이해하기 쉬운 표현을 사용하세요.
    5. 롱폼의 경우 2700-3300자, 숏폼의 경우 250-400자 정도로 작성하세요.
    """
    
    # 사용자 지정 추가 지시사항 병합
    if additional_instructions:
        korean_instruction += f"\n\n추가 지시사항:\n{additional_instructions}"
    
    # 분석 결과 저장 디렉토리
    analysis_dir = os.path.join(project_folder, "analysis")
    
    try:
        # 향상된 스크립트 생성 함수 호출
        scripts = advanced_summarize_texts(
            source_texts, 
            topic, 
            structure, 
            style=style,
            output_dir=analysis_dir,
            additional_instructions=korean_instruction,
            content_types=content_types
        )
        
        if not scripts and "longform" in content_types:
            logger.error("❌ 스크립트 생성 실패")
            return {}
        
        logger.info("✅ 스크립트 생성 완료")
        
        # 스크립트 저장
        script_paths = {}

        # 롱폼 스크립트 저장
        if "longform" in content_types and scripts.get("longform"):
            longform_path = os.path.join(project_folder, "final_longform_script.txt")
            with open(longform_path, "w", encoding="utf-8") as f:
                f.write(scripts["longform"])
            script_paths["longform"] = longform_path

        # 숏폼 스크립트 저장
        for i in range(1, 4):  # 최대 3개의 숏폼 지원
            shortform_key = f"shortform{i}"
            if shortform_key in content_types and scripts.get(shortform_key):
                shortform_path = os.path.join(project_folder, f"final_shortform{i}_script.txt")
                with open(shortform_path, "w", encoding="utf-8") as f:
                    f.write(scripts[shortform_key])
                script_paths[shortform_key] = shortform_path
            
        logger.info(f"✅ 스크립트 저장 완료: {len(script_paths)}개 파일")
        return script_paths
        
    except Exception as e:
        logger.error(f"❌ 스크립트 생성 중 오류: {str(e)}")
        return {}

def generate_media_content(script: str, topic: str, project_folder: str) -> Optional[str]:
    """
    미디어 제안 생성
    
    Args:
        script: 생성된 스크립트
        topic: 콘텐츠 주제
        project_folder: 프로젝트 폴더 경로
        
    Returns:
        생성된 미디어 제안 파일 경로 또는 None
    """
    logger.info("🎨 국제관계/지정학/세계사 전문 미디어 제안 생성 시작")
    
    # 미디어 제안 저장 디렉토리
    media_dir = os.path.join(project_folder, "media")
    os.makedirs(media_dir, exist_ok=True)
    
    try:
        # 미디어 제안 생성
        media_suggestions = generate_media_suggestions(
            script, 
            topic,
            output_dir=media_dir,
            use_cache=True
        )
        
        if not media_suggestions:
            logger.error("❌ 미디어 제안 생성 실패")
            return None
        
        # 미디어 제안 저장
        suggestions_path = os.path.join(project_folder, "media_suggestions.txt")
        with open(suggestions_path, "w", encoding="utf-8") as f:
            f.write(media_suggestions)
            
        logger.info(f"✅ 미디어 제안 생성 완료: {suggestions_path}")
        return suggestions_path
        
    except Exception as e:
        logger.error(f"❌ 미디어 제안 생성 중 오류: {str(e)}")
        return None

def generate_tts_content(
    scripts: Dict[str, str], 
    voice_id: str, 
    project_folder: str,
    optimize: bool = True,
    tts_engine: str = 'elevenlabs'  # 인자 추가
) -> Dict[str, str]:
    """
    스크립트 딕셔너리에 대해 TTS 오디오 생성
    
    Args:
        scripts: 스크립트 파일 경로 딕셔너리
        voice_id: 음성 ID
        project_folder: 프로젝트 폴더 경로
        optimize: 품질 최적화 여부
        
    Returns:
        생성된 오디오 파일 경로 딕셔너리
    """
    logger.info(f"🔊 TTS 생성 시작 (음성: {voice_id})")
    
    # TTS 저장 디렉토리
    audio_dir = os.path.join(project_folder, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_paths = {}
    
    try:
        # 각 스크립트에 대한 TTS 생성
        for script_type, script_path in scripts.items():
            if not os.path.exists(script_path):
                logger.warning(f"⚠️ 스크립트 파일이 존재하지 않음: {script_path}")
                continue
            
            # 스크립트 읽기
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            if not script_content:
                logger.warning(f"⚠️ 스크립트 내용이 비어 있음: {script_path}")
                continue
            
            # TTS 파일명 접두사 설정
            prefix = f"{script_type}_speech"
            
            logger.info(f"🎤 '{script_type}' 스크립트 TTS 생성 중...")
            
            # TTS API 선택 (원하는 것만 주석 해제)
            # -------------------------
            # 1. Eleven Labs TTS 사용 (고품질, 높은 API 비용)
            # -------------------------
            # 사용자가 선택할 수 있게 args에서 tts_engine 파라미터를 추가하고
            # 그 값에 따라 엔진 선택
            use_elevenlabs = tts_engine.lower() == 'elevenlabs'
            
            if use_elevenlabs:
                actual_voice_id = resolve_voice_id(voice_id)

            
                # 최적화 설정
                stability = 0.4 if optimize else 0.3
                similarity_boost = 0.75 if optimize else 0.65
                
                # TTS 생성
                audio_path = generate_tts_elevenlabs(
                    script_content,
                    voice_id=actual_voice_id,
                    output_dir=audio_dir,
                    filename_prefix=prefix,
                    model_id="eleven_multilingual_v2",
                    stability=stability,
                    similarity_boost=similarity_boost,
                    style=0.15,
                    use_parallel=True
                )
            
            # -------------------------
            # 2. OpenAI TTS 사용 (중간 품질, 낮은 API 비용)
            # -------------------------
            else:
                # 음성 ID 해석 (OpenAI용)
                voice_id_map = {
                    'echo': "echo",
                    'james': "onyx",  # 남성 음성으로 매핑
                    'adam': "onyx",
                    'sam': "onyx",
                    'rachel': "nova"  # 여성 음성으로 매핑
                }
                
                actual_voice_id = voice_id_map.get(voice_id.lower(), voice_id)
                
                # 최적화 설정
                model_id = "tts-1-hd" if optimize else "tts-1"
                speed = 1.0  # 기본 속도
                
                # TTS 생성
                audio_path = generate_tts_openai(
                    script_content,
                    voice_id=actual_voice_id,
                    output_dir=audio_dir,
                    filename_prefix=prefix,
                    model_id=model_id,
                    speed=speed,
                    use_parallel=True
                )
            
            if audio_path:
                audio_paths[script_type] = audio_path
                
                # 오디오 정보 로깅
                audio_info = get_audio_info(audio_path)
                duration = audio_info.get('duration')
                if duration:
                    minutes, seconds = divmod(int(duration), 60)
                    logger.info(f"✅ '{script_type}' TTS 생성 완료: {os.path.basename(audio_path)} (길이: {minutes}분 {seconds}초)")
                else:
                    logger.info(f"✅ '{script_type}' TTS 생성 완료: {os.path.basename(audio_path)}")
            else:
                logger.error(f"❌ '{script_type}' TTS 생성 실패")
        
        return audio_paths
            
    except Exception as e:
        logger.error(f"❌ TTS 생성 중 오류: {str(e)}")
        return audio_paths

def generate_subtitle_content(
    scripts: Dict[str, str],
    audio_paths: Dict[str, str],
    project_folder: str,
    use_whisper: bool = True
) -> Dict[str, str]:
    """
    여러 오디오 파일에 대한 자막 파일 생성
    
    Args:
        scripts: 스크립트 파일 경로 딕셔너리
        audio_paths: 오디오 파일 경로 딕셔너리
        project_folder: 프로젝트 폴더 경로
        use_whisper: Whisper 모델 사용 여부
        
    Returns:
        생성된 자막 파일 경로 딕셔너리
    """
    logger.info(f"📃 자막 생성 시작 (Whisper: {'사용' if use_whisper else '미사용'})")
    
    # 자막 저장 디렉토리
    subtitle_dir = os.path.join(project_folder, "subtitles")
    os.makedirs(subtitle_dir, exist_ok=True)
    
    subtitle_paths = {}
    
    try:
        # 각 오디오 파일에 대한 자막 생성
        for content_type, audio_path in audio_paths.items():
            if not os.path.exists(audio_path):
                logger.warning(f"⚠️ 오디오 파일이 존재하지 않음: {audio_path}")
                continue
            
            # 스크립트 로드
            script_content = ""
            if content_type in scripts and os.path.exists(scripts[content_type]):
                with open(scripts[content_type], 'r', encoding='utf-8') as f:
                    script_content = f.read()
            
            if not script_content:
                logger.warning(f"⚠️ '{content_type}' 스크립트 내용이 비어 있습니다")
                
            logger.info(f"🎤 '{content_type}' 자막 생성 중...")
            
            # 자막 생성
            subtitle_path = generate_srt(
                script_content,
                audio_path,
                output_dir=subtitle_dir,
                use_whisper=use_whisper,
                max_chars_per_subtitle=35 if 'shortform' in content_type else 42  # 숏폼은 더 짧은 자막 가능
            )
            
            if subtitle_path:
                subtitle_paths[content_type] = subtitle_path
                logger.info(f"✅ '{content_type}' 자막 생성 완료: {os.path.basename(subtitle_path)}")
            else:
                logger.error(f"❌ '{content_type}' 자막 생성 실패")
        
        return subtitle_paths
        
    except Exception as e:
        logger.error(f"❌ 자막 생성 중 오류: {str(e)}")
        return subtitle_paths

def generate_project_summary(
    args: Any, 
    source_texts: List[str], 
    script_paths: Dict[str, str], 
    results: Dict[str, Any], 
    project_folder: str,
    start_time: float
) -> str:
    """
    프로젝트 요약 파일 생성
    
    Args:
        args: 입력 인자
        source_texts: 파싱된 소스 텍스트 리스트
        script_paths: 스크립트 파일 경로 딕셔너리
        results: 작업 결과 딕셔너리
        project_folder: 프로젝트 폴더 경로
        start_time: 시작 시간 타임스탬프
        
    Returns:
        생성된 요약 파일 경로
    """
    logger.info("📋 프로젝트 요약 생성 중...")
    
    # 현재 시간
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 소요 시간 계산
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(int(elapsed_time), 60)
    
    # 스크립트 정보
    script_info = {}
    longform_path = script_paths.get("longform", "")
    
    if longform_path and os.path.exists(longform_path):
        with open(longform_path, 'r', encoding='utf-8') as f:
            script_content = f.read()
            
            # 단어 수 계산
            words = script_content.split()
            script_info['word_count'] = len(words)
            
            # 예상 재생 시간 (평균 150단어/분 가정)
            estimated_duration = len(words) / 150
            script_info['estimated_duration'] = estimated_duration
    
    # 오디오 정보
    audio_info = {}
    audio_paths = results.get('tts', {})
    
    if "longform" in audio_paths and os.path.exists(audio_paths["longform"]):
        audio_data = get_audio_info(audio_paths["longform"])
        if audio_data:
            audio_info = audio_data
    
    # 요약 파일 경로
    summary_path = os.path.join(project_folder, "project_summary.txt")
    
    # 요약 작성
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(f"# 프로젝트 입력 정보 요약\n\n")
        f.write(f"## 기본 정보\n")
        f.write(f"- 주제: {args.topic}\n")
        f.write(f"- 생성 날짜: {current_time}\n")
        f.write(f"- 논리 구조: {args.structure}\n")
        f.write(f"- 스타일: 국제관계/지정학/세계사 전문가\n")
        f.write(f"- TTS 음성: {args.voice}\n")
        f.write(f"- 병렬 처리 워커 수: {args.parallel_workers}\n")
        f.write(f"- Whisper 사용 여부: {'예' if args.use_whisper else '아니오'}\n")
        f.write(f"- TTS 최적화 여부: {'예' if args.optimize_tts else '아니오'}\n\n")
        
        f.write(f"## 소스 정보 ({len(source_texts)}개)\n")
        sources = args.sources if hasattr(args, 'sources') else []
        for i, src in enumerate(sources[:len(source_texts)]):
            if isinstance(src, str):
                f.write(f"{i+1}. URL: {src[:120]}{'...' if len(src) > 120 else ''}\n")
            else:
                path = src.get('path', 'Unknown')
                f.write(f"{i+1}. 파일: {path} (타입: {src.get('type', '알 수 없음')})\n")
        f.write("\n")
        
        f.write(f"## 생성된 파일\n")
        
        # 스크립트 파일
        for script_type, path in script_paths.items():
            f.write(f"- {script_type} 스크립트: {os.path.basename(path)}\n")
        
        # 오디오 파일
        audio_paths = results.get('tts', {})
        for audio_type, path in audio_paths.items():
            f.write(f"- {audio_type} 오디오: {os.path.basename(path)}\n")
        
        # 자막 파일
        subtitle_paths = results.get('subtitle', {})
        for subtitle_type, path in subtitle_paths.items():
            f.write(f"- {subtitle_type} 자막: {os.path.basename(path)}\n")
        
        # 미디어 제안
        if 'media' in results and results['media']:
            f.write(f"- 미디어 제안: {os.path.basename(results['media'])}\n")
        
        f.write("\n")
        
        f.write(f"## 추가 정보\n")
        if script_info:
            f.write(f"- 롱폼 스크립트 길이: {script_info.get('word_count', 0)}단어\n")
            estimated_mins = int(script_info.get('estimated_duration', 0))
            f.write(f"- 예상 재생 시간: 약 {estimated_mins}분\n")
        
        if audio_info:
            duration = audio_info.get('duration')
            if duration:
                audio_mins, audio_secs = divmod(int(duration), 60)
                f.write(f"- 실제 롱폼 오디오 길이: {audio_mins}분 {audio_secs}초\n")
            
            file_size = audio_info.get('file_size')
            if file_size:
                size_mb = file_size / (1024 * 1024)
                f.write(f"- 롱폼 오디오 파일 크기: {size_mb:.2f} MB\n")
        
        f.write(f"- 처리 시간: {minutes}분 {seconds}초\n")
        f.write(f"- 작업 디렉토리: {os.path.abspath(project_folder)}\n")
    
    logger.info(f"✅ 프로젝트 요약 생성 완료: {summary_path}")
    return summary_path

if __name__ == "__main__":
    main()