import os
import requests
import time
import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
from functools import lru_cache
import concurrent.futures
import io
from pathlib import Path
import threading
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 환경변수에서 API 키 로드
load_dotenv()
api_key = os.getenv("ELEVENLABS_API_KEY")

# 전역 설정
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1  # 초 단위
MAX_CHUNK_SIZE = 4000  # 최대 청크 크기 (문자)
MAX_WORKERS = 3  # 병렬 처리 워커 수

# TTS API 호출 세마포어 추가 - 음성 생성은 무거운 작업이므로 제한 강화
tts_semaphore = threading.Semaphore(2)  # 최대 2개 동시 TTS 요청


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
    with tts_semaphore:
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

def resolve_voice_id(voice_id: str) -> str:
    """음성 이름을 UUID로 변환하거나 UUID를 그대로 반환"""
    voice_id_map = {
        'wyatt': "YXpFCvM1S3JbWEJhoskW",
        'james': "EkK5I93UQWFDigLMpZcX",
        'brian': "nPczCjzI2devNBz1zQrb",
    }
    user_input_voice = voice_id.lower()
    return voice_id_map.get(user_input_voice, voice_id)

def generate_tts_elevenlabs(
    script: str, 
    voice_id: str = "YXpFCvM1S3JbWEJhoskW", 
    output_dir: str = "output_audio", 
    max_chunk_size: int = MAX_CHUNK_SIZE,
    filename_prefix: str = "speech",
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.4,
    similarity_boost: float = 0.75,
    style: float = 0.15,
    use_parallel: bool = True,
    optimize_streaming_latency: Optional[int] = None
) -> str:
    """
    스크립트를 Eleven Labs TTS로 변환하여 MP3 파일로 저장하고 경로를 반환합니다.
    긴 스크립트의 경우 청크로 나누어 병렬 처리하고 결합합니다.
    
    Args:
        script: 음성으로 변환할 텍스트
        voice_id: Eleven Labs 음성 ID (기본값: Adam - 자연스러운 남성 영어 음성)
        output_dir: 오디오 파일 저장 디렉토리
        max_chunk_size: 각 청크의 최대 문자 수
        filename_prefix: 생성된 오디오 파일의 접두사
        model_id: Eleven Labs 모델 ID
        stability: 음성 안정성 (0.0~1.0)
        similarity_boost: 원본 음성과의 유사도 향상 정도 (0.0~1.0)
        style: 스타일 강도 (0.0~1.0)
        use_parallel: 병렬 처리 사용 여부
        optimize_streaming_latency: 스트리밍 지연 최적화 (0~4, None=사용안함)
        
    Returns:
        저장된 오디오 파일 경로
    """
    if not api_key:
        logger.error("❌ Eleven Labs API 키가 설정되지 않았습니다. .env 파일에 ELEVENLABS_API_KEY를 설정하세요.")
        return ""
    
    try:
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)

        # 음성 요소만 추출 (영상 지시사항 제외)
        speech_script = extract_speech_parts(script)
        
        # TTS를 위한 스크립트 전처리
        processed_script = process_script_for_tts(speech_script)
        
        # 타임스탬프 파일명 생성
        timestamp = int(time.time())
        base_filename = f"{filename_prefix}_{timestamp}"
        output_path = os.path.join(output_dir, f"{base_filename}.mp3")
        
        # 스크립트 청크 분리
        chunks = split_script_into_chunks(processed_script, max_chunk_size)
        total_chunks = len(chunks)
        
        logger.info(f"🔊 Eleven Labs TTS 생성 시작 (음성: {get_voice_name(voice_id)}, 청크: {total_chunks}개)")
        
        if total_chunks == 1:
            # 단일 청크 처리
            try:
                logger.info(f"🎤 Eleven Labs TTS 음성 생성 중...")
                audio_data = generate_single_audio_chunk(chunks[0], voice_id, model_id, stability, similarity_boost, style, optimize_streaming_latency)
                
                if audio_data:
                    with open(output_path, "wb") as f:
                        f.write(audio_data)
                    logger.info(f"✅ 음성 생성 완료: {output_path}")
                    return output_path
                else:
                    logger.error("❌ Eleven Labs TTS 생성 실패")
                    return ""
            except Exception as e:
                logger.error(f"❌ Eleven Labs TTS 생성 실패: {e}")
                return ""
        else:
            # 다중 청크 처리
            logger.info(f"📊 스크립트가 {total_chunks}개 청크로 분할되었습니다.")
            
            if use_parallel and total_chunks > 1:
                # 병렬 처리
                chunk_paths = generate_audio_chunks_parallel(
                    chunks, voice_id, model_id, stability, similarity_boost, style,
                    optimize_streaming_latency, base_filename, output_dir
                )
            else:
                # 순차 처리
                chunk_paths = generate_audio_chunks_sequential(
                    chunks, voice_id, model_id, stability, similarity_boost, style,
                    optimize_streaming_latency, base_filename, output_dir
                )
            
            if not chunk_paths:
                logger.error("❌ 모든 청크 처리 실패")
                return ""
            
            # 오디오 청크 결합
            try:
                combined_path = combine_audio_chunks(chunk_paths, output_path)
                if combined_path:
                    logger.info(f"✅ 전체 음성 파일 결합 완료: {combined_path}")
                    return combined_path
                else:
                    # 결합 실패시 첫 번째 청크 반환
                    logger.warning("⚠️ 청크 결합 실패, 첫 번째 청크만 반환합니다.")
                    return chunk_paths[0]
            except Exception as e:
                logger.error(f"❌ 오디오 청크 결합 실패: {e}")
                return chunk_paths[0] if chunk_paths else ""
    
    except Exception as e:
        logger.error(f"❌ TTS 생성 중 예외 발생: {str(e)}")
        return ""

def generate_single_audio_chunk(
    text: str, 
    voice_id: str,
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.4,
    similarity_boost: float = 0.75,
    style: float = 0.15,
    optimize_streaming_latency: Optional[int] = None
) -> Optional[bytes]:
    """
    단일 텍스트 청크를 오디오로 변환
    
    Args:
        text: 변환할 텍스트
        voice_id: Eleven Labs 음성 ID
        model_id: 사용할 모델 ID
        stability: 음성 안정성 (0.0~1.0)
        similarity_boost: 원본 음성과의 유사도 향상 정도 (0.0~1.0)
        style: 스타일 강도 (0.0~1.0)
        optimize_streaming_latency: 스트리밍 지연 최적화 (0~4, None=사용안함)
        
    Returns:
        오디오 데이터 바이트 또는 None
    """
    def make_tts_request():
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": True
            }
        }
        
        # 스트리밍 지연 최적화 설정 추가 (기능 사용 시)
        if optimize_streaming_latency is not None:
            data["optimize_streaming_latency"] = optimize_streaming_latency
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.content
        else:
            error_msg = f"Eleven Labs API 오류 ({response.status_code}): "
            try:
                error_data = response.json()
                error_msg += json.dumps(error_data)
            except:
                error_msg += response.text
            
            logger.error(error_msg)
            raise Exception(error_msg)
    
    # 재시도 로직으로 API 호출
    return api_call_with_retry(make_tts_request)

def generate_audio_chunks_parallel(
    chunks: List[str],
    voice_id: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
    style: float,
    optimize_streaming_latency: Optional[int],
    base_filename: str,
    output_dir: str
) -> List[str]:
    """
    텍스트 청크 리스트를 병렬로 오디오로 변환
    
    Args:
        chunks: 텍스트 청크 리스트
        voice_id: Eleven Labs 음성 ID
        model_id: 사용할 모델 ID
        stability: 음성 안정성
        similarity_boost: 원본 음성과의 유사도
        style: 스타일 강도
        optimize_streaming_latency: 스트리밍 지연 최적화
        base_filename: 기본 파일 이름
        output_dir: 출력 디렉토리
        
    Returns:
        생성된 오디오 파일 경로 리스트
    """
    chunk_paths = []
    total_chunks = len(chunks)
    logger.info(f"🔄 병렬 처리로 {total_chunks}개 청크 생성 중... (최대 {MAX_WORKERS}개 작업자)")
    
    # 병렬 처리 함수
    def process_chunk(chunk_data):
        idx, chunk_text = chunk_data
        chunk_filename = f"{base_filename}_part{idx+1}.mp3"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        try:
            logger.info(f"🎤 청크 {idx+1}/{total_chunks} 생성 중 ({len(chunk_text)} 문자)")
            audio_data = generate_single_audio_chunk(
                chunk_text, voice_id, model_id, stability, 
                similarity_boost, style, optimize_streaming_latency
            )
            
            if audio_data:
                with open(chunk_path, "wb") as f:
                    f.write(audio_data)
                logger.info(f"✅ 청크 {idx+1}/{total_chunks} 생성 완료")
                return idx, chunk_path, True
            else:
                logger.error(f"❌ 청크 {idx+1}/{total_chunks} 생성 실패")
                return idx, "", False
        except Exception as e:
            logger.error(f"❌ 청크 {idx+1}/{total_chunks} 생성 중 오류: {str(e)}")
            return idx, "", False
    
    # 병렬 처리 실행
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, total_chunks)) as executor:
        # 일괄 제출 대신 하나씩 제출하고 완료될 때마다 다음 작업 제출
        future_to_idx = {}
        remaining_items = list(enumerate(chunks))
        
        # 첫 번째 배치 제출 (워커 수만큼)
        worker_count = min(MAX_WORKERS, total_chunks)
        initial_batch = remaining_items[:worker_count]
        remaining_items = remaining_items[worker_count:]
        
        for item in initial_batch:
            future = executor.submit(process_chunk, item)
            future_to_idx[future] = item[0]
        
        # 완료된 작업 처리 및 새 작업 제출
        while future_to_idx:
            # 완료된 작업 하나 가져오기
            done, _ = concurrent.futures.wait(
                future_to_idx, 
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                try:
                    idx, path, success = future.result()
                    if success and path:
                        results.append((idx, path))
                    
                    # 새 작업 제출 (남은 항목이 있는 경우)
                    if remaining_items:
                        new_item = remaining_items.pop(0)
                        new_future = executor.submit(process_chunk, new_item)
                        future_to_idx[new_future] = new_item[0]
                    
                except Exception as e:
                    logger.error(f"❌ 청크 처리 중 예외 발생: {str(e)}")
                
                # 처리된 future 삭제
                del future_to_idx[future]
    
    # 결과 정렬 (원래 순서대로)
    results.sort(key=lambda x: x[0])
    
    # 성공한 경로만 필터링
    chunk_paths = [res[1] for res in results if res[2]]
    
    success_count = len(chunk_paths)
    logger.info(f"🏁 청크 생성 완료: 성공 {success_count}개, 실패 {total_chunks - success_count}개")
    
    return chunk_paths

def generate_audio_chunks_sequential(
    chunks: List[str],
    voice_id: str,
    model_id: str,
    stability: float,
    similarity_boost: float,
    style: float,
    optimize_streaming_latency: Optional[int],
    base_filename: str,
    output_dir: str
) -> List[str]:
    """
    텍스트 청크 리스트를 순차적으로 오디오로 변환
    
    Args:
        chunks: 텍스트 청크 리스트
        voice_id: Eleven Labs 음성 ID
        model_id: 사용할 모델 ID
        stability: 음성 안정성
        similarity_boost: 원본 음성과의 유사도
        style: 스타일 강도
        optimize_streaming_latency: 스트리밍 지연 최적화
        base_filename: 기본 파일 이름
        output_dir: 출력 디렉토리
        
    Returns:
        생성된 오디오 파일 경로 리스트
    """
    chunk_paths = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base_filename}_part{i+1}.mp3"
        chunk_path = os.path.join(output_dir, chunk_filename)
        
        try:
            logger.info(f"🎤 청크 {i+1}/{total_chunks} 생성 중 ({len(chunk)} 문자)")
            audio_data = generate_single_audio_chunk(
                chunk, voice_id, model_id, stability, 
                similarity_boost, style, optimize_streaming_latency
            )
            
            if audio_data:
                with open(chunk_path, "wb") as f:
                    f.write(audio_data)
                
                chunk_paths.append(chunk_path)
                logger.info(f"✅ 청크 {i+1}/{total_chunks} 생성 완료")
            else:
                logger.error(f"❌ 청크 {i+1}/{total_chunks} 생성 실패")
            
            # API 요청 간 간격 (너무 많은 요청을 방지하기 위함)
            if i < total_chunks - 1:
                time.sleep(0.5)  # Eleven Labs 요청 제한 고려
            
        except Exception as e:
            logger.error(f"❌ 청크 {i+1}/{total_chunks} 생성 중 오류: {str(e)}")
            continue
    
    return chunk_paths

@lru_cache(maxsize=32)
def get_available_voices() -> List[Dict[str, Any]]:
    """
    Eleven Labs에서 사용 가능한 음성 목록 가져오기
    캐싱을 통해 반복 호출 최소화
    
    Returns:
        음성 정보 딕셔너리 리스트
    """
    url = "https://api.elevenlabs.io/v1/voices"
    
    headers = {
        "Accept": "application/json",
        "xi-api-key": api_key
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            voices = response.json().get("voices", [])
            return voices
        else:
            logger.warning(f"⚠️ Eleven Labs 음성 목록 가져오기 실패: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"⚠️ Eleven Labs API 요청 실패: {e}")
        return []

def get_voice_name(voice_id: str) -> str:
    """
    음성 ID로 음성 이름 가져오기
    
    Args:
        voice_id: Eleven Labs 음성 ID
        
    Returns:
        음성 이름 (찾을 수 없으면 ID 반환)
    """
    # 추천 음성에서 먼저 확인
    recommended_voices = list_recommended_voices()
    for name, vid, _ in recommended_voices:
        if vid == voice_id:
            return name
    
    # API에서 가져온 음성 목록에서 확인
    try:
        voices = get_available_voices()
        for voice in voices:
            if voice.get("voice_id") == voice_id:
                return voice.get("name", voice_id)
    except:
        pass
    
    return voice_id

def combine_audio_chunks(chunk_paths: List[str], output_path: str) -> Optional[str]:
    """
    오디오 청크 파일들을 하나로 결합
    
    Args:
        chunk_paths: 청크 파일 경로 리스트
        output_path: 출력 파일 경로
        
    Returns:
        결합된 파일 경로 또는 None
    """
    if not chunk_paths:
        logger.error("❌ 결합할 청크가 없습니다.")
        return None
    
    if len(chunk_paths) == 1:
        # 청크가 하나만 있으면 복사
        import shutil
        shutil.copy(chunk_paths[0], output_path)
        logger.info(f"✅ 단일 청크를 최종 파일로 복사: {output_path}")
        # 임시 파일 제거
        try:
            os.remove(chunk_paths[0])
        except Exception as e:
            logger.warning(f"⚠️ 임시 파일 제거 중 오류: {str(e)}")
        return output_path
    
    try:
        # pydub 사용
        from pydub import AudioSegment
        
        logger.info(f"🔄 {len(chunk_paths)}개 오디오 청크 결합 중...")
        combined = AudioSegment.empty()
        
        for i, path in enumerate(chunk_paths):
            try:
                audio_segment = AudioSegment.from_mp3(path)
                combined += audio_segment
                logger.info(f"✅ 청크 {i+1}/{len(chunk_paths)} 결합 완료")
            except Exception as e:
                logger.error(f"❌ 청크 {i+1} 결합 실패: {str(e)}")
        
        # 결합된 오디오 저장
        combined.export(output_path, format="mp3")
        
        # 임시 파일 제거
        cleanup_temp_files(chunk_paths)
        
        return output_path
    except ImportError:
        logger.error("⚠️ pydub 라이브러리가 설치되지 않아 청크를 결합할 수 없습니다.")
        logger.info("⚠️ pip install pydub를 실행하여 pydub를 설치하세요.")
        
        # 대체 방안: 첫 번째 청크만 반환
        if chunk_paths:
            import shutil
            shutil.copy(chunk_paths[0], output_path)
            logger.info(f"⚠️ pydub 없음, 첫 번째 청크를 복사: {output_path}")
            return output_path
        return None
    except Exception as e:
        logger.error(f"⚠️ 청크 결합 중 오류 발생: {e}")
        return None

def cleanup_temp_files(file_paths: List[str]) -> None:
    """
    임시 파일 정리
    
    Args:
        file_paths: 삭제할 파일 경로 리스트
    """
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug(f"✅ 임시 파일 삭제: {path}")
        except Exception as e:
            logger.warning(f"⚠️ 임시 파일 삭제 중 오류: {str(e)}")

def extract_speech_parts(script: str) -> str:
    """
    영상 지시사항과 형식 요소를 제외한 음성 부분만 추출
    
    Args:
        script: 원본 스크립트
        
    Returns:
        음성 부분만 포함된 스크립트
    """
    # 1. 각종 영상 지시사항 제거 (다양한 형식 지원)
    speech_only = re.sub(r'\[(영상|Visual|video):\s*.*?\]', '', script, flags=re.IGNORECASE)
    
    # 2. "Narrator: " 접두어 제거
    speech_only = re.sub(r'(narrator|내레이터):\s*', '', speech_only, flags=re.IGNORECASE)
    
    # 3. 마크다운 헤더(###, ---) 제거
    speech_only = re.sub(r'^#{1,6}\s+.*$', '', speech_only, flags=re.MULTILINE)
    speech_only = re.sub(r'^---+$', '', speech_only, flags=re.MULTILINE)
    
    # 4. 굵은 텍스트(**텍스트**) 처리 - 강조는 유지하되 마크업 제거
    speech_only = re.sub(r'\*\*(.*?)\*\*', r'\1', speech_only)
    
    # 5. 스크립트 섹션 제목 제거 (Introduction, Development 등)
    speech_only = re.sub(r'^\*\*(.*?)\*\*$', '', speech_only, flags=re.MULTILINE)
    
    # 6. 스크립트 중간의 다중 라인 정리
    speech_only = re.sub(r'\n{3,}', '\n\n', speech_only)
    
    # 7. 괄호 안 지시사항 제거
    speech_only = re.sub(r'\([^)]*\)', '', speech_only)
    
    # 8. 특수 문자 정리
    speech_only = speech_only.replace('"', '"')
    speech_only = speech_only.replace('"', '"')
    speech_only = speech_only.replace('"', '"')
    speech_only = speech_only.replace("'", "'")
    speech_only = speech_only.replace("'", "'")
    speech_only = speech_only.replace("'", "'")
    
    # 9. [End] 또는 [end] 태그 제거
    speech_only = re.sub(r'\[end\]', '', speech_only, flags=re.IGNORECASE)
    
    # 10. 빈 줄이 연속된 경우 하나로 정리
    speech_only = re.sub(r'\n\s*\n', '\n\n', speech_only)
    
    return speech_only.strip()

def split_script_into_chunks(script: str, max_chars: int = MAX_CHUNK_SIZE) -> List[str]:
    """
    스크립트를 청크로 분할하는 함수
    자연스러운 분할을 위해 문장 경계 고려
    
    Args:
        script: 분할할 스크립트
        max_chars: 청크 당 최대 문자 수
        
    Returns:
        청크 리스트
    """
    # 문장 단위로 분할
    sentences = split_into_sentences(script)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # 한 문장이 최대 크기를 초과하는 경우
        if len(sentence) > max_chars:
            # 현재 청크가 있으면 먼저 추가
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            
            # 긴 문장 분할 (구두점이나 접속사 기준)
            parts = split_long_sentence(sentence, max_chars)
            
            # 마지막 부분을 제외한 모든 부분을 청크로 추가
            chunks.extend(parts[:-1])
            
            # 마지막 부분은 다음 청크의 시작으로 사용
            current_chunk = parts[-1]
            
        # 현재 청크에 문장을 추가했을 때 최대 문자 수를 초과하는지 확인
        elif len(current_chunk) + len(sentence) + 1 <= max_chars:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # 현재 청크가 차면 chunks 리스트에 추가하고 새 청크 시작
            chunks.append(current_chunk)
            current_chunk = sentence
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리
    
    Args:
        text: 분리할 텍스트
        
    Returns:
        문장 리스트
    """
    # 문장 종료 표시로 분리 (영어/한국어 호환)
    sentence_endings = r'(?<=[.!?])\s+'
    raw_sentences = re.split(sentence_endings, text)
    
    # 빈 문장 제거 및 정리
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    return sentences

def split_long_sentence(sentence: str, max_length: int) -> List[str]:
    """
    긴 문장을 여러 부분으로 분할
    
    Args:
        sentence: 분할할 문장
        max_length: 각 부분의 최대 길이
        
    Returns:
        분할된 부분 리스트
    """
    # 구두점이나 접속사를 기준으로 분할
    split_points = [
        r'(?<=,)\s+',  # 쉼표 후
        r'(?<=;)\s+',  # 세미콜론 후
        r'(?<=:)\s+',  # 콜론 후
        r'\s+(?=and|or|but|because|however|therefore|thus|meanwhile|moreover|furthermore)\s+',  # 접속사 전
        r'\s+-\s+',    # 대시 주변
        r'\s+'         # 공백 (마지막 수단)
    ]
    
    parts = []
    remaining = sentence
    
    while len(remaining) > max_length:
        split_found = False
        
        # 각 분할 포인트 시도
        for pattern in split_points:
            # 남은 텍스트에서 패턴의 모든 매치 찾기
            matches = list(re.finditer(pattern, remaining))
            
            # 최대 길이 내에서 가장 먼 매치 지점 찾기
            valid_matches = [m for m in matches if m.end() <= max_length]
            
            if valid_matches:
                # 최대 길이에 가까운 지점에서 분할
                split_at = valid_matches[-1].end()
                parts.append(remaining[:split_at].strip())
                remaining = remaining[split_at:].strip()
                split_found = True
                break
        
        # 분할 지점을 찾지 못한 경우 (적절한 구두점이나 접속사가 없음)
        if not split_found:
            # 최대 길이에서 강제 분할
            parts.append(remaining[:max_length].strip())
            remaining = remaining[max_length:].strip()
    
    # 남은 부분 추가
    if remaining:
        parts.append(remaining)
    
    return parts

def process_script_for_tts(script: str) -> str:
    """
    TTS를 위한 스크립트 전처리
    
    Args:
        script: 원본 스크립트
        
    Returns:
        전처리된 스크립트
    """
    # 발음 개선을 위한 텍스트 처리
    processed = script
    
    # 줄바꿈 정리
    processed = re.sub(r'\n{2,}', '\n\n', processed)
    
    # 다양한 인용부호 통일
    processed = processed.replace('"', '"')
    processed = processed.replace('"', '"')
    processed = processed.replace('"', '"')
    processed = processed.replace("'", "'")
    processed = processed.replace("'", "'")
    processed = processed.replace("'", "'")
    
    # TTS가 잘 처리하지 못하는 특수 문자 정리
    processed = processed.replace('…', '...')
    processed = processed.replace('–', '-')
    processed = processed.replace('—', '-')
    
    return processed

def list_recommended_voices() -> List[Tuple[str, str, str]]:
    """
    영어 전문가 콘텐츠에 적합한 Eleven Labs 음성 추천 목록
    API에서 가져온 목록과 로컬 추천 목록 결합
    
    Returns:
        (이름, ID, 설명) 튜플 리스트
    """
    # 기본 추천 목록
    default_recommendations = [
        ("Wyatt", "YXpFCvM1S3JbWEJhoskW", "Wise Rustic Cowboy 목소리"),
        ("James", "EkK5I93UQWFDigLMpZcX", "Husky & Engaging"),
        ("Brian", "nPczCjzI2devNBz1zQrb", "권위 있는 남성 목소리, 군사/안보 전문가 느낌"),
    ]
    
    # API 호출이 가능하면 전체 음성 목록 가져오기 시도
    try:
        if api_key:
            all_voices = get_available_voices()
            
            # API에서 가져온 음성 중 영어권 음성만 추출
            english_voices = [
                (voice.get("name"), voice.get("voice_id"), "API에서 가져온 음성")
                for voice in all_voices
                if "en" in voice.get("labels", {}).get("language", "").lower()
            ]
            
            # 기본 추천 목록과 API 음성 목록 결합
            default_ids = [rec[1] for rec in default_recommendations]
            combined_list = list(default_recommendations)
            
            # 기본 목록에 없는 음성만 추가
            for voice in english_voices:
                if voice[1] not in default_ids:
                    combined_list.append(voice)
            
            return combined_list
    except Exception as e:
        logger.warning(f"⚠️ API에서 음성 목록을 가져오지 못했습니다: {str(e)}")
    
    # API 호출이 실패하면 기본 목록만 반환
    return default_recommendations

def batch_generate_tts(
    scripts: List[str], 
    voice_id: str = "pqHfZKP75CvOlQylNhV4", 
    output_dir: str = "output_audio",
    filename_prefix: str = "speech",
    **kwargs
) -> List[str]:
    """
    여러 스크립트에 대한 TTS 생성을 병렬로 처리
    
    Args:
        scripts: 스크립트 리스트
        voice_id: Eleven Labs 음성 ID
        output_dir: 오디오 파일 저장 디렉토리
        filename_prefix: 생성된 오디오 파일의 접두사
        **kwargs: generate_tts_elevenlabs에 전달할 추가 인자
        
    Returns:
        생성된 오디오 파일 경로 리스트
    """
    if not scripts:
        logger.warning("⚠️ 처리할 스크립트가 없습니다.")
        return []
    
    if not api_key:
        logger.error("❌ Eleven Labs API 키가 설정되지 않았습니다.")
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_scripts = len(scripts)
    logger.info(f"🔄 {total_scripts}개 스크립트 TTS 생성 시작")

    # Eleven Labs API 요청 제한 고려하여 병렬 처리
    max_parallel = min(3, total_scripts)  # 최대 3개 동시 처리
    
    # 병렬 처리 함수
    def process_script(script_data):
        idx, script = script_data
        try:
            # 각 스크립트마다 고유한 파일명 생성
            script_prefix = f"{filename_prefix}_{idx+1}"
            
            logger.info(f"[{idx+1}/{total_scripts}] 스크립트 TTS 생성 중...")
            output_path = generate_tts_elevenlabs(
                script, 
                voice_id=voice_id, 
                output_dir=output_dir,
                filename_prefix=script_prefix,
                **kwargs
            )
            
            if output_path:
                logger.info(f"✅ [{idx+1}/{total_scripts}] TTS 생성 완료: {os.path.basename(output_path)}")
                return idx, output_path, True
            else:
                logger.error(f"❌ [{idx+1}/{total_scripts}] TTS 생성 실패")
                return idx, "", False
        except Exception as e:
            logger.error(f"❌ [{idx+1}/{total_scripts}] TTS 생성 중 오류: {str(e)}")
            return idx, "", False
    
    # 병렬 처리 실행
    results = []
    
    # Eleven Labs API 요청 제한 고려하여 병렬 처리
    max_parallel = min(3, total_scripts)  # 최대 3개 동시 처리
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # 일괄 제출 대신 하나씩 제출하고 완료될 때마다 다음 작업 제출
        future_to_idx = {}
        remaining_items = list(enumerate(scripts))
        
        # 첫 번째 배치 제출 (워커 수만큼)
        initial_batch = remaining_items[:max_parallel]
        remaining_items = remaining_items[max_parallel:]
        
        for item in initial_batch:
            future = executor.submit(process_script, item)
            future_to_idx[future] = item[0]
        
        # 완료된 작업 처리 및 새 작업 제출
        while future_to_idx:
            # 완료된 작업 하나 가져오기
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
                    logger.error(f"❌ TTS 생성 중 예외 발생: {str(e)}")
                    results.append((future_to_idx[future], "", False))
                
                # 처리된 future 삭제
                del future_to_idx[future]
    
    # 결과 정렬 (원래 순서대로)
    results.sort(key=lambda x: x[0])
    
    # 성공한 경로만 필터링
    successful_paths = [res[1] for res in results if res[2]]
    
    success_count = len(successful_paths)
    logger.info(f"🏁 TTS 생성 완료: 성공 {success_count}개, 실패 {total_scripts - success_count}개")
    
    return successful_paths

def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """
    오디오 파일 정보 가져오기
    
    Args:
        audio_path: 오디오 파일 경로
        
    Returns:
        오디오 정보 딕셔너리
    """
    info = {
        "duration": None,
        "format": None,
        "channels": None,
        "sample_rate": None,
        "bit_rate": None,
        "file_size": None
    }
    
    # 파일 크기
    try:
        info["file_size"] = os.path.getsize(audio_path)
    except:
        pass
    
    # pydub으로 정보 가져오기
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        info["duration"] = audio.duration_seconds
        info["channels"] = audio.channels
        info["sample_rate"] = audio.frame_rate
        info["format"] = Path(audio_path).suffix.lstrip('.')
        return info
    except:
        pass
    
    # mutagen으로 시도
    try:
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        info["duration"] = audio.info.length
        info["sample_rate"] = audio.info.sample_rate
        info["bit_rate"] = audio.info.bitrate
        info["format"] = "mp3"
        return info
    except:
        pass
    
    # ffprobe로 시도
    try:
        import subprocess
        import json
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", audio_path
        ]
        output = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(output)
        
        if "format" in data:
            fmt = data["format"]
            info["format"] = fmt.get("format_name")
            info["duration"] = float(fmt.get("duration", 0))
            info["bit_rate"] = int(fmt.get("bit_rate", 0))
        
        if "streams" in data and data["streams"]:
            stream = data["streams"][0]
            info["channels"] = stream.get("channels")
            info["sample_rate"] = int(stream.get("sample_rate", 0))
        
        return info
    except:
        pass
    
    return info

if __name__ == "__main__":
    # 테스트 코드
    test_script = """
    The recent developments in Eastern Europe have significantly altered the strategic landscape of the region.
    [Video: Map of Eastern Europe with highlighted borders]
    
    Military analysts at the RAND Corporation suggest that this shift could impact NATO's defensive posture along its eastern flank.
    """
    
    logger.info("📋 Eleven Labs 추천 음성 목록:")
    recommended_voices = list_recommended_voices()
    for i, (name, voice_id, desc) in enumerate(recommended_voices):
        logger.info(f"{i+1}. {name}: {desc}")
    
    if api_key:
        logger.info("\n🔊 테스트 스크립트로 TTS 생성 중...")
        voice_id = "pqHfZKP75CvOlQylNhV4"  # Bill 음성 사용
        output_path = generate_tts_elevenlabs(test_script, voice_id=voice_id)
        if output_path:
            logger.info(f"✅ 테스트 완료: {output_path}")
            
            # 오디오 정보 출력
            audio_info = get_audio_info(output_path)
            logger.info(f"📊 오디오 정보: 길이 {audio_info['duration']:.1f}초, 크기 {audio_info['file_size']/1024:.1f} KB")
        else:
            logger.error("❌ TTS 생성 실패")
    else:
        logger.warning("⚠️ API 키가 설정되지 않아 테스트를 건너뜁니다.")