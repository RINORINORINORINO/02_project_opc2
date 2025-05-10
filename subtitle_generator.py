import os
import re
import time
import json
import logging
from datetime import timedelta
from typing import Optional, Dict, List, Tuple, Any, Union
from functools import lru_cache
import concurrent.futures
import threading
import random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 전역 설정
MAX_RETRIES = 3
BASE_RETRY_DELAY = 1  # 초 단위
MAX_WORKERS = 2  # 자막 처리 병렬 워커 수

# Whisper API 호출용 세마포어 - 음성 인식은 무거운 작업이므로 제한 강화
whisper_semaphore = threading.Semaphore(2)  # 최대 2개 동시 Whisper 요청

def api_call_with_retry(func, *args, max_retries=MAX_RETRIES, **kwargs):
    """
    API 호출 함수를 재시도 로직으로 감싸는 유틸리티 함수
    지수 백오프 전략 사용
    
    Args:
        func: 호출할 함수
        *args, **kwargs: 함수에 전달할 인자들
        
    Returns:
        함수 호출 결과
    """
    with whisper_semaphore:
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

def generate_srt(
    script: str, 
    audio_path: str, 
    output_dir: str = "output_subtitles", 
    use_whisper: bool = True,
    max_chars_per_subtitle: int = 42  # 자막 당 최대 문자 수
) -> str:
    """
    오디오 파일과 스크립트를 기반으로 SRT 자막 파일 생성
    
    Args:
        script: 원본 스크립트 텍스트
        audio_path: 오디오 파일 경로
        output_dir: 자막 파일 저장 디렉토리
        use_whisper: Whisper 모델을 사용하여 정확한 타이밍 생성 여부
        max_chars_per_subtitle: 자막 당 최대 문자 수
        
    Returns:
        생성된 SRT 파일 경로
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 오디오 파일 존재 확인
        if not os.path.exists(audio_path):
            logger.error(f"❌ 오디오 파일이 존재하지 않습니다: {audio_path}")
            return ""
        
        # 출력 파일 경로 설정
        filename = os.path.splitext(os.path.basename(audio_path))[0] + ".srt"
        srt_path = os.path.join(output_dir, filename)
        
        # 스크립트 전처리
        clean_script = preprocess_script(script)
        
        if use_whisper:
            try:
                logger.info(f"🎤 Whisper 기반 자막 생성 시작: {os.path.basename(audio_path)}")
                return generate_whisper_srt(clean_script, audio_path, srt_path)
            except Exception as e:
                logger.warning(f"⚠️ Whisper 자막 생성 실패: {e}")
                logger.info("⚠️ 단순 시간 분할 방식으로 대체합니다.")
                return generate_simple_srt(clean_script, audio_path, srt_path, max_chars_per_subtitle)
        else:
            logger.info(f"📝 단순 시간 분할 자막 생성 시작: {os.path.basename(audio_path)}")
            return generate_simple_srt(clean_script, audio_path, srt_path, max_chars_per_subtitle)
    
    except Exception as e:
        logger.error(f"❌ 자막 생성 중 오류 발생: {str(e)}")
        return ""

def preprocess_script(script: str) -> str:
    """
    자막 생성을 위한 스크립트 전처리
    
    Args:
        script: 원본 스크립트
        
    Returns:
        전처리된 스크립트
    """
    # 스크립트에서 영상 지시사항 제거
    clean_script = re.sub(r'\[영상:.*?\]', '', script)
    clean_script = re.sub(r'\[Video:.*?\]', '', clean_script)
    
    # 제목 구조 (#으로 시작하는 마크다운 헤더) 제거
    clean_script = re.sub(r'^#+ .*$', '', clean_script, flags=re.MULTILINE)
    
    # 괄호 안 내용 (방향 지시 등) 제거
    clean_script = re.sub(r'\(([^)]*)\)', '', clean_script)
    
    # 특수 문자 정리
    clean_script = re.sub(r'["""]', '"', clean_script)
    clean_script = re.sub(r'[\'"]', "'", clean_script)
    clean_script = re.sub(r'["]', "'", clean_script)
    
    # 빈 줄 정리
    clean_script = re.sub(r'\n{3,}', '\n\n', clean_script)
    
    # 공백 정리
    clean_script = re.sub(r' {2,}', ' ', clean_script)
    
    return clean_script.strip()

def generate_whisper_srt(
    script: str, 
    audio_path: str, 
    srt_path: str, 
    model_size: str = "base",
    use_script_matching: bool = True,
    max_chars_per_subtitle: int = 42  # 이 매개변수를 추가
) -> str:
    """
    OpenAI Whisper 모델을 사용하여 오디오 파일에서 음성 인식 기반 자막 생성
    스크립트는 음성 인식 결과 개선에 사용
    
    Args:
        script: 전처리된 스크립트 텍스트
        audio_path: 오디오 파일 경로
        srt_path: 출력 SRT 파일 경로
        model_size: Whisper 모델 크기 ("tiny", "base", "small", "medium", "large")
        use_script_matching: 스크립트 매칭 사용 여부
        
    Returns:
        생성된 SRT 파일 경로
    """
    start_time = time.time()
    
    try:
        # Whisper 모델 로드
        import whisper
        logger.info(f"🔄 Whisper 모델 로드 중: {model_size}")
        model = whisper.load_model(model_size)
        
        # 경로 안정화 + 절대 경로로 변경
        safe_path = os.path.abspath(os.path.normpath(audio_path))
        
        # 오디오 파일로부터 자막 생성
        logger.info(f"🎧 오디오 분석 중: {os.path.basename(safe_path)}")
        
        # Whisper API 호출 함수
        def run_whisper():
            return model.transcribe(
                safe_path,
                language="en",  # 자동 언어 감지 (또는 "ko", "en" 등으로 명시)
                task="transcribe",
                vad_filter=True,  # 음성 구간 탐지 필터링
                word_timestamps=True  # 단어 단위 타임스탬프 (가능한 경우)
            )
        
        # 재시도 로직 적용
        result = api_call_with_retry(run_whisper)
        
        # 결과에서 세그먼트 추출
        segments = result["segments"]
        logger.info(f"✅ Whisper 음성 인식 완료: {len(segments)}개 세그먼트 감지")
        
        # 스크립트를 문장 단위로 분리
        script_sentences = []
        if use_script_matching and script:
            script_sentences = split_into_sentences(script)
            logger.info(f"📄 스크립트에서 {len(script_sentences)}개 문장 추출")
        
        # 원본 스크립트와 Whisper 인식 결과를 매칭하여 개선된 자막 생성
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments):
                start_time_sec = segment["start"]
                end_time_sec = segment["end"]
                text = segment["text"].strip()
                
                # 스크립트 매칭을 사용하는 경우
                if use_script_matching and script_sentences:
                    # 단어 수가 일정 개수 이하인 세그먼트는 스크립트에서 가장 유사한 부분으로 대체
                    if len(text.split()) <= 3:
                        best_match = find_best_match(text, script_sentences)
                        if best_match:
                            text = best_match
                            # 사용한 문장 제거 (안전하게 처리)
                            try:
                                script_sentences = [s for s in script_sentences if s != best_match]
                            except:
                                # 오류 발생 시 무시하고 계속 진행
                                pass
                    
                    # 세그먼트 텍스트가 너무 긴 경우 분할
                    if len(text) > max_chars_per_subtitle:
                        subtitles = split_long_subtitle(text, max_chars_per_subtitle)
                        duration_per_part = (end_time_sec - start_time_sec) / len(subtitles)
                        
                        for j, subtitle_text in enumerate(subtitles):
                            sub_start = start_time_sec + (j * duration_per_part)
                            sub_end = sub_start + duration_per_part
                            
                            # SRT 형식으로 작성
                            f.write(f"{i + j + 1}\n")
                            f.write(f"{format_timestamp(sub_start)} --> {format_timestamp(sub_end)}\n")
                            f.write(f"{subtitle_text}\n\n")
                    else:
                        # SRT 형식으로 작성 (단일 세그먼트)
                        f.write(f"{i + 1}\n")
                        f.write(f"{format_timestamp(start_time_sec)} --> {format_timestamp(end_time_sec)}\n")
                        f.write(f"{text}\n\n")
                else:
                    # 스크립트 매칭 없이 Whisper 결과만 사용
                    # 세그먼트 텍스트가 너무 긴 경우 분할
                    if len(text) > max_chars_per_subtitle:
                        subtitles = split_long_subtitle(text, max_chars_per_subtitle)
                        duration_per_part = (end_time_sec - start_time_sec) / len(subtitles)
                        
                        for j, subtitle_text in enumerate(subtitles):
                            sub_start = start_time_sec + (j * duration_per_part)
                            sub_end = sub_start + duration_per_part
                            
                            # SRT 형식으로 작성
                            f.write(f"{i + j + 1}\n")
                            f.write(f"{format_timestamp(sub_start)} --> {format_timestamp(sub_end)}\n")
                            f.write(f"{subtitle_text}\n\n")
                    else:
                        # SRT 형식으로 작성 (단일 세그먼트)
                        f.write(f"{i + 1}\n")
                        f.write(f"{format_timestamp(start_time_sec)} --> {format_timestamp(end_time_sec)}\n")
                        f.write(f"{text}\n\n")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Whisper 자막 생성 완료: {srt_path} ({elapsed_time:.1f}초 소요)")
        return srt_path
        
    except ImportError:
        logger.error("❌ Whisper 라이브러리가 설치되지 않았습니다.")
        raise ImportError("Whisper 라이브러리가 설치되지 않았습니다. pip install openai-whisper로 설치하세요.")
    except Exception as e:
        logger.error(f"❌ Whisper 자막 생성 실패: {str(e)}")
        raise

def generate_simple_srt(
    script: str, 
    audio_path: str, 
    srt_path: str,
    max_chars_per_subtitle: int = 42
) -> str:
    """
    오디오 파일 길이와 스크립트를 기반으로 간단한 시간 분할 방식의 자막 생성
    
    Args:
        script: 전처리된 스크립트
        audio_path: 오디오 파일 경로
        srt_path: 출력 SRT 파일 경로
        max_chars_per_subtitle: 자막 당 최대 문자 수
        
    Returns:
        생성된 SRT 파일 경로
    """
    start_time = time.time()
    
    try:
        # 오디오 파일 길이 확인
        audio_duration = get_audio_duration(audio_path)
        
        if audio_duration is None:
            logger.warning("⚠️ 오디오 길이를 확인할 수 없습니다. 추정값을 사용합니다.")
            # 영어 기준 평균 말하기 속도: 1분당 약 150단어
            # 한국어는 다를 수 있으므로 조정 가능
            words = re.findall(r'\S+', script)
            estimated_duration = len(words) / 2.5  # 초당 약 2.5단어로 가정
            audio_duration = estimated_duration
            logger.info(f"📊 추정된 오디오 길이: {audio_duration:.1f}초 (단어 수: {len(words)})")
        else:
            logger.info(f"📊 오디오 길이: {audio_duration:.1f}초")
        
        # 스크립트를 자막 단위로 분할
        subtitles = split_script_into_subtitles(script, max_chars_per_subtitle)
        
        if not subtitles:
            logger.error("❌ 스크립트에서 자막을 추출할 수 없습니다.")
            return ""
        
        logger.info(f"📝 스크립트를 {len(subtitles)}개 자막으로 분할")
        
        # 자막별 시간 할당
        duration_per_subtitle = audio_duration / len(subtitles)
        
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, subtitle in enumerate(subtitles):
                start_time_sec = i * duration_per_subtitle
                end_time_sec = (i + 1) * duration_per_subtitle
                
                # SRT 형식으로 작성
                f.write(f"{i + 1}\n")
                f.write(f"{format_timestamp(start_time_sec)} --> {format_timestamp(end_time_sec)}\n")
                f.write(f"{subtitle}\n\n")
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ 단순 자막 생성 완료: {srt_path} ({elapsed_time:.1f}초 소요)")
        return srt_path
        
    except Exception as e:
        logger.error(f"❌ 단순 자막 생성 실패: {str(e)}")
        raise

def split_script_into_subtitles(script: str, max_chars_per_subtitle: int = 42) -> List[str]:
    """
    스크립트를 자막 단위로 분할
    
    Args:
        script: 전처리된 스크립트
        max_chars_per_subtitle: 자막 당 최대 문자 수
        
    Returns:
        자막 텍스트 리스트
    """
    # 문장 단위로 분리
    sentences = split_into_sentences(script)
    
    subtitles = []
    current_subtitle = ""
    
    for sentence in sentences:
        # 문장이 한 자막에 들어갈 수 있으면 현재 자막에 추가
        if len(current_subtitle) + len(sentence) + 1 <= max_chars_per_subtitle:
            if current_subtitle:
                current_subtitle += " " + sentence
            else:
                current_subtitle = sentence
        else:
            # 현재 문장이 너무 길면 여러 자막으로 분할
            if not current_subtitle:
                # 문장 자체가 한 자막보다 길면 분할
                parts = split_long_subtitle(sentence, max_chars_per_subtitle)
                subtitles.extend(parts)
            else:
                # 이전까지의 자막 저장하고 새 자막 시작
                subtitles.append(current_subtitle)
                
                # 새 문장이 자막 길이보다 짧으면 새 자막으로, 길면 분할
                if len(sentence) <= max_chars_per_subtitle:
                    current_subtitle = sentence
                else:
                    parts = split_long_subtitle(sentence, max_chars_per_subtitle)
                    subtitles.extend(parts[:-1])  # 마지막 부분은 다음 자막 시작으로 사용
                    current_subtitle = parts[-1]
    
    # 마지막 자막 추가
    if current_subtitle:
        subtitles.append(current_subtitle)
    
    return subtitles

def split_into_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리
    
    Args:
        text: 분리할 텍스트
        
    Returns:
        문장 리스트
    """
    # 문장 종료 표시로 분리
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # 빈 문장 제거 및 정리
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    
    return sentences

def split_long_subtitle(text: str, max_length: int) -> List[str]:
    """
    긴 텍스트를 여러 자막으로 분할
    
    Args:
        text: 분할할 텍스트
        max_length: 자막 당 최대 문자 수
        
    Returns:
        자막 텍스트 리스트
    """
    # 단어 또는 구 단위로 분리
    parts = []
    words = text.split()
    current_part = ""
    
    for word in words:
        if len(current_part) + len(word) + 1 <= max_length:
            if current_part:
                current_part += " " + word
            else:
                current_part = word
        else:
            parts.append(current_part)
            current_part = word
    
    if current_part:
        parts.append(current_part)
    
    # 부분이 없으면 최대 길이로 자르기
    if not parts:
        return [text[:max_length]]
    
    return parts

def get_audio_duration(audio_path: str) -> Optional[float]:
    """
    오디오 파일 길이 확인 (초 단위)
    여러 라이브러리를 시도하여 가능한 방법으로 길이 추출
    
    Args:
        audio_path: 오디오 파일 경로
        
    Returns:
        오디오 길이(초) 또는 None
    """
    methods = [
        get_duration_pydub,
        get_duration_librosa,
        get_duration_mutagen,
        get_duration_ffprobe
    ]
    
    for method in methods:
        try:
            duration = method(audio_path)
            if duration is not None and duration > 0:
                return duration
        except Exception:
            continue
    
    return None

def get_duration_pydub(audio_path: str) -> Optional[float]:
    """pydub으로 오디오 길이 가져오기"""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return audio.duration_seconds
    except Exception:
        return None

def get_duration_librosa(audio_path: str) -> Optional[float]:
    """librosa로 오디오 길이 가져오기"""
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except Exception:
        return None

def get_duration_mutagen(audio_path: str) -> Optional[float]:
    """mutagen으로 오디오 길이 가져오기"""
    try:
        from mutagen.mp3 import MP3
        audio = MP3(audio_path)
        return audio.info.length
    except Exception:
        try:
            # 다른 포맷 시도
            from mutagen.wave import WAVE
            audio = WAVE(audio_path)
            return audio.info.length
        except Exception:
            return None

def get_duration_ffprobe(audio_path: str) -> Optional[float]:
    """ffprobe로 오디오 길이 가져오기"""
    try:
        import subprocess
        import json
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", audio_path
        ]
        output = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(output)
        if "format" in data and "duration" in data["format"]:
            return float(data["format"]["duration"])
        elif "streams" in data and len(data["streams"]) > 0 and "duration" in data["streams"][0]:
            return float(data["streams"][0]["duration"])
        return None
    except Exception:
        return None

def format_timestamp(seconds: float) -> str:
    """
    초를 SRT 타임스탬프 형식으로 변환 (HH:MM:SS,mmm)
    
    Args:
        seconds: 초 단위 시간
        
    Returns:
        SRT 타임스탬프 문자열
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def find_best_match(text: str, candidates: List[str]) -> Optional[str]:
    """
    텍스트 유사성 기반으로 가장 잘 맞는 후보 찾기 (개선된 버전)
    
    Args:
        text: 검색할 텍스트
        candidates: 후보 텍스트 리스트
        
    Returns:
        가장 유사한 후보 텍스트 또는 None
    """
    if not text or not candidates:
        return None
    
    # 정확히 포함되는 경우 먼저 확인
    exact_matches = []
    for candidate in candidates:
        if text.lower() in candidate.lower():
            exact_matches.append(candidate)
    
    if exact_matches:
        # 가장 짧은 정확한 일치 반환 (더 정확한 컨텍스트 제공)
        return min(exact_matches, key=len)
    
    # 단어 기반 유사도 점수 계산
    text_words = set(text.lower().split())
    if not text_words:  # 단어가 없는 경우
        return None
        
    best_score = 0
    best_match = None
    
    for candidate in candidates:
        candidate_words = set(candidate.lower().split())
        if not candidate_words:  # 빈 후보 건너뛰기
            continue
            
        common_words = text_words.intersection(candidate_words)
        
        if common_words:
            # 자카드 유사도: 교집합 / 합집합
            score = len(common_words) / len(text_words.union(candidate_words))
            
            # 보너스: 연속된 단어 매칭 확인
            text_seq = text.lower().split()
            candidate_seq = candidate.lower().split()
            
            for i in range(len(text_seq) - 1):
                if i + 1 < len(text_seq):
                    text_bigram = f"{text_seq[i]} {text_seq[i+1]}"
                    # 연속된 두 단어가 후보에 있는지 확인
                    for j in range(len(candidate_seq) - 1):
                        if j + 1 < len(candidate_seq):
                            candidate_bigram = f"{candidate_seq[j]} {candidate_seq[j+1]}"
                            if text_bigram == candidate_bigram:
                                score += 0.1  # 연속 매칭 보너스
            
            if score > best_score:
                best_score = score
                best_match = candidate
    
    # 최소 유사도 임계값
    if best_score > 0.15:  # 임계값 상향 조정
        return best_match
    return None

def batch_generate_srt(scripts: List[str], audio_paths: List[str], output_dir: str = "output_subtitles") -> List[str]:
    """
    여러 오디오 파일에 대한 자막 파일을 병렬로 생성
    
    Args:
        scripts: 스크립트 리스트
        audio_paths: 오디오 파일 경로 리스트
        output_dir: 자막 파일 저장 디렉토리
        
    Returns:
        생성된 SRT 파일 경로 리스트
    """
    if len(scripts) != len(audio_paths):
        logger.error(f"❌ 스크립트 수 ({len(scripts)})와 오디오 파일 수 ({len(audio_paths)})가 일치하지 않습니다.")
        return []
    
    # 작업량에 따라 워커 수 동적 조정 (자막 생성은 무거운 작업)
    total_items = len(scripts)
    worker_count = min(MAX_WORKERS, total_items)
    
    logger.info(f"🔄 {total_items}개 자막 생성 시작 (워커: {worker_count}개)")
    
    # 병렬 처리 함수
    def process_single(args):
        idx, script, audio_path = args
        try:
            logger.info(f"[{idx+1}/{len(scripts)}] 자막 생성 중: {os.path.basename(audio_path)}")
            srt_path = generate_srt(script, audio_path, output_dir)
            if srt_path:
                logger.info(f"✅ [{idx+1}/{len(scripts)}] 자막 생성 완료: {os.path.basename(srt_path)}")
                return idx, srt_path, True
            return idx, "", False
        except Exception as e:
            logger.error(f"❌ [{idx+1}/{len(scripts)}] 자막 생성 실패: {str(e)}")
            return idx, "", False
    
    # 병렬 처리 실행
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(scripts))) as executor:
        
        # 일괄 제출 대신 하나씩 제출하고 완료될 때마다 다음 작업 제출
        future_to_idx = {}
        remaining_items = list(enumerate(zip(scripts, audio_paths)))
        
        # 첫 번째 배치 제출 (워커 수만큼)
        initial_batch = remaining_items[:worker_count]
        remaining_items = remaining_items[worker_count:]
        
        for item in initial_batch:
            future = executor.submit(process_single, (item[0], item[1][0], item[1][1]))
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
                        new_future = executor.submit(
                            process_single, 
                            (new_item[0], new_item[1][0], new_item[1][1])
                        )
                        future_to_idx[new_future] = new_item[0]
                    
                except Exception as e:
                    logger.error(f"❌ 자막 생성 중 예외 발생: {str(e)}")
                    results.append((future_to_idx[future], "", False))
                
                # 처리된 future 삭제
                del future_to_idx[future]
    
    # 결과 정렬 (원래 순서대로)
    results.sort(key=lambda x: x[0])
    
    # 성공한 경로만 필터링
    successful_paths = [res[1] for res in results if res[2]]
    
    success_count = len(successful_paths)
    logger.info(f"🏁 자막 생성 완료: 성공 {success_count}개, 실패 {len(scripts) - success_count}개")
    
    return successful_paths

if __name__ == "__main__":
    # 테스트 코드
    test_script = """
    인공지능의 발전은 우리 사회를 크게 변화시키고 있습니다.
    특히 자연어 처리 기술의 발전으로 기계와의 대화가 더욱 자연스러워지고 있습니다.
    이러한 변화는 일상생활뿐만 아니라 산업 전반에 걸쳐 혁신을 가져오고 있으며,
    우리는 이제 인공지능과 함께하는 새로운 시대를 맞이하고 있습니다.
    """
    
    # 테스트용 오디오 파일이 있는 경우
    test_audio = "output_audio/test.mp3"
    if os.path.exists(test_audio):
        logger.info("🔄 테스트 자막 생성 시작")
        srt_path = generate_srt(test_script, test_audio)
        if srt_path:
            logger.info(f"✅ 테스트 자막 생성 완료: {srt_path}")
        else:
            logger.error("❌ 테스트 자막 생성 실패")
    else:
        logger.warning("⚠️ 테스트할 오디오 파일이 없습니다.")
        
    # 추가 테스트: 여러 자막 파일 일괄 생성 (테스트용)
    test_scripts = [test_script] * 2
    test_audios = [test_audio] * 2 if os.path.exists(test_audio) else []
    
    if test_audios:
        logger.info("🔄 일괄 자막 생성 테스트 시작")
        batch_results = batch_generate_srt(test_scripts, test_audios)
        logger.info(f"✅ 일괄 자막 생성 테스트 완료: {len(batch_results)}개 성공")