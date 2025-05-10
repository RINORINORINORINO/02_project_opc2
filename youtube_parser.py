import os
import re
import time
import json
import logging
from typing import Dict, Optional, Tuple, List, Any, Union
from functools import lru_cache
import concurrent.futures
from urllib.parse import urlparse, parse_qs
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 세션 생성 및 재시도 설정
def create_session() -> requests.Session:
    """향상된 재시도 로직을 가진 요청 세션 생성"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# 기본 세션 생성
session = create_session()

# API 호출 재시도 유틸리티
def api_call_with_retry(func, *args, max_retries=3, **kwargs):
    """API 호출 함수에 재시도 로직 추가"""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:  # 마지막 시도였으면 예외 발생
                raise
            
            # 지수 백오프 (1초, 2초, 4초...)
            delay = 1 * (2 ** attempt)
            logger.warning(f"⚠️ API 호출 실패 ({attempt+1}/{max_retries}), {delay}초 후 재시도: {str(e)}")
            time.sleep(delay)

@lru_cache(maxsize=32)
def parse_youtube(url: str, output_dir: str = "temp_youtube") -> str:
    """
    유튜브 영상 URL에서 콘텐츠 추출 - 캐싱 및 오류 처리 강화
    
    1. 자막이 있으면 자막 추출
    2. 자막이 없으면 오디오 다운로드 후 Whisper로 음성 인식
    3. 설명, 제목 등 메타데이터도 함께 추출
    
    Args:
        url: 유튜브 영상 URL
        output_dir: 임시 파일 저장 디렉토리
        
    Returns:
        추출된 텍스트 내용
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 ID 추출
    video_id = extract_video_id(url)
    if not video_id:
        logger.error(f"❌ 유효한 YouTube 비디오 ID를 추출할 수 없습니다: {url}")
        return f"오류: 유효한 YouTube URL이 아닙니다 ({url})"
    
    try:
        logger.info(f"🎬 유튜브 영상 분석 중: {url} (ID: {video_id})")
        
        # 캐시 파일 경로
        cache_path = os.path.join(output_dir, f"{video_id}_content.txt")
        
        # 캐시 파일이 있으면 읽어서 반환
        if os.path.exists(cache_path):
            logger.info(f"📂 캐시에서 유튜브 콘텐츠 로드: {video_id}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        
        # 메타데이터 가져오기
        metadata = get_youtube_metadata(url, video_id)
        
        # 메타데이터를 텍스트로 변환
        meta_text = f"제목: {metadata['title']}\n"
        meta_text += f"채널: {metadata['author']}\n"
        meta_text += f"게시일: {metadata['publish_date']}\n"
        meta_text += f"조회수: {metadata['views']:,}\n\n"
        meta_text += f"설명:\n{metadata['description']}\n\n"
        
        # 자막 처리 시도
        transcript_text = ""
        has_transcript = False
        
        # 1. YouTube에서 직접 자막 가져오기 시도
        try:
            transcript_text = get_youtube_transcript(video_id)
            if transcript_text:
                has_transcript = True
                logger.info("✅ YouTube 자막 추출 성공")
        except Exception as e:
            logger.warning(f"⚠️ YouTube 자막 추출 실패: {e}")
        
        # 2. 자막이 없는 경우 음성 인식 수행
        if not has_transcript:
            logger.info("🔊 자막이 없어 음성 인식을 시도합니다...")
            
            # 오디오 다운로드
            audio_path = os.path.join(output_dir, f"{video_id}.mp3")
            
            # 이미 다운로드된 파일이 있는지 확인
            if not os.path.exists(audio_path):
                download_youtube_audio(url, video_id, audio_path)
            else:
                logger.info(f"✅ 이미 다운로드된 오디오 파일 사용: {audio_path}")
            
            # Whisper로 음성 인식
            if os.path.exists(audio_path):
                transcript_text = transcribe_with_whisper(audio_path)
                if transcript_text:
                    has_transcript = True
                    logger.info("✅ Whisper 음성 인식 성공")
            else:
                logger.error(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_path}")
        
        # 최종 결과 조합
        if has_transcript:
            result = meta_text + "내용 스크립트:\n" + transcript_text
        else:
            result = meta_text + "내용을 추출할 수 없습니다."
            
        # 결과 캐싱
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(result)
            
        logger.info(f"✅ 유튜브 콘텐츠 추출 완료: {metadata['title']}")
        return result
            
    except Exception as e:
        logger.error(f"❌ 유튜브 영상 처리 오류: {str(e)}")
        return f"유튜브 영상 처리 실패 ({url}): {str(e)}"

def download_youtube_audio(url: str, video_id: str, output_path: str) -> bool:
    """유튜브 영상에서 오디오 다운로드"""
    try:
        # 동적 임포트 - 필요할 때만 라이브러리 로드
        from pytube import YouTube
        
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if not audio_stream:
            logger.error("❌ 오디오 스트림을 찾을 수 없습니다")
            return False
            
        logger.info(f"🔽 오디오 다운로드 중... ({yt.title})")
        audio_stream.download(filename=output_path)
        logger.info(f"✅ 오디오 다운로드 완료: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ 오디오 다운로드 실패: {str(e)}")
        return False

def get_youtube_metadata(url: str, video_id: str) -> Dict[str, Any]:
    """유튜브 영상 메타데이터 가져오기"""
    try:
        # pytube 사용 시도
        from pytube import YouTube
        
        yt = YouTube(url)
        return {
            "title": yt.title or "제목 없음",
            "author": yt.author or "채널 정보 없음",
            "publish_date": str(yt.publish_date) if yt.publish_date else "알 수 없음",
            "views": yt.views or 0,
            "description": yt.description or "설명 없음"
        }
    except:
        # Web 파싱 대체 방법
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = session.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 제목 추출
            title_elem = soup.find('meta', property='og:title')
            title = title_elem['content'] if title_elem else "제목 없음"
            
            # 채널명 추출
            author_elem = soup.find('link', itemprop='name')
            author = author_elem['content'] if author_elem else "채널 정보 없음"
            
            # 설명 추출
            desc_elem = soup.find('meta', property='og:description')
            description = desc_elem['content'] if desc_elem else "설명 없음"
            
            return {
                "title": title,
                "author": author,
                "publish_date": "알 수 없음",
                "views": 0,
                "description": description
            }
        except Exception as e:
            logger.error(f"❌ 웹 파싱을 통한 메타데이터 추출 실패: {str(e)}")
            
            # 기본값 반환
            return {
                "title": f"YouTube 비디오 {video_id}",
                "author": "채널 정보 없음",
                "publish_date": "알 수 없음",
                "views": 0,
                "description": "설명을 가져올 수 없습니다."
            }

@lru_cache(maxsize=64)
def get_youtube_transcript(video_id: str) -> str:
    """
    유튜브에서 자막 추출 시도 (여러 방법 사용)
    
    1. youtube_transcript_api 사용
    2. 실패 시 웹 파싱 시도
    """
    try:
        logger.info(f"🔤 자막 추출 시도: {video_id}")
        
        # 방법 1: youtube_transcript_api 사용
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # 한국어 자막 우선, 없으면 영어, 그 외 언어 순으로 시도
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript_data = None
            
            # 언어 우선순위
            preferred_langs = ['ko', 'en', 'ja', 'zh-Hans', 'zh-Hant', 'es', 'fr', 'de']
            
            # 수동 자막 먼저 시도
            try:
                for lang in preferred_langs:
                    try:
                        transcript = transcript_list.find_manually_created_transcript([lang])
                        transcript_data = transcript.fetch()
                        logger.info(f"✅ 수동 자막 찾음 (언어: {lang})")
                        break
                    except:
                        continue
            except:
                pass
                
            # 수동 자막 실패 시 자동 생성 자막 시도
            if not transcript_data:
                try:
                    for lang in preferred_langs:
                        try:
                            transcript = transcript_list.find_generated_transcript([lang])
                            transcript_data = transcript.fetch()
                            logger.info(f"✅ 자동 생성 자막 찾음 (언어: {lang})")
                            break
                        except:
                            continue
                except:
                    pass
            
            # 위 방법 실패 시 사용 가능한 첫 번째 자막 시도
            if not transcript_data:
                try:
                    transcript = next(iter(transcript_list._manually_created_transcripts.values()))
                    transcript_data = transcript.fetch()
                    logger.info(f"✅ 기타 수동 자막 찾음 (언어: {transcript.language_code})")
                except:
                    try:
                        transcript = next(iter(transcript_list._generated_transcripts.values()))
                        transcript_data = transcript.fetch()
                        logger.info(f"✅ 기타 자동 자막 찾음 (언어: {transcript.language_code})")
                    except:
                        pass
            
            # 자막 텍스트 추출 및 처리
            if transcript_data:
                # 시간 정보 포함 여부 결정 (기본: 포함하지 않음)
                include_timestamps = False
                
                if include_timestamps:
                    # 시간 정보를 포함한 형식
                    texts = []
                    for item in transcript_data:
                        start_time = format_timestamp(item['start'])
                        text = item['text'].strip()
                        texts.append(f"[{start_time}] {text}")
                    return '\n'.join(texts)
                else:
                    # 시간 정보 없이 텍스트만
                    texts = [item['text'].strip() for item in transcript_data]
                    
                    # 자막 후처리 - 문장 완성 및 정리
                    processed_text = process_transcript_text(texts)
                    return processed_text
        
        except ImportError:
            logger.warning("⚠️ youtube_transcript_api 라이브러리가 설치되지 않았습니다. 웹 파싱을 시도합니다.")
        except Exception as e:
            logger.warning(f"⚠️ youtube_transcript_api로 자막 추출 실패: {e}")
        
        # 방법 2: 웹 페이지 파싱 시도
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = session.get(f"https://www.youtube.com/watch?v={video_id}", headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 자막 데이터 찾기 (복잡하고 변경 가능성 높음)
            scripts = soup.find_all('script')
            transcript_text = ""
            
            for script in scripts:
                if script.string and '"captionTracks"' in script.string:
                    # 자막 URL 추출 시도
                    caption_match = re.search(r'"captionTracks":\s*(\[.*?\])', script.string)
                    if caption_match:
                        captions_data = json.loads(caption_match.group(1))
                        for item in captions_data:
                            if 'baseUrl' in item:
                                caption_url = item['baseUrl']
                                caption_response = session.get(caption_url)
                                
                                # XML 파싱 (간소화 버전)
                                caption_soup = BeautifulSoup(caption_response.text, 'xml')
                                texts = [text.get_text() for text in caption_soup.find_all('text')]
                                transcript_text = '\n'.join(texts)
                                
                                # 자막 후처리
                                transcript_text = process_transcript_text(texts)
                                break
                        if transcript_text:
                            break
            
            if transcript_text:
                logger.info("✅ 웹 파싱을 통해 자막 추출 성공")
                return transcript_text
            
        except Exception as e:
            logger.warning(f"⚠️ 웹 파싱을 통한 자막 추출 실패: {e}")
        
        # 모든 방법 실패
        logger.warning("❌ 모든 자막 추출 방법 실패")
        return ""
        
    except Exception as e:
        logger.error(f"❌ 자막 추출 중 예외 발생: {str(e)}")
        return ""

def process_transcript_text(texts: List[str]) -> str:
    """자막 텍스트 후처리 - 문장 완성 및 포맷팅"""
    if not texts:
        return ""
    
    # 문장 조합
    combined_text = " ".join(texts)
    
    # 중복 공백 제거
    combined_text = re.sub(r'\s+', ' ', combined_text)
    
    # 문장 구분
    sentences = re.split(r'(?<=[.!?])\s+', combined_text)
    
    # 각 문장 첫 글자 대문자로 변환
    sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
    
    # 결과 조합
    result = "\n".join(sentences)
    
    return result

def transcribe_with_whisper(audio_path: str, model_size: str = "base") -> str:
    """Whisper 모델을 사용하여 오디오 파일에서 음성 인식"""
    try:
        logger.info(f"🎤 Whisper 음성 인식 중... (모델: {model_size})")
        
        # 모델 있는지 확인
        try:
            import whisper
        except ImportError:
            logger.error("❌ Whisper 라이브러리가 설치되지 않았습니다.")
            return "음성 인식 라이브러리(Whisper)가 설치되지 않았습니다."
        
        # Whisper 모델 로드
        model = whisper.load_model(model_size)
        
        # 음성 인식 수행
        def run_whisper():
            result = model.transcribe(
                audio_path,
                language=None,  # 자동 감지
                task="transcribe",
                verbose=False
            )
            return result["text"]
            
        # 재시도 로직 적용
        transcription = api_call_with_retry(run_whisper)
        
        return transcription
        
    except Exception as e:
        logger.error(f"❌ Whisper 음성 인식 오류: {str(e)}")
        return f"음성 인식 오류: {str(e)}"

def extract_video_id(url: str) -> Optional[str]:
    """유튜브 URL에서 비디오 ID 추출 - 다양한 URL 형식 지원"""
    if not url:
        return None
        
    # URL 파싱
    parsed_url = urlparse(url)
    
    # youtu.be 링크 (단축 URL)
    if parsed_url.netloc == 'youtu.be':
        return parsed_url.path.strip('/')
    
    # 일반 youtube.com 링크
    if parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
        # watch 페이지
        if parsed_url.path == '/watch':
            query = parse_qs(parsed_url.query)
            if 'v' in query:
                return query['v'][0]
        
        # shorts 페이지
        elif '/shorts/' in parsed_url.path:
            parts = parsed_url.path.split('/')
            if len(parts) >= 3:
                return parts[2]
        
        # embed 페이지
        elif parsed_url.path.startswith('/embed/'):
            return parsed_url.path.split('/')[2]
    
    # 정규식으로 마지막 시도
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*?v=([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/v\/([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def format_timestamp(seconds: float) -> str:
    """초를 mm:ss 형식의 타임스탬프로 변환"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def cleanup_temp_files(output_dir: str, video_id: str) -> None:
    """임시 파일 정리"""
    try:
        audio_path = os.path.join(output_dir, f"{video_id}.mp3")
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"✅ 임시 파일 삭제: {audio_path}")
    except Exception as e:
        logger.warning(f"⚠️ 임시 파일 정리 중 오류: {str(e)}")

if __name__ == "__main__":
    # 테스트 코드
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # 샘플 영상
    logger.info(f"테스트 URL: {test_url}")
    
    video_id = extract_video_id(test_url)
    logger.info(f"추출된 비디오 ID: {video_id}")
    
    content = parse_youtube(test_url)
    logger.info("\n=== 추출된 콘텐츠 일부 ===")
    print(content[:500] + "..." if len(content) > 500 else content)
    
    # 추가 URL 포맷 테스트
    test_urls = [
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ"
    ]
    
    logger.info("\n=== 다양한 URL 포맷 테스트 ===")
    for url in test_urls:
        vid_id = extract_video_id(url)
        logger.info(f"URL: {url} -> 비디오 ID: {vid_id}")