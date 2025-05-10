import os
import re
import time
import logging
import json
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import lru_cache
import concurrent.futures
import requests
import io
import base64
import uuid
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CloudOCRProcessor:
    """여러 클라우드 OCR 서비스를 활용한 학술 이미지 처리 클래스"""
    
    def __init__(self, 
                 ocr_engine: str = "google",  # 'google', 'aws', 'azure', 'naver'
                 api_key: Optional[str] = None,
                 api_endpoint: Optional[str] = None,
                 api_region: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 languages: List[str] = ['en', 'ko'],
                 dpi: int = 300,
                 temp_dir: str = 'temp_ocr'):
        """
        초기화
        
        Args:
            ocr_engine: 사용할 OCR 엔진 ('google', 'aws', 'azure', 'naver')
            api_key: API 키 또는 액세스 키
            api_endpoint: API 엔드포인트 URL (필요한 경우)
            api_region: API 리전 (필요한 경우)
            api_secret: API 시크릿 키 (필요한 경우)
            languages: 인식할 언어 목록
            dpi: OCR 처리 DPI
            temp_dir: 임시 파일 저장 경로
        """
        self.ocr_engine = ocr_engine.lower()
        self.api_key = api_key
        self.api_endpoint = api_endpoint
        self.api_region = api_region
        self.api_secret = api_secret
        self.languages = languages
        self.dpi = dpi
        self.temp_dir = temp_dir
        self.client = None
        self.available_engines = []  # 사용 가능한 엔진 목록
        
        # 임시 디렉터리 생성
        os.makedirs(temp_dir, exist_ok=True)
        
        # API 키 환경 변수에서 로드 (제공되지 않은 경우)
        if not self.api_key:
            self._load_api_keys_from_env()

        # 사용 가능한 엔진 확인
        self._check_available_engines()    
        
        # OCR 엔진 검증
        supported_engines = ["google", "aws", "azure", "naver"]
        if self.ocr_engine not in supported_engines:
            logger.warning(f"⚠️ 지원하지 않는 OCR 엔진: {self.ocr_engine}. 기본값 'google'로 설정합니다.")
            self.ocr_engine = "google"

        # 선택한 엔진이 사용 가능하지 않은 경우 사용 가능한 첫 번째 엔진으로 대체
        if self.ocr_engine not in self.available_engines:
            if self.available_engines:
                logger.warning(f"⚠️ {self.ocr_engine} 엔진은 사용할 수 없습니다. {self.available_engines[0]}(으)로 대체합니다.")
                self.ocr_engine = self.available_engines[0]
            else:
                logger.error("❌ 사용 가능한 OCR 엔진이 없습니다. API 키를 확인하세요.")            
        
        logger.info(f"✅ OCR 엔진 설정: {self.ocr_engine}")
        
        # 특정 엔진별 클라이언트 초기화
        if self.ocr_engine == "google" and "google" in self.available_engines:
            self._init_google_vision()
        elif self.ocr_engine == "aws" and "aws" in self.available_engines:
            self._init_aws_textract()
        elif self.ocr_engine == "azure" and "azure" in self.available_engines:
            self._init_azure_document_intelligence()
        elif self.ocr_engine == "naver" and "naver" in self.available_engines:
            self._init_naver_clova()

    def _check_available_engines(self):
        """사용 가능한 OCR 엔진 확인"""
        # Google Vision
        if self.api_key or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            try:
                from google.cloud import vision
                self.available_engines.append("google")
            except ImportError:
                logger.warning("⚠️ Google Cloud Vision 패키지가 설치되지 않았습니다.")
        
        # AWS Textract
        if (self.api_key and self.api_secret) or (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")):
            try:
                import boto3
                self.available_engines.append("aws")
            except ImportError:
                logger.warning("⚠️ Boto3 패키지가 설치되지 않았습니다.")
        
        # Azure Document Intelligence
        if self.api_key and self.api_endpoint:
            try:
                from azure.ai.formrecognizer import DocumentAnalysisClient
                self.available_engines.append("azure")
            except ImportError:
                logger.warning("⚠️ Azure AI Form Recognizer 패키지가 설치되지 않았습니다.")
        
        # Naver Clova OCR
        if self.api_key and self.api_secret:
            self.available_engines.append("naver")
        
        logger.info(f"✅ 사용 가능한 OCR 엔진: {', '.join(self.available_engines) if self.available_engines else '없음'}")

    def _load_api_keys_from_env(self):
        """환경 변수에서 API 키 로드"""
        env_keys = {
            "google": "GOOGLE_VISION_API_KEY",
            "aws_key": "AWS_ACCESS_KEY_ID",
            "aws_secret": "AWS_SECRET_ACCESS_KEY",
            "aws_region": "AWS_REGION",
            "azure": "AZURE_DOCUMENT_API_KEY",
            "azure_endpoint": "AZURE_DOCUMENT_ENDPOINT",
            "naver_key": "NAVER_OCR_API_KEY",
            "naver_secret": "NAVER_OCR_SECRET_KEY"
        }
        
        if self.ocr_engine == "google":
            self.api_key = os.getenv(env_keys["google"])
        elif self.ocr_engine == "aws":
            self.api_key = os.getenv(env_keys["aws_key"])
            self.api_secret = os.getenv(env_keys["aws_secret"])
            self.api_region = os.getenv(env_keys["aws_region"], "us-east-1")
        elif self.ocr_engine == "azure":
            self.api_key = os.getenv(env_keys["azure"])
            self.api_endpoint = os.getenv(env_keys["azure_endpoint"])
        elif self.ocr_engine == "naver":
            self.api_key = os.getenv(env_keys["naver_key"])
            self.api_secret = os.getenv(env_keys["naver_secret"])
    
    def _init_google_vision(self):
        """Google Vision API 클라이언트 초기화"""
        try:
            # Google Cloud Vision 사용 시 필요한 패키지 확인
            from google.cloud import vision
            from google.oauth2 import service_account
            
            # API 키가 없으면 credential 파일 확인
            if not self.api_key and os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                self.client = vision.ImageAnnotatorClient()
                logger.info("✅ Google Vision API 클라이언트 초기화 완료 (서비스 계정 사용)")
            elif self.api_key:
                # API 키가 JSON 형식이면 파일로 저장하여 사용
                if self.api_key.startswith('{'):
                    temp_credential_path = os.path.join(self.temp_dir, 'google_credentials.json')
                    with open(temp_credential_path, 'w') as f:
                        f.write(self.api_key)
                    credentials = service_account.Credentials.from_service_account_file(temp_credential_path)
                    self.client = vision.ImageAnnotatorClient(credentials=credentials)
                else:
                    # API 키만 있는 경우 (비표준 인증)
                    self.client = None
                    self.api_endpoint = "https://vision.googleapis.com/v1/images:annotate"
                logger.info("✅ Google Vision API 키 설정 완료")
            else:
                logger.warning("⚠️ Google Vision API 키 또는 서비스 계정 자격 증명이 필요합니다.")
                self.client = None
        except ImportError:
            logger.warning("⚠️ Google Cloud Vision 패키지가 설치되지 않았습니다. pip install google-cloud-vision 명령으로 설치하세요.")
            self.client = None
    
    def _init_aws_textract(self):
        """AWS Textract 클라이언트 초기화"""
        try:
            import boto3
            
            if self.api_key and self.api_secret:
                self.client = boto3.client(
                    'textract',
                    region_name=self.api_region,
                    aws_access_key_id=self.api_key,
                    aws_secret_access_key=self.api_secret
                )
                logger.info(f"✅ AWS Textract 클라이언트 초기화 완료 (리전: {self.api_region})")
            else:
                # AWS 자격 증명이 환경 변수 또는 설정 파일에 있는 경우
                self.client = boto3.client('textract', region_name=self.api_region or 'us-east-1')
                logger.info(f"✅ AWS Textract 클라이언트 초기화 완료 (기본 자격 증명 사용)")
        except ImportError:
            logger.warning("⚠️ Boto3 패키지가 설치되지 않았습니다. pip install boto3 명령으로 설치하세요.")
            self.client = None
        except Exception as e:
            logger.error(f"❌ AWS Textract 클라이언트 초기화 실패: {str(e)}")
            self.client = None
    
    def _init_azure_document_intelligence(self):
        """Azure Document Intelligence 클라이언트 초기화"""
        try:
            from azure.ai.formrecognizer import DocumentAnalysisClient
            from azure.core.credentials import AzureKeyCredential
            
            if self.api_key and self.api_endpoint:
                self.client = DocumentAnalysisClient(
                    endpoint=self.api_endpoint,
                    credential=AzureKeyCredential(self.api_key)
                )
                logger.info("✅ Azure Document Intelligence 클라이언트 초기화 완료")
            else:
                logger.warning("⚠️ Azure Document Intelligence API 키와 엔드포인트가 필요합니다.")
                self.client = None
        except ImportError:
            logger.warning("⚠️ Azure AI Form Recognizer 패키지가 설치되지 않았습니다. pip install azure-ai-formrecognizer 명령으로 설치하세요.")
            self.client = None
        except Exception as e:
            logger.error(f"❌ Azure Document Intelligence 클라이언트 초기화 실패: {str(e)}")
            self.client = None
    
    def _init_naver_clova(self):
        """Naver Clova OCR 설정 초기화"""
        # Naver Clova는 REST API 사용하므로 별도 클라이언트 초기화 불필요
        if self.api_key and self.api_secret:
            logger.info("✅ Naver Clova OCR API 키 설정 완료")
            self.api_endpoint = "https://naveropenapi.apigw.ntruss.com/vision/v1/ocr"
        else:
            logger.warning("⚠️ Naver Clova OCR API 키와 시크릿이 필요합니다.")
    
    def parse_image(self, 
                   image_path: str, 
                   preprocess: bool = True,
                   ocr_level: str = 'advanced') -> str:
        """
        이미지에서 텍스트 추출 - 논문 특화 처리
        
        Args:
            image_path: 이미지 파일 경로
            preprocess: 이미지 전처리 사용 여부
            ocr_level: OCR 처리 수준 ('basic', 'advanced')
            
        Returns:
            추출된 텍스트
        """
        try:
            logger.info(f"🖼️ 이미지 OCR 처리 시작 ({self.ocr_engine}): {os.path.basename(image_path)}")
            
            # 선택한 엔진이 사용 가능한지 확인
            if not self.available_engines:
                return "OCR 처리를 위한 사용 가능한 엔진이 없습니다. API 키를 확인하세요."
            
            if self.ocr_engine not in self.available_engines:
                return f"{self.ocr_engine} 엔진은 사용할 수 없습니다. 사용 가능한 엔진: {', '.join(self.available_engines)}"

            # 이미지 로드
            image = self._load_image(image_path)
            
            # 이미지 전처리
            if preprocess:
                processed_image = self._preprocess_image(image)
            else:
                processed_image = image
            
            # 임시 파일로 저장 (API 요청용)
            temp_img_path = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.png")
            processed_image.save(temp_img_path)
            
            # OCR 실행
            if self.ocr_engine == "google":
                extracted_text = self._process_with_google_vision(temp_img_path)
            elif self.ocr_engine == "aws":
                extracted_text = self._process_with_aws_textract(temp_img_path)
            elif self.ocr_engine == "azure":
                extracted_text = self._process_with_azure(temp_img_path)
            elif self.ocr_engine == "naver":
                extracted_text = self._process_with_naver_clova(temp_img_path)
            else:
                extracted_text = "지원하지 않는 OCR 엔진입니다."
            
            # 임시 파일 삭제
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
            
            # 학술 텍스트 후처리
            if extracted_text:
                extracted_text = self._post_process_academic(extracted_text)
            
            # 파일 정보 추가
            file_info = f"파일명: {os.path.basename(image_path)}\n"
            file_info += f"OCR 엔진: {self.ocr_engine.upper()}\n"
            file_info += f"크기: {os.path.getsize(image_path) / 1024:.1f} KB\n"
            file_info += f"해상도: {processed_image.width}x{processed_image.height}\n\n"
            
            result = file_info + extracted_text
            logger.info(f"✅ {self.ocr_engine.upper()} OCR 완료: {os.path.basename(image_path)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {self.ocr_engine.upper()} OCR 처리 오류 ({image_path}): {str(e)}")
            return f"이미지 파싱 오류: {image_path}\n오류 세부사항: {str(e)}"
    
    def _load_image(self, image_path: str) -> Image.Image:
        """이미지 로드 및 기본 검증"""
        try:
            img = Image.open(image_path)
            
            # 이미지가 너무 크면 처리 속도를 위해 리사이징
            max_dimension = 4000
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
                logger.info(f"🔄 큰 이미지 리사이징: {img.size}")
            
            return img
        except Exception as e:
            logger.error(f"❌ 이미지 로드 실패: {str(e)}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """학술 문서 이미지 전처리 - 논문 가독성 향상"""
        try:
            # 원본 이미지 복사
            img = image.copy()
            
            # 그레이스케일 변환 (컬러 이미지인 경우)
            if img.mode != 'L':
                img = img.convert('L')
            
            # 노이즈 제거
            img = img.filter(ImageFilter.MedianFilter(3))
            
            # 대비 향상
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # 선명도 증가
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)
            
            # 이진화 (적응형 임계값 - OpenCV 사용)
            img_np = np.array(img)
            _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 노이즈 제거를 위한 모폴로지 연산
            kernel = np.ones((1, 1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # NumPy 배열을 PIL 이미지로 변환
            processed_img = Image.fromarray(binary)
            
            return processed_img
        except Exception as e:
            logger.warning(f"⚠️ 이미지 전처리 실패, 원본 이미지 사용: {str(e)}")
            return image
    
    def _process_with_google_vision(self, image_path: str) -> str:
        """Google Cloud Vision API를 사용한 OCR"""
        try:
            # 클라이언트 객체가 있는 경우
            if hasattr(self, 'client') and self.client:
                from google.cloud import vision
                
                # 이미지 파일 열기
                with open(image_path, 'rb') as image_file:
                    content = image_file.read()
                
                image = vision.Image(content=content)
                
                # OCR 실행
                response = self.client.document_text_detection(image=image)
                
                if response.error.message:
                    raise Exception(f"Google Vision API 오류: {response.error.message}")
                
                # 전체 텍스트 추출
                texts = response.text_annotations
                if texts:
                    return texts[0].description
                return ""
            
            # API 키만 있는 경우 REST API 직접 호출
            elif self.api_key and not self.api_key.startswith('{'):
                with open(image_path, 'rb') as image_file:
                    image_content = base64.b64encode(image_file.read()).decode('utf-8')
                
                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'X-Goog-Api-Key': self.api_key
                }
                
                data = {
                    'requests': [
                        {
                            'image': {'content': image_content},
                            'features': [{'type': 'DOCUMENT_TEXT_DETECTION'}],
                            'imageContext': {
                                'languageHints': self.languages
                            }
                        }
                    ]
                }
                
                response = requests.post(
                    self.api_endpoint,
                    headers=headers,
                    json=data
                )
                
                if response.status_code != 200:
                    raise Exception(f"Google Vision API 오류: {response.status_code} - {response.text}")
                
                result = response.json()
                
                if 'responses' in result and result['responses'] and 'fullTextAnnotation' in result['responses'][0]:
                    return result['responses'][0]['fullTextAnnotation']['text']
                return ""
            
            else:
                raise Exception("Google Vision API 클라이언트와 API 키가 설정되지 않았습니다.")
                
        except ImportError:
            raise ImportError("Google Cloud Vision 패키지가 설치되지 않았습니다. pip install google-cloud-vision 명령으로 설치하세요.")
        except Exception as e:
            logger.error(f"❌ Google Vision OCR 오류: {str(e)}")
            raise
    
    def _process_with_aws_textract(self, image_path: str) -> str:
        """AWS Textract를 사용한 OCR"""
        try:
            import boto3
            
            if not hasattr(self, 'client') or not self.client:
                raise Exception("AWS Textract 클라이언트가 초기화되지 않았습니다.")
            
            # 이미지 파일 읽기
            with open(image_path, 'rb') as image_file:
                bytes_data = image_file.read()
            
            # Textract API 호출
            response = self.client.detect_document_text(Document={'Bytes': bytes_data})
            
            # 응답에서 텍스트 추출
            text_blocks = []
            for item in response['Blocks']:
                if item['BlockType'] == 'LINE':
                    text_blocks.append(item['Text'])
            
            return '\n'.join(text_blocks)
            
        except ImportError:
            raise ImportError("Boto3 패키지가 설치되지 않았습니다. pip install boto3 명령으로 설치하세요.")
        except Exception as e:
            logger.error(f"❌ AWS Textract OCR 오류: {str(e)}")
            raise
    
    def _process_with_azure(self, image_path: str) -> str:
        """Azure Document Intelligence를 사용한 OCR"""
        try:
            if not hasattr(self, 'client') or not self.client:
                raise Exception("Azure Document Intelligence 클라이언트가 초기화되지 않았습니다.")
            
            # 이미지 파일 열기
            with open(image_path, "rb") as f:
                poller = self.client.begin_analyze_document("prebuilt-read", document=f)
            
            # 결과 가져오기
            result = poller.result()
            
            # 텍스트 추출
            extracted_text = []
            for page in result.pages:
                for line in page.lines:
                    extracted_text.append(line.content)
            
            return '\n'.join(extracted_text)
            
        except ImportError:
            raise ImportError("Azure AI Form Recognizer 패키지가 설치되지 않았습니다. pip install azure-ai-formrecognizer 명령으로 설치하세요.")
        except Exception as e:
            logger.error(f"❌ Azure Document Intelligence OCR 오류: {str(e)}")
            raise
    
    def _process_with_naver_clova(self, image_path: str) -> str:
        """Naver Clova OCR을 사용한 OCR"""
        try:
            if not self.api_key or not self.api_secret:
                raise Exception("Naver Clova OCR API 키와 시크릿이 설정되지 않았습니다.")
            
            # 이미지 파일을 base64로 인코딩
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode('utf-8')
            
            # API 요청 헤더
            headers = {
                "X-OCR-SECRET": self.api_secret,
                "Content-Type": "application/json"
            }
            
            # API 요청 데이터
            data = {
                "images": [
                    {
                        "format": os.path.splitext(image_path)[1][1:].lower(),
                        "name": "image",
                        "data": img_base64
                    }
                ],
                "requestId": str(uuid.uuid4()),
                "version": "V2",
                "timestamp": int(round(time.time() * 1000))
            }
            
            # API 요청
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            
            if response.status_code != 200:
                raise Exception(f"Naver Clova OCR API 오류: {response.status_code} - {response.text}")
            
            # 응답 파싱
            result = response.json()
            
            # 텍스트 추출
            extracted_text = []
            for image in result.get('images', []):
                for field in image.get('fields', []):
                    extracted_text.append(field.get('inferText', ''))
            
            return '\n'.join(extracted_text)
            
        except Exception as e:
            logger.error(f"❌ Naver Clova OCR 오류: {str(e)}")
            raise
    
    def _post_process_academic(self, text: str) -> str:
        """논문 텍스트 후처리"""
        if not text:
            return ""
        
        # 1. 일반적인 OCR 오류 수정
        corrections = {
            # 학술 용어 관련 오류
            'Tne': 'The',
            'ana': 'and',
            'tne': 'the',
            'Fig,': 'Fig.',
            'Tab,': 'Tab.',
            'et a1': 'et al',
            'et a1.': 'et al.',
            
            # 참조 관련 오류
            '[1 ]': '[1]',
            '[ 1]': '[1]',
            '[ 1 ]': '[1]',
            '(1 )': '(1)',
            '( 1)': '(1)',
            '( 1 )': '(1)',
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        # 2. 학술 용어 및 수식 처리
        # 수식 패턴 (예: a^2 + b^2 = c^2)
        text = re.sub(r'([a-zA-Z])(\s*)\^(\s*)(\d+)', r'\1^\4', text)  # 띄어쓰기 없는 지수 표기
        
        # 3. 참조 형식 통일 ([1], [2], 등)
        text = re.sub(r'\[(\d+)\s*\]', r'[\1]', text)  # [1 ] -> [1]
        
        # 4. 불필요한 줄바꿈 정리
        text = re.sub(r'([^\n])\n([a-z])', r'\1 \2', text)  # 문장 중간의 줄바꿈 제거
        text = re.sub(r'\n{3,}', '\n\n', text)  # 3개 이상 연속 줄바꿈 정리
        
        # 5. 단어 경계 수정 (공백 누락)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        return text

def parse_cloud_ocr(path: str, engine: str = "google") -> str:
    """
    클라우드 OCR 서비스를 사용하여 이미지에서 텍스트 추출
    
    Args:
        path: 이미지 파일 경로
        engine: 사용할 OCR 엔진 ('google', 'aws', 'azure', 'naver')
        
    Returns:
        추출된 텍스트
    """
    try:
        # OCR 처리기 초기화
        processor = CloudOCRProcessor(
            ocr_engine=engine,
            languages=['en', 'ko'],
            temp_dir="temp_ocr"
        )
        
        # 이미지 처리
        result = processor.parse_image(
            path,
            preprocess=True,
            ocr_level='advanced'
        )
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 클라우드 OCR 파싱 오류 ({path}): {str(e)}")
        return f"이미지 파싱 오류: {path}\n오류 세부사항: {str(e)}"