# coding: utf-8  # 파일 인코딩을 UTF-8로 설정
# 필요한 패키지를 가져옴
from imutils import face_utils  # 얼굴 랜드마크 처리를 위한 imutils의 유틸리티
import numpy as np  # NumPy 라이브러리 (배열 연산용)
import dlib  # dlib 라이브러리 (얼굴 감지 및 랜드마크 추출용)
import cv2  # OpenCV 라이브러리 (이미지 처리용)
import matplotlib.pyplot as plt  # Matplotlib 라이브러리 (시각화용, 여기서는 사용되지 않음)

class DetectFace:  # DetectFace라는 클래스를 정의
    # 이미지 입력을 처리하고, dlib의 얼굴 감지기와 랜드마크 예측기를 초기화한 뒤 얼굴 부위를 감지합니다.
    def __init__(self, image):  # 클래스 초기화 메서드, 이미지를 인자로 받음
        # dlib의 HOG 기반 얼굴 감지기 초기화
        # 그리고 얼굴 랜드마크 예측기를 생성
        self.detector = dlib.get_frontal_face_detector()  # 얼굴 감지기 객체 생성
        self.predictor = dlib.shape_predictor('../res/shape_predictor_68_face_landmarks.dat')  # 68개 랜드마크 예측기 로드

        # 얼굴 감지 부분
        if isinstance(image, str):  # 입력이 파일 경로(문자열)인 경우
            self.img = cv2.imread(image)  # 이미지 파일을 읽음
            if self.img is None:  # 이미지가 로드되지 않은 경우
                raise ValueError(f"Image file not found at path: {image}")  # 오류 발생
        elif isinstance(image, np.ndarray):  # 입력이 NumPy 배열인 경우
            self.img = image  # 이미지를 그대로 사용
        else:  # 그 외의 경우
            raise ValueError("Input must be a file path or a NumPy array")  # 오류 발생

        # 이미지 크기가 500pxを超える場合、リサイズする (주석 처리됨)
        #if self.img.shape[0] > 500:
        #    self.img = cv2.resize(self.img, dsize=(0,0), fx=0.8, fy=0.8)  # 이미지 크기를 80%로 축소

        # 얼굴 부위 변수 초기화
        self.right_eyebrow = []  # 오른쪽 눈썹 좌표 리스트
        self.left_eyebrow = []  # 왼쪽 눈썹 좌표 리스트
        self.right_eye = []  # 오른쪽 눈 좌표 리스트
        self.left_eye = []  # 왼쪽 눈 좌표 리스트
        self.left_cheek = []  # 왼쪽 뺨 이미지 데이터
        self.right_cheek = []  # 오른쪽 뺨 이미지 데이터

        # 얼굴 부위를 감지하고 변수에 설정
        self.detect_face_part()  # 얼굴 부위 감지 메서드 호출

    # 얼굴을 감지하고, 68개의 랜드마크를 추출하여 눈썹, 눈, 뺨을 변수에 저장합니다.
    # 반환 타입: np.array
    def detect_face_part(self):  # 얼굴 부위를 감지하고 추출하는 메서드
        face_parts = [[] for _ in range(len(face_utils.FACIAL_LANDMARKS_IDXS))]  # 얼굴 부위별 좌표를 저장할 리스트 초기화
        # 그레이스케일 이미지에서 얼굴 감지
        rect = self.detector(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), 1)[0]  # 첫 번째 얼굴 영역 감지

        # 얼굴 영역에서 랜드마크를 예측하고, 좌표를 NumPy 배열로 변환
        shape = self.predictor(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), rect)  # 랜드마크 예측
        shape = face_utils.shape_to_np(shape)  # 랜드마크를 NumPy 배열로 변환

        idx = 0  # 인덱스 초기화
        # 얼굴 부위를 개별적으로 반복 처리
        print(face_utils.FACIAL_LANDMARKS_IDXS.items())  # 랜드마크 인덱스 사전 출력 (디버깅용)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():  # 각 얼굴 부위에 대해 반복
            face_parts[idx] = shape[i:j]  # 해당 부위의 좌표를 저장
            idx += 1  # 인덱스 증가
        face_parts = face_parts[1:5]  # 필요한 부위(눈썹, 눈)만 선택 (1~4번 인덱스)

        # 변수에 값 설정
        # 주의: 이 좌표는 리사이즈된 이미지에 맞춰짐 (리사이즈가 주석 처리되어 있으므로 원본 기준)
        self.right_eyebrow = self.extract_face_part(face_parts[0])  # 오른쪽 눈썹 추출
        #cv2.imshow("right_eyebrow", self.right_eyebrow)  # 오른쪽 눈썹 이미지 표시 (주석 처리됨)
        #cv2.waitKey(0)  # 키 입력 대기 (주석 처리됨)
        self.left_eyebrow = self.extract_face_part(face_parts[1])  # 왼쪽 눈썹 추출
        self.right_eye = self.extract_face_part(face_parts[2])  # 오른쪽 눈 추출
        self.left_eye = self.extract_face_part(face_parts[3])  # 왼쪽 눈 추출
        # 뺨은 랜드마크의 상대적 위치로 감지
        self.left_cheek = self.img[shape[29][1]:shape[33][1], shape[4][0]:shape[48][0]]  # 왼쪽 뺨 이미지 추출
        self.right_cheek = self.img[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]]  # 오른쪽 뺨 이미지 추출

    # 특정 부위의 경계 사각형을 계산하고, 해당 영역을 자른 뒤 마스크를 적용해 부위만 남기고 배경을 빨간색으로 채웁니다.
    # 매개변수 예시: self.right_eye
    # 반환 타입: 이미지
    def extract_face_part(self, face_part_points):  # 얼굴 부위 이미지를 추출하는 메서드
        (x, y, w, h) = cv2.boundingRect(face_part_points)  # 부위의 경계 사각형 계산 (x, y, 너비, 높이)
        crop = self.img[y:y+h, x:x+w]  # 경계 사각형 영역을 이미지에서 자름
        adj_points = np.array([np.array([p[0]-x, p[1]-y]) for p in face_part_points])  # 좌표를 상대 좌표로 조정

        # 마스크 생성
        mask = np.zeros((crop.shape[0], crop.shape[1]))  # 자른 이미지 크기의 마스크 초기화
        cv2.fillConvexPoly(mask, adj_points, 1)  # 조정된 좌표로 다각형 영역 채움 (1로 설정)
        mask = mask.astype(np.bool)  # 마스크를 불리언 타입으로 변환
        crop[np.logical_not(mask)] = [255, 0, 0]  # 마스크 외부 영역을 빨간색으로 설정

        return crop  # 추출된 이미지 반환