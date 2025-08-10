import cv2  # OpenCV 라이브러리를 가져옴 (이미지 처리용)
import numpy as np  # NumPy 라이브러리를 가져옴 (배열 연산용)
from sklearn.cluster import KMeans  # scikit-learn에서 KMeans 클러스터링 알고리즘을 가져옴
import matplotlib.pyplot as plt  # Matplotlib 라이브러리를 가져옴 (시각화용)
from mpl_toolkits.mplot3d import Axes3D  # 3D 플롯을 그리기 위한 Matplotlib 모듈 (여기서는 사용되지 않음)
from skimage import io  # scikit-image에서 이미지 입출력을 위한 모듈 (여기서는 사용되지 않음)
from itertools import compress  # 리스트 필터링을 위한 itertools 모듈의 compress 함수 가져옴

class DominantColors:  # DominantColors라는 클래스를 정의

    CLUSTERS = None  # 클러스터 수를 저장할 클래스 변수 (초기값 None)
    IMAGE = None  # 처리할 이미지를 저장할 클래스 변수 (초기값 None)
    COLORS = None  # 지배적인 색상들을 저장할 클래스 변수 (초기값 None)
    LABELS = None  # 각 픽셀의 클러스터 레이블을 저장할 클래스 변수 (초기값 None)

    # 이미지를 RGB 형식으로 변환하고, KMeans 알고리즘을 사용해 픽셀을 클러스터링합니다. 클러스터 중심이 지배적인 색상이 됩니다.
    def __init__(self, image, clusters=3):  # 클래스 초기화 메서드, 이미지와 클러스터 수를 인자로 받음
        self.CLUSTERS = clusters  # 클러스터 수를 인스턴스 변수로 설정
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR 형식의 이미지를 RGB 형식으로 변환
        self.IMAGE = img.reshape((img.shape[0] * img.shape[1], 3))  # 이미지를 2D 배열로 변환 (픽셀 수 x 3(RGB))

        # K-means를 사용하여 픽셀을 클러스터링
        kmeans = KMeans(n_clusters=self.CLUSTERS)  # KMeans 객체 생성, 클러스터 수 설정
        kmeans.fit(self.IMAGE)  # 이미지 데이터를 클러스터링

        # 클러스터 중심은 지배적인 색상을 나타냄
        self.COLORS = kmeans.cluster_centers_  # 클러스터 중심(지배적인 색상)을 저장
        self.LABELS = kmeans.labels_  # 각 픽셀의 클러스터 레이블을 저장

    # RGB 값을 헥사 코드로 변환합니다 (예: #FF0000은 빨간색).
    def rgb_to_hex(self, rgb):  # RGB 값을 16진수 헥사 코드로 변환하는 메서드
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))  # RGB를 헥사 코드로 변환하여 반환

    # 색상이 가장 많이 나타난 순서대로 리스트를 반환
    # 각 클러스터의 빈도를 계산하고, 빈도순으로 색상을 정렬하며, 특정 조건(Blue < 250, Red > 10)을 통해 색상을 필터링합니다.
    def getHistogram(self):  # 지배적인 색상과 그 비율을 계산하는 메서드
        numLabels = np.arange(0, self.CLUSTERS+1)  # 클러스터 레이블 범위를 생성 (0부터 클러스터 수까지)
        # 빈도 수 테이블 생성
        (hist, _) = np.histogram(self.LABELS, bins=numLabels)  # 레이블의 히스토그램 계산
        hist = hist.astype("float")  # 히스토그램을 실수형으로 변환
        hist /= hist.sum()  # 히스토그램을 정규화 (총합이 1이 되도록)

        colors = self.COLORS  # 지배적인 색상 가져오기
        # 빈도 수에 따라 내림차순 정렬
        colors = colors[(-hist).argsort()]  # 색상을 빈도 수 기준으로 정렬
        hist = hist[(-hist).argsort()]  # 히스토그램도 동일한 순서로 정렬
        for i in range(self.CLUSTERS):  # 각 색상을 정수형으로 변환
            colors[i] = colors[i].astype(int)
        # Blue 값이 250 미만이고 Red 값이 10 초과인 색상만 필터링 (파란색 마스크 제거)
        fil = [colors[i][2] < 250 and colors[i][0] > 10 for i in range(self.CLUSTERS)]
        colors = list(compress(colors, fil))  # 필터링된 색상만 추출
        return colors, hist  # 정렬된 색상과 히스토그램 반환

    # 지배적인 색상을 사각형으로 표시한 히스토그램을 생성하고 시각화합니다.
    def plotHistogram(self):  # 지배적인 색상을 히스토그램으로 시각화하는 메서드
        colors, hist = self.getHistogram()  # 색상과 히스토그램 가져오기
        # 빈 차트 생성 (높이 50, 너비 500, RGB 3채널)
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0  # 사각형 시작 위치 초기화

        # 색상 사각형 생성
        for i in range(len(colors)):  # 각 색상에 대해 반복
            end = start + hist[i] * 500  # 사각형 끝 위치 계산 (비율에 따라 너비 설정)
            r, g, b = colors[i]  # RGB 값 추출
            # cv2.rectangle을 사용하여 색상 사각형 그림
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end  # 다음 사각형의 시작 위치 업데이트

        # 차트 표시
        plt.figure()  # 새 플롯 창 생성
        plt.axis("off")  # 축 숨김
        plt.imshow(chart)  # 차트 이미지 표시
        plt.show()  # 플롯 창 띄우기

        return colors  # 색상 리스트 반환