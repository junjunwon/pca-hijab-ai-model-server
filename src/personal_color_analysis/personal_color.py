import cv2  # OpenCV 라이브러리 (이미지 처리용)
import numpy as np  # NumPy 라이브러리 (배열 연산용)
from personal_color_analysis import tone_analysis  # 톤 분석 모듈 가져옴
from personal_color_analysis.detect_face import DetectFace  # 얼굴 감지 클래스 가져옴
from personal_color_analysis.color_extract import DominantColors  # 지배적 색상 추출 클래스 가져옴
from colormath.color_objects import LabColor, sRGBColor, HSVColor  # 색상 객체 가져옴
from colormath.color_conversions import convert_color  # 색상 변환 함수 가져옴

##################################################################################################
# 목적: 이 함수는 이미지에서 얼굴 부위(뺨, 눈썹, 눈)의 지배적 색상을 분석하고, 이를 기반으로 퍼스널 컬러 톤을 결정합니다.
# 주요 동작:
# 얼굴 감지: DetectFace 클래스를 사용해 이미지에서 뺨, 눈썹, 눈을 추출.
# 지배적 색상 추출: DominantColors 클래스를 사용해 각 부위의 지배적 색상을 추출하고, 양쪽 부위의 색상 평균을 계산.
# 색상 변환: RGB 색상을 Lab과 HSV 색상 공간으로 변환하여 b값(황색-청색)과 S값(채도)을 추출.
# 퍼스널 컬러 분석: tone_analysis 모듈의 함수를 사용해 Lab b값과 HSV S값에 가중치를 적용하여 웜톤/쿨톤 및 세부 톤(봄, 가을, 여름, 겨울)을 판단.
##################################################################################################
def analysis(imgpath):  # 이미지 경로를 받아 퍼스널 컬러를 분석하는 함수
    #######################################
    #           얼굴 감지                 #
    #######################################
    df = DetectFace(imgpath)  # DetectFace 클래스로 얼굴 감지 객체 생성
    face = [df.left_cheek, df.right_cheek,  # 얼굴 부위(왼쪽 뺨, 오른쪽 뺨,
            df.left_eyebrow, df.right_eyebrow,  # 왼쪽 눈썹, 오른쪽 눈썹,
            df.left_eye, df.right_eye]  # 왼쪽 눈, 오른쪽 눈)를 리스트로 저장

    #######################################
    #         지배적 색상 추출            #
    #######################################
    temp = []  # 각 부위의 지배적 색상을 저장할 임시 리스트
    clusters = 4  # 클러스터 수를 4로 설정
    for f in face:  # 각 얼굴 부위에 대해 반복
        dc = DominantColors(f, clusters)  # DominantColors 클래스로 색상 추출
        face_part_color, _ = dc.getHistogram()  # 지배적 색상과 히스토그램 가져옴 (히스토그램은 사용 안 함)
        #dc.plotHistogram()  # 히스토그램 표시 (주석 처리됨)
        temp.append(np.array(face_part_color[0]))  # 첫 번째 지배적 색상을 리스트에 추가
    cheek = np.mean([temp[0], temp[1]], axis=0)  # 양쪽 뺨 색상의 평균 계산
    eyebrow = np.mean([temp[2], temp[3]], axis=0)  # 양쪽 눈썹 색상의 평균 계산
    eye = np.mean([temp[4], temp[5]], axis=0)  # 양쪽 눈 색상의 평균 계산

    Lab_b, hsv_s = [], []  # Lab 색상의 b값과 HSV의 S값을 저장할 리스트
    color = [cheek, eyebrow, eye]  # 뺨, 눈썹, 눈의 평균 색상 리스트
    for i in range(3):  # 각 부위 색상에 대해 반복
        rgb = sRGBColor(color[i][0], color[i][1], color[i][2], is_upscaled=True)  # RGB 색상 객체 생성 (0-255 스케일)
        lab = convert_color(rgb, LabColor, through_rgb_type=sRGBColor)  # RGB를 Lab 색상으로 변환
        hsv = convert_color(rgb, HSVColor, through_rgb_type=sRGBColor)  # RGB를 HSV 색상으로 변환
        Lab_b.append(float(format(lab.lab_b, ".2f")))  # Lab의 b값(황색-청색 축)을 소수점 2자리로 저장
        hsv_s.append(float(format(hsv.hsv_s, ".2f")) * 100)  # HSV의 S값(채도)을 백분율로 변환 후 저장

    print('Lab_b[skin, eyebrow, eye]', Lab_b)  # Lab b값 출력 (피부, 눈썹, 눈 순서)
    print('hsv_s[skin, eyebrow, eye]', hsv_s)  # HSV S값 출력 (피부, 눈썹, 눈 순서)
    #######################################
    #      퍼스널 컬러 분석               #
    #######################################
    Lab_weight = [40, 30, 30]  # Lab b값 가중치 (피부: 40, 눈썹: 30, 눈: 30), 웜톤/쿨톤
    hsv_weight = [40, 30, 30]  # HSV S값 가중치 (피부: 40, 눈썹: 30, 눈: 30), 봄/가을, 여름/겨울
    if tone_analysis.is_warm(Lab_b, Lab_weight):  # 웜톤 여부 확인
        if tone_analysis.is_spr(hsv_s, hsv_weight):  # 봄 웜톤 여부 확인
            tone = 'Spring Warm Tone'  # 봄 웜톤으로 판단
        else:
            tone = 'Fall Warm Tone'  # 가을 웜톤으로 판단
    else:  # 쿨톤인 경우
        if tone_analysis.is_smr(hsv_s, hsv_weight):  # 여름 쿨톤 여부 확인
            tone = 'Summer Cool Tone'  # 여름 쿨톤으로 판단
        else:
            tone = 'Winter Cool Tone'  # 겨울 쿨톤으로 판단
    # 결과 출력
    print('{}의 퍼스널 컬러는 {}입니다.'.format(imgpath, tone))  # 이미지 경로와 분석 결과 출력
    # Lab b값과 HSV S값을 출력하며, 최종적으로 이미지의 퍼스널 컬러 톤을 출력하고 반환.
    return tone  # 분석된 톤 반환