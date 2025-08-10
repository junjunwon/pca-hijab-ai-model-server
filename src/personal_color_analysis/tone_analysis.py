from scipy.spatial import distance  # 공간 거리 계산 모듈 (여기서는 사용되지 않음)
import copy  # 객체 복사 모듈 (여기서는 사용되지 않음)
import math  # 수학 연산 모듈 (여기서는 사용되지 않음)
import operator  # 연산자 모듈 (여기서는 사용되지 않음)

def is_warm(lab_b, a):  # 웜톤 여부를 판단하는 함수 (웜톤(1) 또는 쿨톤(0) 여부)
    '''
    파라미터 lab_b = [skin_b, hair_b, eye_b]  # 피부, 눈썹, 눈의 Lab b값 리스트
    a = 가중치 [skin, hair, eye]  # 각 부위별 가중치 리스트
    질의색상 lab_b값에서 warm의 lab_b, cool의 lab_b값 간의 거리를
    각각 계산하여 warm이 가까우면 1, 반대 경우 0 리턴
    '''
    # 피부, 눈썹, 눈의 웜톤 및 쿨톤 기준값
    warm_b_std = [11.6518, 11.71445, 3.6484]  # 웜톤 Lab b 기준값
    cool_b_std = [4.64255, 4.86635, 0.18735]  # 쿨톤 Lab b 기준값

    warm_dist = 0  # 웜톤 기준값과의 거리 합 초기화
    cool_dist = 0  # 쿨톤 기준값과의 거리 합 초기화

    body_part = ['skin', 'eyebrow', 'eye']  # 부위 이름 리스트
    for i in range(3):  # 각 부위에 대해 반복
        warm_dist += abs(lab_b[i] - warm_b_std[i]) * a[i]  # 웜톤 기준값과의 가중치 적용 거리 계산
        #print(body_part[i],"의 warm 기준값과의 거리")  # 부위별 웜톤 거리 출력 (주석 처리됨)
        #print(abs(lab_b[i] - warm_b_std[i]))  # 거리 값 출력 (주석 처리됨)
        cool_dist += abs(lab_b[i] - cool_b_std[i]) * a[i]  # 쿨톤 기준값과의 가중치 적용 거리 계산
        #print(body_part[i],"의 cool 기준값과의 거리")  # 부위별 쿨톤 거리 출력 (주석 처리됨)
        #print(abs(lab_b[i] - cool_b_std[i]))  # 거리 값 출력 (주석 처리됨)
    if warm_dist <= cool_dist:  # 웜톤 거리가 쿨톤 거리보다 작거나 같으면
        return 1  # 웜톤으로 판단 (1 반환)
    else:
        return 0  # 쿨톤으로 판단 (0 반환)

def is_spr(hsv_s, a):  # 봄 웜톤 여부를 판단하는 함수 (웜톤일 경우 봄(1) 또는 가을(0) 여부)
    '''
    파라미터 hsv_s = [skin_s, hair_s, eye_s]  # 피부, 눈썹, 눈의 HSV S값 리스트
    a = 가중치 [skin, hair, eye]  # 각 부위별 가중치 리스트
    질의색상 hsv_s값에서 spring의 hsv_s, fall의 hsv_s값 간의 거리를
    각각 계산하여 spring이 가까우면 1, 반대 경우 0 리턴
    '''
    # 피부, 눈썹, 눈의 봄 및 가을 기준값
    spr_s_std = [18.59296, 30.30303, 25.80645]  # 봄 웜톤 HSV S 기준값
    fal_s_std = [27.13987, 39.75155, 37.5]  # 가을 웜톤 HSV S 기준값

    spr_dist = 0  # 봄 기준값과의 거리 합 초기화
    fal_dist = 0  # 가을 기준값과의 거리 합 초기화

    body_part = ['skin', 'eyebrow', 'eye']  # 부위 이름 리스트
    for i in range(3):  # 각 부위에 대해 반복
        spr_dist += abs(hsv_s[i] - spr_s_std[i]) * a[i]  # 봄 기준값과의 가중치 적용 거리 계산
        print(body_part[i],"의 spring 기준값과의 거리")  # 부위별 봄 거리 출력
        print(abs(hsv_s[i] - spr_s_std[i]) * a[i])  # 가중치 적용 거리 값 출력
        fal_dist += abs(hsv_s[i] - fal_s_std[i]) * a[i]  # 가을 기준값과의 가중치 적용 거리 계산
        print(body_part[i],"의 fall 기준값과의 거리")  # 부위별 가을 거리 출력
        print(abs(hsv_s[i] - fal_s_std[i]) * a[i])  # 가중치 적용 거리 값 출력

    if spr_dist <= fal_dist:  # 봄 거리가 가을 거리보다 작거나 같으면
        return 1  # 봄 웜톤으로 판단 (1 반환)
    else:
        return 0  # 가을 웜톤으로 판단 (0 반환)

def is_smr(hsv_s, a):  # 여름 쿨톤 여부를 판단하는 함수 (쿨톤일 경우 여름(1) 또는 겨울(0) 여부)
    '''
    파라미터 hsv_s = [skin_s, hair_s, eye_s]  # 피부, 눈썹, 눈의 HSV S값 리스트
    a = 가중치 [skin, hair, eye]  # 각 부위별 가중치 리스트
    질의색상 hsv_s값에서 summer의 hsv_s, winter의 hsv_s값 간의 거리를
    각각 계산하여 summer가 가까우면 1, 반대 경우 0 리턴
    '''
    # 피부, 눈썹, 눈의 여름 및 겨울 기준값
    smr_s_std = [12.5, 21.7195, 24.77064]  # 여름 쿨톤 HSV S 기준값
    wnt_s_std = [16.73913, 24.8276, 31.3726]  # 겨울 쿨톤 HSV S 기준값
    a[1] = 0.5  # 눈썹의 영향력을 줄이기 위해 가중치를 0.5로 조정

    smr_dist = 0  # 여름 기준값과의 거리 합 초기화
    wnt_dist = 0  # 겨울 기준값과의 거리 합 초기화

    body_part = ['skin', 'eyebrow', 'eye']  # 부위 이름 리스트
    for i in range(3):  # 각 부위에 대해 반복
        smr_dist += abs(hsv_s[i] - smr_s_std[i]) * a[i]  # 여름 기준값과의 가중치 적용 거리 계산
        print(body_part[i],"의 summer 기준값과의 거리")  # 부위별 여름 거리 출력
        print(abs(hsv_s[i] - smr_s_std[i]) * a[i])  # 가중치 적용 거리 값 출력
        wnt_dist += abs(hsv_s[i] - wnt_s_std[i]) * a[i]  # 겨울 기준값과의 가중치 적용 거리 계산
        print(body_part[i],"의 winter 기준값과의 거리")  # 부위별 겨울 거리 출력
        print(abs(hsv_s[i] - wnt_s_std[i]) * a[i])  # 가중치 적용 거리 값 출력

    if smr_dist <= wnt_dist:  # 여름 거리가 겨울 거리보다 작거나 같으면
        return 1  # 여름 쿨톤으로 판단 (1 반환)
    else:
        return 0  # 겨울 쿨톤으로 판단 (0 반환)