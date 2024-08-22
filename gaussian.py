import cv2 as cv
import numpy as np
import os

def process_images_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 필요한 파일 확장자만 필터링
            image_path = os.path.join(input_folder, filename)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to load {image_path}")
                continue
            
            # 로그 변환 적용
            log_transformed = np.log1p(image.astype(np.float64))  # np.log1p는 log(1 + x)의 변형

            # 노이즈 제거를 위한 간단한 필터 (예: 가우시안 블러)
            filtered = cv.GaussianBlur(log_transformed, (5, 5), 0)

            # 역 로그 변환
            restored_image = np.expm1(filtered)  # np.expm1은 exp(x) - 1

            # 이미지 범위를 [0, 255]로 제한
            restored_image = np.clip(restored_image, 0, 255)
            
            # 결과 이미지 저장
            output_path = os.path.join(output_folder, f"{filename}")
            cv.imwrite(output_path, np.uint8(restored_image))
            print(f"Processed and saved: {output_path}")

# 입력 및 출력 폴더 설정
input_folder = '/root/workplace/Pytorch-UNet/data/sar2sar/pre'
output_folder = '/root/workplace/Pytorch-UNet/data/sar2sar/pre/pre'

# 폴더 내의 모든 이미지 처리
process_images_in_folder(input_folder, output_folder)
