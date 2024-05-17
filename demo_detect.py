import cv2
import numpy as np

def detect_shape(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



    # # Áp dụng GaussianBlur để làm mịn ảnh 
    # blur = cv2.GaussianBlur(image, (5,5), 0)  
     # Áp dụng ngưỡng để tách biệt tam giác 
    threshold = 50  # Điều chỉnh giá trị ngưỡng nếu cần 
    ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV) 
    
    # Áp dụng Median Filter để lọc nhiễu
    median = cv2.medianBlur(thresh, 5)

    # Áp dụng morphologyEx để kết nối cạnh
    kernel = np.ones((1, 1), np.uint8)
    closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)
    
    # # Áp dụng Laplacian of Gaussian
    # log = cv2.Laplacian(closing, cv2.CV_64F, ksize=3)
    # abs_log = np.absolute(log)
    # edges = cv2.convertScaleAbs(abs_log)
    # Áp dụng Sobel để phát hiện cạnh
    sobelX = cv2.Sobel(closing, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(closing, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobelX, sobelY)

    # Chuyển gradient magnitude sang dạng 8-bit
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # Ngưỡng để tạo ảnh nhị phân từ gradient magnitude
    _,binary_output = cv2.threshold(gradient_magnitude, 50, 255, cv2.THRESH_BINARY)

    # Tìm contour
    contours, hierarchy = cv2.findContours(binary_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 6000
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > max_area]
    # Lọc contour theo hierarchy (chỉ lấy contour cha)
    print("so contours",len(filtered_contours))
    final_contours = []
    for i, contour in enumerate(filtered_contours):
        if hierarchy[0][i][3] == -1: # Kiểm tra contour cha
            final_contours.append(contour)
    for contour in final_contours:
        # Tính toán gần đúng chu vi của contour
        perimeter = cv2.arcLength(contour, True)
        # Tìm hình dạng gần đúng với độ chính xác nhất định
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)  # Điều chỉnh giá trị này
        print("so canh",len(approx))
        # Phân loại hình dạng dựa trên số cạnh
        if len(approx) == 3:
            shape = "Tam giác"
        elif len(approx) == 4:
            # Kiểm tra xem hình dạng có phải hình vuông hay hình chữ nhật
            x, y, w, h = cv2.boundingRect(approx)
            aspectRatio = float(w) / h
            print("ti so: %f",aspectRatio);
            if 0.9 <= aspectRatio <= 1.01:
                shape = "vuong"
            else:
                # Tính toán độ dài của hai đường chéo
                d1 = np.linalg.norm(approx[0] - approx[2])
                d2 = np.linalg.norm(approx[1] - approx[3])

                # Tính toán tỉ lệ giữa hai đường chéo
                ratio = d1 / d2
                print("ti le duong cheo",ratio)
                # Đặt ngưỡng để xác định hình thoi
                threshold = 0.98  # Có thể điều chỉnh ngưỡng này tùy thuộc vào yêu cầu

                # Kiểm tra xem tỉ lệ có gần với 1 không
                if 1 - threshold <= ratio <= 1 + threshold:
                    shape = "hinh thoi"
                else:
                    shape = "hinh chu nhat"
        else:
             # Kiểm tra xem hình dạng có phải hình tròn hoặc hình elip
            circularity = 4 * np.pi * cv2.contourArea(contour) / (perimeter * perimeter)
            print("ti so hinh tron",circularity)
            if 0.8 <= circularity <= 1.2:
                shape = "Tron"
            else:
                # Tính toán tỷ lệ giữa các trục chính của elip
                if len(contour) >= 5:  # fitEllipse yêu cầu ít nhất 5 điểm
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, orientation) = ellipse
                    major_axis_length = max(axes)
                    minor_axis_length = min(axes)
                    axis_ratio = minor_axis_length / major_axis_length
                    if axis_ratio < 0.9:
                        shape = "Elip"
                    else:
                        shape = "Khac"
                else:
                    shape = "Khac"

        # Vẽ contour và hiển thị tên hình dạng
        cv2.drawContours(image, [approx], -1, (255, 255, 255), 2)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 125, 125), 2)

    # Hiển thị ảnh kết quả
    cv2.imshow("Hình ảnh",image)
    # cv2.imshow("Hình ảnh", closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Đường dẫn đến ảnh của bạn
image_path = r"D:\vision\anhtestfinal\015.bmp"

# Gọi hàm để phát hiện hình dạng
detect_shape(image_path)