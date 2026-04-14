import cv2
import numpy as np
import os
import glob


def save_step_image(output_dir, original_filename, step_no, step_name, image):
    """
    Lưu ảnh theo format:
    [ten_file_goc].[step]_[buoc].[ext]

    Ví dụ:
    rice1.step_1_gray.jpg
    rice1.step_2_blur.jpg
    """
    base_name, ext = os.path.splitext(original_filename)
    step_filename = f"{base_name}.step_{step_no}_{step_name}{ext}"
    save_path = os.path.join(output_dir, step_filename)
    cv2.imwrite(save_path, image)


def count_rice_grains(image_path, output_folder=None, save_steps=False, show_steps=False):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    original_filename = os.path.basename(image_path)

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lưu bước 0: ảnh gốc
    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 0, "original", original)

    # 1. Khử nhiễu
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur = cv2.medianBlur(blur, 5)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 1, "gray", gray)
        save_step_image(output_folder, original_filename, 2, "blur", blur)

    # 2. Threshold
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.7:
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 3, "thresh", thresh)

    # 3. Morphology để làm sạch
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 4, "opening", opening)
        save_step_image(output_folder, original_filename, 5, "closing", closing)

    # 4. Tìm nền chắc chắn
    sure_bg = cv2.dilate(closing, kernel, iterations=3)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 6, "sure_bg", sure_bg)

    # 5. Distance transform để tìm tâm hạt
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    dist_transform_norm = cv2.normalize(
        dist_transform, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, sure_fg = cv2.threshold(
        dist_transform, 0.35 * dist_transform.max(), 255, 0
    )

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 7, "distance_transform", dist_transform_norm)
        save_step_image(output_folder, original_filename, 8, "sure_fg", sure_fg)
        save_step_image(output_folder, original_filename, 9, "unknown", unknown)

    # 6. Connected components làm marker
    num_markers, markers = cv2.connectedComponents(sure_fg)

    # Marker cho watershed phải > 1, nền = 1, unknown = 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Lưu marker trước watershed để dễ debug
    if save_steps and output_folder:
        markers_vis_before = cv2.normalize(markers.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        markers_vis_before = markers_vis_before.astype(np.uint8)
        save_step_image(output_folder, original_filename, 10, "markers_before_watershed", markers_vis_before)

    # 7. Watershed để tách các hạt dính nhau
    markers = cv2.watershed(original, markers)

    if save_steps and output_folder:
        markers_vis_after = markers.copy().astype(np.float32)
        markers_vis_after = cv2.normalize(markers_vis_after, None, 0, 255, cv2.NORM_MINMAX)
        markers_vis_after = markers_vis_after.astype(np.uint8)
        save_step_image(output_folder, original_filename, 11, "markers_after_watershed", markers_vis_after)

    # 8. Đếm các vùng hợp lệ + đánh số từng hạt
    rice_count = 0
    output = original.copy()

    unique_markers = np.unique(markers)

    for marker_id in unique_markers:
        if marker_id <= 1:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        mask[markers == marker_id] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        min_area = 20
        max_area = 5000

        if min_area <= area <= max_area:
            rice_count += 1
            x, y, w, h = cv2.boundingRect(cnt)

            # Vẽ khung
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # Tính vị trí ghi số thứ tự
            text = str(rice_count)
            text_x = x + w // 2 - 5
            text_y = y + h // 2 + 5

            # Nền đen cho dễ nhìn
            cv2.putText(
                output,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                2,
                cv2.LINE_AA
            )

            # Chữ vàng/đỏ hoặc xanh tùy thích, ở đây dùng đỏ
            cv2.putText(
                output,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    # Ghi tổng số hạt ở góc ảnh
    cv2.putText(
        output,
        f"Total: {rice_count}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 12, "final_output", output)

    if show_steps:
        cv2.imshow("Original", original)
        cv2.imshow("Gray", gray)
        cv2.imshow("Blur", blur)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Opening", opening)
        cv2.imshow("Closing", closing)
        cv2.imshow("Sure BG", sure_bg)
        cv2.imshow("Distance Transform", dist_transform_norm)
        cv2.imshow("Sure FG", sure_fg)
        cv2.imshow("Unknown", unknown)
        cv2.imshow("Output", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rice_count, output


def process_folder(input_folder, output_folder="output", save_steps=True):
    os.makedirs(output_folder, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))

    results = []

    for image_path in image_files:
        try:
            count, output_img = count_rice_grains(
                image_path,
                output_folder=output_folder,
                save_steps=save_steps,
                show_steps=False
            )

            filename = os.path.basename(image_path)
            results.append((filename, count))
            print(f"{filename}: {count} hạt gạo")

        except Exception as e:
            print(f"Lỗi ảnh {image_path}: {e}")

    # Ghi file kết quả
    result_txt = os.path.join(output_folder, "results.txt")
    with open(result_txt, "w", encoding="utf-8") as f:
        for filename, count in results:
            f.write(f"{filename}: {count}\n")

    print(f"\nĐã lưu kết quả tại: {result_txt}")
    return results


if __name__ == "__main__":
    input_folder = "images"   # thư mục chứa ảnh
    process_folder(input_folder, output_folder="output", save_steps=True)