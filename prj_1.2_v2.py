import cv2
import numpy as np
import os
import glob


def save_step_image(output_dir, original_filename, step_no, step_name, image):
    base_name, ext = os.path.splitext(original_filename)
    step_filename = f"{base_name}.step_{step_no}_{step_name}{ext}"
    save_path = os.path.join(output_dir, step_filename)
    cv2.imwrite(save_path, image)


def split_large_component_with_watershed(component_mask, original_roi, min_area=20):
    """
    Chỉ dùng watershed trên 1 blob lớn để tách các hạt dính nhau.
    Trả về list contour sau khi tách.
    """
    kernel = np.ones((3, 3), np.uint8)

    sure_bg = cv2.dilate(component_mask, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)

    # threshold cao hơn để tránh 1 hạt bị chia đôi
    _, sure_fg = cv2.threshold(
        dist_transform, 0.45 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    roi_color = original_roi.copy()
    markers = cv2.watershed(roi_color, markers)

    contours_out = []
    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        mask = np.zeros(component_mask.shape, dtype=np.uint8)
        mask[markers == marker_id] = 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area >= min_area:
            contours_out.append(cnt)

    return contours_out


def count_rice_grains_optimized(image_path, output_folder=None, save_steps=False, show_steps=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    original_filename = os.path.basename(image_path)
    original = img.copy()

    # 1. Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Tăng tương phản cục bộ để đỡ sót hạt
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray)

    # 3. Blur nhẹ
    blur = cv2.GaussianBlur(gray_enhanced, (5, 5), 0)
    blur = cv2.medianBlur(blur, 5)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 0, "original", original)
        save_step_image(output_folder, original_filename, 1, "gray", gray)
        save_step_image(output_folder, original_filename, 2, "gray_enhanced", gray_enhanced)
        save_step_image(output_folder, original_filename, 3, "blur", blur)

    # 4. Threshold
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Nếu nền trắng nhiều quá thì đảo
    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.5:
        thresh = cv2.bitwise_not(thresh)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 4, "thresh", thresh)

    # 5. Morphology nhẹ để tránh mất hạt
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 5, "opening", opening)
        save_step_image(output_folder, original_filename, 6, "closing", closing)

    # 6. Tìm connected components ban đầu
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)

    # Thu thập area hợp lệ để ước lượng size 1 hạt
    candidate_areas = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 20 <= area <= 3000:
            candidate_areas.append(area)

    # fallback nếu ít dữ liệu
    if len(candidate_areas) == 0:
        estimated_single_area = 80
    else:
        estimated_single_area = int(np.median(candidate_areas))

    output = original.copy()
    rice_count = 0

    final_boxes = []

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < max(15, int(0.25 * estimated_single_area)):
            continue

        component_mask = np.zeros((h, w), dtype=np.uint8)
        local_labels = labels[y:y+h, x:x+w]
        component_mask[local_labels == i] = 255

        roi_original = original[y:y+h, x:x+w]

        # Nếu vùng không quá lớn -> coi là 1 hạt
        if area <= 1.8 * estimated_single_area:
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            cnt = max(contours, key=cv2.contourArea)
            cnt_area = cv2.contourArea(cnt)

            if cnt_area >= max(15, int(0.25 * estimated_single_area)):
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                final_boxes.append((x + rx, y + ry, rw, rh))
        else:
            # Blob lớn -> mới split bằng watershed
            split_contours = split_large_component_with_watershed(
                component_mask=component_mask,
                original_roi=roi_original,
                min_area=max(15, int(0.25 * estimated_single_area))
            )

            if len(split_contours) == 0:
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    final_boxes.append((x + rx, y + ry, rw, rh))
            else:
                for cnt in split_contours:
                    cnt_area = cv2.contourArea(cnt)

                    # bỏ vùng quá nhỏ để tránh 1 hạt bị tách đôi
                    if cnt_area < max(15, int(0.30 * estimated_single_area)):
                        continue

                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    final_boxes.append((x + rx, y + ry, rw, rh))

    # 7. Gộp box quá gần nhau để tránh 1 hạt bị đếm 2 lần
    merged_boxes = []
    used = [False] * len(final_boxes)

    def boxes_close(b1, b2, gap=6):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        return not (
            x1 + w1 + gap < x2 or
            x2 + w2 + gap < x1 or
            y1 + h1 + gap < y2 or
            y2 + h2 + gap < y1
        )

    for i in range(len(final_boxes)):
        if used[i]:
            continue

        x1, y1, w1, h1 = final_boxes[i]
        group = [final_boxes[i]]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j in range(len(final_boxes)):
                if used[j]:
                    continue
                if any(boxes_close(g, final_boxes[j], gap=4) for g in group):
                    group.append(final_boxes[j])
                    used[j] = True
                    changed = True

        xs = [b[0] for b in group]
        ys = [b[1] for b in group]
        xes = [b[0] + b[2] for b in group]
        yes = [b[1] + b[3] for b in group]

        nx = min(xs)
        ny = min(ys)
        nw = max(xes) - nx
        nh = max(yes) - ny
        merged_boxes.append((nx, ny, nw, nh))

    # 8. Vẽ kết quả
    for idx, (x, y, w, h) in enumerate(merged_boxes, start=1):
        rice_count += 1
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

        text = str(idx)
        text_x = x + w // 2 - 5
        text_y = y + h // 2 + 5

        cv2.putText(output, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(output, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(output, f"Total: {rice_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 7, "final_output", output)

    if show_steps:
        cv2.imshow("Gray", gray)
        cv2.imshow("Gray Enhanced", gray_enhanced)
        cv2.imshow("Blur", blur)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Opening", opening)
        cv2.imshow("Closing", closing)
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
            count, output_img = count_rice_grains_optimized(
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

    result_txt = os.path.join(output_folder, "results.txt")
    with open(result_txt, "w", encoding="utf-8") as f:
        for filename, count in results:
            f.write(f"{filename}: {count}\n")

    print(f"\nĐã lưu kết quả tại: {result_txt}")
    return results


if __name__ == "__main__":
    input_folder = "images"
    process_folder(input_folder, output_folder="output", save_steps=True)