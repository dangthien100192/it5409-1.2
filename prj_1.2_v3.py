import cv2
import numpy as np
import os
import glob


def save_step_image(output_dir, original_filename, step_no, step_name, image):
    """
    Lưu ảnh theo format:
    [ten_file_goc].step_[so]_[tenbuoc].[ext]
    """
    base_name, ext = os.path.splitext(original_filename)
    step_filename = f"{base_name}.step_{step_no}_{step_name}{ext}"
    save_path = os.path.join(output_dir, step_filename)
    cv2.imwrite(save_path, image)


def make_color_overlay(gray_or_bgr, mask, color=(0, 255, 0), alpha=0.35):
    """
    Tạo ảnh overlay để debug vùng mask.
    """
    if len(gray_or_bgr.shape) == 2:
        base = cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    else:
        base = gray_or_bgr.copy()

    overlay = base.copy()
    overlay[mask > 0] = color
    out = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)
    return out


def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_contains(box_big, box_small, margin=2):
    xb, yb, wb, hb = box_big
    xs, ys, ws, hs = box_small

    return (
        xs >= xb - margin and
        ys >= yb - margin and
        xs + ws <= xb + wb + margin and
        ys + hs <= yb + hb + margin
    )


def suppress_duplicate_boxes(boxes, areas, iou_thresh=0.45):
    """
    Loại box trùng/đè nhau để giảm 1 hạt bị đếm 2 lần.
    Giữ box có area lớn hơn.
    """
    if not boxes:
        return []

    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    keep = []

    for i in order:
        candidate = boxes[i]
        candidate_area = areas[i]
        should_keep = True

        for kept_idx in keep:
            kept_box = boxes[kept_idx]

            iou = compute_iou(candidate, kept_box)

            # Trùng mạnh
            if iou > iou_thresh:
                should_keep = False
                break

            # Box nhỏ nằm gần như lọt trong box lớn
            if box_contains(kept_box, candidate, margin=2):
                should_keep = False
                break

            # Hai box rất sát nhau, cùng vùng -> bỏ box nhỏ hơn
            x1, y1, w1, h1 = candidate
            x2, y2, w2, h2 = kept_box
            cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
            cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

            size_ref = min(max(w1, h1), max(w2, h2))
            if dist < 0.25 * size_ref and candidate_area < 0.75 * areas[kept_idx]:
                should_keep = False
                break

        if should_keep:
            keep.append(i)

    return keep
def split_touching_rice_local(roi_bgr, roi_mask, min_area=18):
    """
    Tách cục bộ 1 vùng nghi có nhiều hạt dính nhau.
    Đầu vào:
        - roi_bgr: ảnh màu của vùng box
        - roi_mask: mask nhị phân của đúng vùng đó (255 là foreground)
    Đầu ra:
        - danh sách contour đã tách trong hệ tọa độ local của ROI
    """
    kernel = np.ones((3, 3), np.uint8)

    # Làm sạch nhẹ
    local = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Distance transform
    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)

    if dist.max() <= 0:
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # Threshold hơi cao hơn để tạo nhiều tâm tách hơn trong vùng dính
    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(local, kernel, iterations=2)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(roi_bgr.copy(), markers)

    split_contours = []
    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue

        piece = np.zeros(local.shape, dtype=np.uint8)
        piece[markers == marker_id] = 255

        contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area >= min_area:
            split_contours.append(cnt)

    # Nếu watershed không tách được thì trả contour gốc
    if len(split_contours) == 0:
        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    return split_contours

def count_rice_grains(image_path, output_folder=None, save_steps=False, show_steps=False):
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    original_filename = os.path.basename(image_path)

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 0, "original", original)
        save_step_image(output_folder, original_filename, 1, "gray", gray)

    # 1. Tăng tương phản nhẹ để đỡ sót hạt
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # 2. Blur nhẹ hơn bản cũ để bớt mất hạt nhỏ
    blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
    blur = cv2.medianBlur(blur, 3)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 2, "gray_clahe", gray_clahe)
        save_step_image(output_folder, original_filename, 3, "blur", blur)

    # 3. Threshold Otsu
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Nếu foreground trắng quá nhiều thì đảo lại
    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.70:
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 4, "thresh", thresh)

    # 4. Morphology nhẹ để vừa sạch vừa không làm mất hạt
    kernel = np.ones((3, 3), np.uint8)

    # opening 1 lần thôi để bớt xóa hạt nhỏ
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # closing 1 lần để nối vùng đứt
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 5, "opening", opening)
        save_step_image(output_folder, original_filename, 6, "closing", closing)

    # 5. Nền chắc chắn
    sure_bg = cv2.dilate(closing, kernel, iterations=2)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 7, "sure_bg", sure_bg)

    # 6. Distance transform
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    dist_transform_norm = cv2.normalize(
        dist_transform, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Hạ threshold xuống một chút để giảm sót hạt
    # Bản cũ 0.35, ở đây dùng 0.28 để bắt foreground đầy đủ hơn
    _, sure_fg = cv2.threshold(
        dist_transform, 0.28 * dist_transform.max(), 255, 0
    )
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 8, "distance_transform", dist_transform_norm)
        save_step_image(output_folder, original_filename, 9, "sure_fg", sure_fg)
        save_step_image(output_folder, original_filename, 10, "unknown", unknown)

    # 7. Marker
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    if save_steps and output_folder:
        markers_vis_before = cv2.normalize(
            markers.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        save_step_image(output_folder, original_filename, 11, "markers_before_watershed", markers_vis_before)

    # 8. Watershed
    markers = cv2.watershed(original, markers)

    if save_steps and output_folder:
        markers_vis_after = markers.copy().astype(np.float32)
        markers_vis_after = cv2.normalize(
            markers_vis_after, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        save_step_image(output_folder, original_filename, 12, "markers_after_watershed", markers_vis_after)

    # 9. Lấy các vùng hợp lệ
    output = original.copy()
    valid_mask = np.zeros(gray.shape, dtype=np.uint8)

    unique_markers = np.unique(markers)

    candidate_boxes = []
    candidate_areas = []
    candidate_contours = []
    min_area = 18
    max_area = 7000
    # Ước lượng diện tích hạt chuẩn từ các vùng hiện có
    if len(candidate_areas) > 0:
        median_area = np.median(candidate_areas)
    else:
        median_area = 60


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

        if not (min_area <= area <= max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # lọc nhiễu cực mảnh nhưng vẫn giữ hạt dài
        aspect_ratio = max(w, h) / max(1, min(w, h))
        rect_area = w * h
        fill_ratio = area / max(1, rect_area)

        # chỉ loại nhiễu rất cực đoan
        if rect_area < 15:
            continue

        if fill_ratio < 0.18 and area < 40:
            continue

        if aspect_ratio > 12 and area < 50:
            continue

        candidate_boxes.append((x, y, w, h))
        candidate_areas.append(area)
        candidate_contours.append(cnt)

    # 10. Nếu box quá to thì thử tách cục bộ
    refined_boxes = []
    refined_areas = []
    refined_contours = []

    for i in range(len(candidate_boxes)):
        x, y, w, h = candidate_boxes[i]
        cnt = candidate_contours[i]
        area = candidate_areas[i]

        # box to bất thường hoặc area lớn bất thường => thử tách
        too_large = (
                area > 1.55 * median_area or
                w > 1.7 * np.sqrt(median_area) or
                h > 1.7 * np.sqrt(median_area)
        )

        if not too_large:
            refined_boxes.append((x, y, w, h))
            refined_areas.append(area)
            refined_contours.append(cnt)
            continue

        # Tạo ROI local để split
        roi_bgr = original[y:y + h, x:x + w].copy()
        roi_mask = np.zeros((h, w), dtype=np.uint8)

        # chuyển contour global -> local
        cnt_local = cnt.copy()
        cnt_local[:, 0, 0] -= x
        cnt_local[:, 0, 1] -= y
        cv2.drawContours(roi_mask, [cnt_local], -1, 255, -1)

        split_contours = split_touching_rice_local(
            roi_bgr=roi_bgr,
            roi_mask=roi_mask,
            min_area=min_area
        )

        # Nếu tách không ra hoặc chỉ ra 1 mảnh thì giữ nguyên
        if len(split_contours) <= 1:
            refined_boxes.append((x, y, w, h))
            refined_areas.append(area)
            refined_contours.append(cnt)
            continue

        # Nếu tách ra nhiều mảnh thì đưa từng mảnh vào danh sách mới
        for scnt in split_contours:
            sa = cv2.contourArea(scnt)
            if sa < min_area:
                continue

            sx, sy, sw, sh = cv2.boundingRect(scnt)

            # contour local -> global
            scnt_global = scnt.copy()
            scnt_global[:, 0, 0] += x
            scnt_global[:, 0, 1] += y

            refined_boxes.append((x + sx, y + sy, sw, sh))
            refined_areas.append(sa)
            refined_contours.append(scnt_global)

    # 11. Hậu kiểm giảm trùng sau khi refine
    keep_indices = suppress_duplicate_boxes(
        refined_boxes,
        refined_areas,
        iou_thresh=0.45
    )

    rice_count = 0
    for idx in keep_indices:
        cnt = refined_contours[idx]
        x, y, w, h = refined_boxes[idx]

        rice_count += 1
        cv2.drawContours(valid_mask, [cnt], -1, 255, -1)

        # Vẽ contour + box
        cv2.drawContours(output, [cnt], -1, (255, 0, 0), 1)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

        text = str(rice_count)
        text_x = x + w // 2 - 5
        text_y = y + h // 2 + 5

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
    valid_overlay = make_color_overlay(gray, valid_mask, color=(0, 255, 0), alpha=0.45)
    if save_steps and output_folder:
        save_step_image(output_folder, original_filename, 13, "valid_mask", valid_mask)
        save_step_image(output_folder, original_filename, 14, "valid_overlay", valid_overlay)
        save_step_image(output_folder, original_filename, 15, "final_output", output)

    if show_steps:
        cv2.imshow("Original", original)
        cv2.imshow("Gray", gray)
        cv2.imshow("Gray CLAHE", gray_clahe)
        cv2.imshow("Blur", blur)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Opening", opening)
        cv2.imshow("Closing", closing)
        cv2.imshow("Sure BG", sure_bg)
        cv2.imshow("Distance Transform", dist_transform_norm)
        cv2.imshow("Sure FG", sure_fg)
        cv2.imshow("Unknown", unknown)
        cv2.imshow("Valid Overlay", valid_overlay)
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

    result_txt = os.path.join(output_folder, "results.txt")
    with open(result_txt, "w", encoding="utf-8") as f:
        for filename, count in results:
            f.write(f"{filename}: {count}\n")

    print(f"\nĐã lưu kết quả tại: {result_txt}")
    return results


if __name__ == "__main__":
    input_folder = "images"   # thư mục chứa ảnh
    process_folder(input_folder, output_folder="output", save_steps=True)