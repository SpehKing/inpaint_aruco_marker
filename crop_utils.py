def safe_crop_center(cx, cy, crop_size, img_width, img_height):
    half = crop_size // 2
    start_x = int(cx - half)
    start_y = int(cy - half)
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    if start_x < 0:
        end_x -= start_x
        start_x = 0
    if start_y < 0:
        end_y -= start_y
        start_y = 0
    if end_x > img_width:
        diff = end_x - img_width
        start_x -= diff
        end_x = img_width
    if end_y > img_height:
        diff = end_y - img_height
        start_y -= diff
        end_y = img_height

    final_width = end_x - start_x
    final_height = end_y - start_y
    return start_x, start_y, end_x, end_y, final_width, final_height


def compute_patch_area(cx, cy, crop_size, img_w, img_h):
    half = crop_size // 2
    start_x = int(cx - half)
    start_y = int(cy - half)
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    # Clamp
    if start_x < 0:
        end_x -= start_x
        start_x = 0
    if start_y < 0:
        end_y -= start_y
        start_y = 0
    if end_x > img_w:
        diff = end_x - img_w
        start_x -= diff
        end_x = img_w
    if end_y > img_h:
        diff = end_y - img_h
        start_y -= diff
        end_y = img_h
    return start_x, start_y, end_x, end_y
