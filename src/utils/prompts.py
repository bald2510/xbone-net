"""Cung cấp tiện ích prompts cho huấn luyện, đánh giá và phân tích XBone-Net.

Notes
-----
Mô-đun này thuộc cơ sở mã nguồn nghiên cứu XBone-Net và giữ các quy ước dùng chung của dự án.
"""


ORIGINAL_CLIP_PROMPT_TEMPLATE = "A radiograph showing {label}."


# ============================================================
# Thiết lập thành phần dùng chung cho quy trình xử lý của mô-đun.
# ============================================================

def generate_clip_class_prompts(
    class_names: list[str],
    template: str = ORIGINAL_CLIP_PROMPT_TEMPLATE,
) -> dict[str, str]:
    """Sinh clip class prompts cho bước xử lý hiện tại.

    Parameters
    ----------
    class_names : list[str]
        Nhãn hoặc chỉ số lớp liên quan.
    template : str, optional
        Giá trị ``template`` được sử dụng trong phép xử lý.

    Returns
    -------
    dict[str, str]
        Kết quả được tạo bởi bước xử lý của hàm.

    Raises
    ------
    ValueError
        Khi dữ liệu hoặc trạng thái đầu vào không hợp lệ.
    """
    if "{label}" not in template:
        raise ValueError("The CLIP prompt template must contain the '{label}' field.")
    return {
        class_name: template.format(label=class_name)
        for class_name in class_names
    }


def generate_custom_prompts(
    pathologies: list[str],
    image_context: str = "a bone x-ray",
) -> dict[str, dict[str, str]]:
    """Sinh custom prompts cho bước xử lý hiện tại.

    Parameters
    ----------
    pathologies : list[str]
        Danh sách tên bệnh lý hoặc lớp đích.
    image_context : str, optional
        Ảnh hoặc biểu diễn ảnh đầu vào.

    Returns
    -------
    dict[str, dict[str, str]]
        Kết quả được tạo bởi bước xử lý của hàm.
    """
    # Bước hỗ trợ để sinh custom prompts cho bước xử lý hiện tại.
    return {
        path: {
            "positive": f"this is an image of {image_context}; {path.lower()} presented in image",
            "negative": f"this is an image of {image_context}; no {path.lower()} presented in image",
        }
        for path in pathologies
    }

