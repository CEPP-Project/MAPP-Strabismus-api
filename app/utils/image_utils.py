from pathlib import Path
from PIL import Image

current_dir = Path(__file__).parent
TEMP_DIR = current_dir.parent / 'tmp'

def crop_and_save_upload_file(file_location: Path) -> None:
    try:
        # print(file_location)
        # print(type(file_location))
        # suffix = file_location.suffix
        filename = file_location.name
        # print(suffix)
        im = Image.open(file_location)
        w, h = im.size
        eye_width = w / 4
        eye_heigh = h / 4
        print(im.size, eye_width, eye_heigh)
        # left_w_start_pos = (w/4) - (eye_width/2)
        # right_w_start_pos = 3*(w/4) - (eye_width/2)
        left_w_start_pos = 5/16 * w - (eye_width/2)
        right_w_start_pos = 11/16 * w - (eye_width/2)
        # h_start_pos = (h/2) - (eye_heigh/2)
        h_start_pos = (h/2) - (eye_heigh/2)
        # print(left_w_start_pos, h_start_pos, left_w_start_pos+eye_width, h_start_pos+eye_heigh)
        # print(right_w_start_pos, h_start_pos, right_w_start_pos+eye_width, h_start_pos+eye_heigh)
        left_eye_box = (left_w_start_pos, h_start_pos, left_w_start_pos+eye_width, h_start_pos+eye_heigh)
        right_eye_box = (right_w_start_pos, h_start_pos, right_w_start_pos+eye_width, h_start_pos+eye_heigh)
        left_eye_im = im.crop(left_eye_box)
        right_eye_im = im.crop(right_eye_box)
        left_eye_im.save(TEMP_DIR / f'L_{filename}')
        right_eye_im.save(TEMP_DIR / f'R_{filename}')

    except Exception as error:
        print("Error occured when croping file :", error)