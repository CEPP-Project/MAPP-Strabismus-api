from pathlib import Path
from PIL import Image

current_dir = Path(__file__).parent
parent_dir = current_dir.parent
TEMP_DIR = parent_dir.parent / 'tmp'

def crop_and_save_upload_file(file_location: Path) -> None:
    try:
        filename = file_location.name
        im = Image.open(file_location)
        w, h = im.size
        eye_width = w / 4
        eye_heigh = h / 4
        print(im.size, eye_width, eye_heigh)

        left_w_start_pos = 5/16 * w - (eye_width/2)
        right_w_start_pos = 11/16 * w - (eye_width/2)
        h_start_pos = (h/2) - (eye_heigh/2)

        left_eye_box = (left_w_start_pos, h_start_pos, left_w_start_pos+eye_width, h_start_pos+eye_heigh)
        right_eye_box = (right_w_start_pos, h_start_pos, right_w_start_pos+eye_width, h_start_pos+eye_heigh)
        left_eye_im = im.crop(left_eye_box)
        right_eye_im = im.crop(right_eye_box)

        newfile_left_location = TEMP_DIR / f'L_{filename}'
        newfile_right_location = TEMP_DIR / f'R_{filename}'
        left_eye_im.save(newfile_left_location)
        right_eye_im.save(newfile_right_location)

        return [str(newfile_left_location), str(newfile_right_location)]

    except Exception as error:
        print("Error occured when croping file :", error)