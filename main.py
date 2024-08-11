import math

import numpy as np
import json
import os
from maflib.reader import MafReader
from PIL import Image
import cv2
import keras

Image.MAX_IMAGE_PIXELS = None

READ_SIZE_WXS_MB = 38  # sources: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7178217/ && https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4051326/
TMB_PER_MB_HIGH_THRESHOLD = 20  # sources https://pubmed.ncbi.nlm.nih.gov/29899191/
GBM_WSSM_GREEN = np.array([5, 208, 4])  # Cellular Tumor (CT)
GBM_WSSM_BLUE = np.array([33, 143, 166])  # Leading Edge (LE)
GBM_WSSM_PURPLE = np.array([210, 5, 208])  # Infiltrating Tumor (IT)
GBM_WSSM_LIGHT_BLUE = np.array([37, 209, 247])  # Perinecrotic Zone (CTpnz)
GBM_WSSM_SEA_GREEN = np.array([6, 208,
                               170])  # Pseudopalisading Cells Around Necrosis (CTpan) > https://academic.oup.com/jnen/article/65/6/529/2645248
GBM_WSSM_RED = np.array([255, 102, 0])  # Microvascular Proliferation (CTmvp)
GBM_WSSM_BLACK = np.array([5, 5, 5])  # Necrosis (CTne)
GBM_WSSM_WHITE = np.array([255, 255, 255])  # Background (BC)


def playground():
    segmented_img = np.array(Image.open(r"./data/prediction/1024_segment.png").resize((512, 512), Image.Resampling.NEAREST))
    img = np.array(Image.open(r"./data/resized/1024.png").resize((512, 512), Image.Resampling.NEAREST))
    mask = filter_image_by_color(segmented_img, GBM_WSSM_GREEN, 1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for idx, cnt in enumerate(cnts):
        if cv2.contourArea(cnt) > 256:
            cv2.drawContours(img, cnts, idx, (0, 0, 255), 2)

    cv2.imshow("segmented", segmented_img)
    cv2.imshow("tumor img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cache_maf_folder_to_json(path_to_folder: str, path_to_json: str):
    nof_snv_container: dict[str, list] = {}

    for dir_or_file in os.listdir(path_to_folder):
        dir_path = os.path.join(path_to_folder, dir_or_file)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(".gz"):
                    file_path = os.path.join(dir_path, file)
                    maf: MafReader = MafReader.reader_from(file_path)
                    file_lines = [str(line).split("\t") for line in maf]
                    nof_snv_container[file] = file_lines
    with open(path_to_json, "w") as f:
        json.dump(nof_snv_container, f, indent=2)


def read_cached_maf(path_to_json_cache: str) -> dict[str, list]:
    with open(path_to_json_cache, "r") as f:
        return json.load(f)


def compute_stats_on_cache(maf_cache_container: dict[str, list], drop_percentage: float) -> None:
    """
    :param maf_cache_container: json cache of MAF loaded by read_cached_maf
    :param drop_percentage: percentage of the lowest AND highest values to drop from the sorted distribution of values
     (use to drop outliers). It can range from 0 to 1. E.g. 0.1 means 5% of the lowest and 5% of the highest values get dropped.
    :return: None
    """
    print(f"Dropping {drop_percentage / 2 * 100}% of the lowest and {drop_percentage / 2 * 100}% highest values")
    cache_len = len(maf_cache_container)
    nof_snv_container = np.zeros(cache_len)
    start = round(cache_len * (drop_percentage / 2))
    end = round(cache_len * (1 - (drop_percentage / 2)))
    # todo: optimize this?
    for index, (k, v) in enumerate(maf_cache_container.items()):  # drop silent mutations
        nof_snv = 0
        for line in v:
            if line[8] != "Silent":
                nof_snv += 1
        nof_snv_container[index] = nof_snv
    nof_snv_container.sort()
    avg_mut = np.mean(nof_snv_container[start:end])
    print(f"AVG Mut/patient = {avg_mut}")
    print(f"AVG Mut/Mb = {avg_mut / READ_SIZE_WXS_MB}")
    high_tmb_counter = 0
    for i in range(len(nof_snv_container)):
        if nof_snv_container[i] / READ_SIZE_WXS_MB > TMB_PER_MB_HIGH_THRESHOLD:
            high_tmb_counter += 1
    print(
        f"Number of patients that exceed {TMB_PER_MB_HIGH_THRESHOLD}Mut/Mb = {high_tmb_counter} of {len(nof_snv_container)} [{round(high_tmb_counter / len(nof_snv_container), 2)}%]")


def filter_image_by_color(image: np.array, color: np.array, should_dilate: int) -> np.array:
    """
    :param image: numpy array of the image to filter
    :param color: numpy array of the color to filter by
    :param should_dilate:  any positive integers to dilate the mask by should_dilate iterations. 0 means no dilation
    :return: mask image
    """
    mask = cv2.inRange(image, color - 1, color + 1)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=should_dilate)


def get_bounding_rects_on_segmented_image(segmented: np.array, color: np.array, area_threshold: int = 256) -> list:
    mask = filter_image_by_color(segmented, color, 1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    rect_list = []
    for idx_cnt, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > area_threshold:
            x, y, w, h = cv2.boundingRect(cnt)
            rect_list.append((x, y, w, h))
    return rect_list


def resize_gdc_img_to_square(loaded_gdc_img: Image.Image, size: int, resized_output_path: str) -> None:
    # util function to resize gdc images to squares and save them
    loaded_gdc_img.resize((size, size), Image.Resampling.NEAREST).save(resized_output_path)


def snap_to_prev_multiple_of(number: int, multiple: int) -> int:
    return math.floor(number / multiple) * multiple


def enlarge_square_image_by_factor(image: np.array, factor: int) -> np.array:
    enlarge_height = image.shape[0] * factor
    enlarge_width = image.shape[1] * factor
    return np.array(Image.fromarray(image).resize((enlarge_width, enlarge_height), Image.Resampling.NEAREST))


def main():
    SNV_only_file_path = r"./data/SNV_only"
    json_path = r"./data/SNV_only.json"
    gdc_gdc_image = r"C:\Users\loren\PycharmProjects\glio2TMB\data\images\TCGA-06-0188-01Z-00-DX7.f175e12e-b350-41a7-8dfa-3095909a7e04.svs"

    mid_res_img = cv2.imread(r"./data/resized/8192.png")
    segmented_img = cv2.imread(r"./data/prediction/1024_segment.png")

    scaling_factor = round(mid_res_img.shape[0] / segmented_img.shape[0])

    bounding_rects = get_bounding_rects_on_segmented_image(segmented_img, GBM_WSSM_GREEN)
    mask = filter_image_by_color(segmented_img, GBM_WSSM_GREEN, 1)
    enlarged_mask = enlarge_square_image_by_factor(mask, scaling_factor)
    masked_img = cv2.bitwise_and(mid_res_img, mid_res_img, mask=enlarged_mask)

    tiles_container = []
    tile_size = 64
    for x, y, w, h in bounding_rects:
        tumor_only = masked_img[y * scaling_factor:(y + h) * scaling_factor, x * scaling_factor:(x + w) * scaling_factor]
        new_height = snap_to_prev_multiple_of(tumor_only.shape[0], tile_size)
        new_width = snap_to_prev_multiple_of(tumor_only.shape[1], tile_size)
        tumor_only = tumor_only[:new_height, :new_width]
        # todo: color norm?

        for i in range(0, new_height, tile_size):
            for j in range(0, new_width, tile_size):
                tile = tumor_only[i:i + tile_size, j:j + tile_size]
                nof_black_pixel = np.count_nonzero(np.all(tile == [0, 0, 0], axis=2))
                if nof_black_pixel / (tile_size * tile_size) < 0.5:  # drop those tiles that are mostly black
                    tiles_container.append(tile)

    # from https://keras.io/api/applications/resnet/ and https://keras.io/guides/transfer_learning/
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        input_tensor=keras.Input(shape=(tile_size, tile_size, 3)),
        pooling="avg"
        )
    # todo: consider dropping at layer conv2_block3_add or conv3_block4_out because of the reduced size of the tiles
    model = keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('conv2_block3_out').output)

    tile_level_FV_container = []
    for tile in tiles_container:
        preprocess_tile = keras.applications.resnet.preprocess_input(tile)
        feature_vector = model.predict(np.expand_dims(preprocess_tile, 0))  # FV = features vector
        tile_level_FV = keras.layers.GlobalAveragePooling2D()(feature_vector)
        tile_level_FV_container.append(tile_level_FV)
    patient_level_features = np.mean(tile_level_FV_container, axis=0)

    print(patient_level_features.shape)
    print(patient_level_features)



if __name__ == '__main__':
    main()
