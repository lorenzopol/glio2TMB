import numpy as np
import gzip
import os
from maflib.reader import MafReader

READ_SIZE_WXS_MB = 3.5  # https://en.wikipedia.org/wiki/Exome_sequencing


def main():
    nof_snv_container: dict[str, int] = {}
    SNV_only_file_path = r"./data/SNV_only"
    for dir_file in os.listdir(SNV_only_file_path):
        dir_path = os.path.join(SNV_only_file_path, dir_file)
        # print(f"\n{dir_path=}")
        if os.path.isdir(dir_path):
            i = 0
            for file in os.listdir(dir_path):
                if file.endswith(".gz"):
                    file_path = os.path.join(dir_path, file)
                    maf: MafReader = MafReader.reader_from(file_path)
                    for line in maf:
                        line_container = (str(line).split("\t"))
                        # print(f"{line_container=}")
                        if line_container[8] != "Silent":
                            i += 1
                    nof_snv_container[file] = i
    print(f"{nof_snv_container.items()=}")
    v = nof_snv_container.values()
    print(f"{v=}")
    a = sum(v) / len(v)
    print(f"Avg: {a}")
    print(f"eTMB: {a / READ_SIZE_WXS_MB}")


def lol():
    v = np.array(
        [58, 55, 54, 29, 13, 59, 40, 41, 25, 32, 107, 55, 64, 60, 48, 95, 49, 49, 64, 75, 33, 32, 66, 68, 77, 68, 3, 39,
         93, 152, 39, 103, 62, 54, 51, 79, 40, 68, 51, 51, 1, 310, 68, 79, 8107, 48, 36, 29, 87, 42, 45, 82, 67, 70, 56,
         38, 85, 52, 44, 645, 55, 55, 46, 39, 54, 43, 75, 37, 67, 65, 55, 0, 64, 64, 91, 37, 69, 88, 15, 87, 65, 15,
         112, 84, 98, 32, 34, 98, 32, 59, 98, 74, 87, 59, 75, 63, 85, 94, 63, 52, 92, 63, 76, 59, 61, 72, 24, 18, 47,
         73, 63, 69, 58, 70, 81, 46, 32, 53, 20, 59, 53, 47, 41, 71, 51, 90, 35, 62, 54, 100, 67, 57, 75, 56, 55, 1167,
         61, 60, 68, 62, 62, 44, 39, 52, 70, 45, 110, 52, 51, 68, 71, 58, 56, 97, 92, 48, 68, 75, 64, 52, 357, 43, 70,
         60, 55, 53, 59, 33, 49, 55, 84, 63, 35, 40, 63, 77, 74, 102, 52, 67, 15, 37, 62, 58, 71, 65, 21, 51, 63, 60,
         47, 121, 65, 55, 52, 70, 29, 74, 43, 80, 102, 50, 80, 38, 56, 103, 91, 53, 115, 78, 72, 27, 31, 64, 54, 68, 81,
         83, 60, 55, 71, 70, 54, 62, 82, 63, 100, 63, 73, 62, 57, 51, 82, 77, 41, 34, 126, 31, 62, 47, 65, 65, 0, 62,
         82, 90, 42, 43, 66, 89, 56, 50, 28, 63, 86, 15218, 52, 90, 67, 84, 90, 43, 63, 56, 56, 87, 50, 38, 79, 55, 85,
         57, 57, 53, 208, 105, 122, 37, 65, 199, 50, 35, 38, 0, 53, 88, 57, 45, 64, 69, 42, 58, 46, 51, 68, 98, 113,
         101, 36, 12, 34, 80, 89, 45, 80, 89, 56, 36, 50, 91, 51, 63, 64, 41, 57, 66, 60, 193, 68, 55, 38, 61, 65, 53,
         35, 42, 68, 107, 56, 72, 63, 73, 48, 71, 87, 52, 85, 43, 63, 72, 68, 90, 45, 63, 70, 54, 46, 60, 51, 42, 29,
         66, 40, 60, 52, 63, 70, 58, 49, 60, 74, 31, 71, 49, 65, 38, 52, 90, 98, 41, 680, 21, 54, 37, 21, 28, 77, 51,
         65, 76, 87, 39, 57, 713, 91, 92, 61, 69, 34, 55, 56, 93, 58, 77, 92, 45, 68, 66, 38, 89, 81, 52, 64, 52, 34,
         45, 61, 61, 62, 57, 165, 55, 117, 41, 55, 77, 53, 36, 53, 52, 97, 51, 113, 34, 48, 46, 50, 77, 18, 46, 33, 25,
         45, 68, 16, 36, 68, 80, 41, 65, 52, 54, 85, 41, 27, 35, 69, 46, 50, 56, 54, 61, 168, 78, 56, 86, 58, 43, 26,
         83, 72])
    l = len(v)
    split_factor = .1
    start = round(l*(split_factor/2))
    end = round(l*(1-(split_factor/2)))
    v.sort()
    eheh = []
    high_counter = 0
    low_counter = 0
    for i in v[start:end]:
        c = i/READ_SIZE_WXS_MB
        print(c)
        if c > 20:
            high_counter += 1
        else:
            low_counter += 1
        eheh.append(c)
    print()
    print(f"AVG Mut/Mb = {np.mean(eheh)}")
    print(f"{low_counter=}")
    print(f"{high_counter=}")
    print(f"H/TOTAL = {high_counter/(high_counter+low_counter)}")


if __name__ == '__main__':
    main()
