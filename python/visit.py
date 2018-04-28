import cv2
import time
import os


def visit(directory_in_str):

    directory = os.fsencode(directory_in_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename:
            input_path = directory_in_str + "/" + filename
            # inputPath = directory_in_str + "\\" + filename
            print("Processing %s", input_path)
            start = time.time()
            result_mat = computeVpScore(input_path)
            end = time.time()
            print("Use time: %f sec", (end-start))
            output_path = directory_in_str + "/results" + filename
            # output_path = directory_in_str + "\\results" + filename
            cv2.imwrite(output_path, result_mat)
    if not filename:
        print("No valid input file.")