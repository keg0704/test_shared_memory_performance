import numpy as np
import cv2
from multiprocessing import Process, Array
import simplejpeg
import ctypes
import time


def image_pre_process(shared_memory, rep, path):
    cnt = 0
    while cnt < rep:
        # with open(path, "rb") as f:
        #     data = f.read()
        #     img = simplejpeg.decode_jpeg(data)
        # f.close()
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        write_data(shared_memory, img.astype(np.uint8))
        cnt += 1


def to_numpy_array(mp_arr, dtype, shape):
    return np.ascontiguousarray(
        np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)
    )


def write_data(shared_mem_meta, processed_data):
    shared_mem = to_numpy_array(*shared_mem_meta)
    np.copyto(shared_mem, processed_data)


if __name__ == "__main__":
    for t in range(0, 90, 10):
        t = 1 if t == 0 else t
        cv2.setNumThreads(
            0
        )  # disable opencv's default multithreading features, make sure every process only use 100% CPU usage.
        num_workers = t
        reps = 1000
        path = "/concurrency_benchmarking/1659420200000.jpg"
        # # shared_memory initialization
        image_process_output = []
        for i in range(num_workers):
            # 1 * 3 * 224 * 224 image in fp16(half)
            image_process_output.append(
                (
                    Array(ctypes.c_int8, 3 * 224 * 224),
                    np.uint8,
                    (224, 224, 3),
                )
            )

        processes = [
            Process(
                target=image_pre_process, args=(image_process_output[i], reps, path)
            )
            for i in range(num_workers)
        ]

        st = time.time()
        for process in processes:
            process.start()

        for process in processes:
            process.join()

        print(reps * num_workers / (time.time() - st))
