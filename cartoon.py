import cv2 as cv
import numpy as np
from scipy import stats
from collections import defaultdict


def update_cluster_centers(centers, hist):
    while True:
        groups = defaultdict(list)

        for i in range(len(hist)):
            if hist[i] == 0:
                continue
            dist = np.abs(centers - i)
            index = np.argmin(dist)
            groups[index].append(i)

        new_centers = np.array(centers)
        for i, indices in groups.items():
            if np.sum(hist[indices]) == 0:
                continue
            new_centers[i] = int(np.sum(indices * hist[indices]) / np.sum(hist[indices]))

        if np.sum(new_centers - centers) == 0:
            break
        centers = new_centers

    return centers, groups


def k_histogram_clustering(hist):
    alpha = 0.001
    min_group_size = 50
    centers = np.array([128])

    while True:
        centers, groups = update_cluster_centers(centers, hist)

        new_centers = set()
        for i, indices in groups.items():
            if len(indices) < min_group_size:
                new_centers.add(centers[i])
                continue

            z, pval = stats.normaltest(hist[indices])
            if pval < alpha:
                left = 0 if i == 0 else centers[i - 1]
                right = len(hist) - 1 if i == len(centers) - 1 else centers[i + 1]
                delta = right - left
                if delta >= 3:
                    c1 = (centers[i] + left) / 2
                    c2 = (centers[i] + right) / 2
                    new_centers.add(c1)
                    new_centers.add(c2)
                else:
                    new_centers.add(centers[i])
            else:
                new_centers.add(centers[i])
        if len(new_centers) == len(centers):
            break
        else:
            centers = np.array(sorted(new_centers))
    return centers


def cartoonize_image(img):
    kernel = np.ones((2, 2), np.uint8)
    output = np.array(img)
    x, y, c = output.shape
    for i in range(c):
        output[:, :, i] = cv.bilateralFilter(output[:, :, i], 5, 150, 150)

    edge = cv.Canny(output, 100, 200)
    output = cv.cvtColor(output, cv.COLOR_RGB2HSV)

    hists = []

    hist, _ = np.histogram(output[:, :, 0], bins=np.arange(180 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 1], bins=np.arange(256 + 1))
    hists.append(hist)
    hist, _ = np.histogram(output[:, :, 2], bins=np.arange(256 + 1))
    hists.append(hist)

    cluster_centers = []
    for h in hists:
        cluster_centers.append(k_histogram_clustering(h))

    output = output.reshape((-1, c))
    for i in range(c):
        channel = output[:, i]
        index = np.argmin(np.abs(channel[:, np.newaxis] - cluster_centers[i]), axis=1)
        output[:, i] = cluster_centers[i][index]
    output = output.reshape((x, y, c))
    output = cv.cvtColor(output, cv.COLOR_HSV2RGB)

    contours, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(output, contours, -1, 0, thickness=2)
    for i in range(3):
        output[:, :, i] = cv.erode(output[:, :, i], kernel, iterations=1)
    return output


def process_video(input_file, skip=1):
    video_capture = cv.VideoCapture(input_file)
    frame_n = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_n % skip == 0:
            cartoonized_frame = cartoonize_image(frame)
        frame_n += 1
        merge = np.hstack((frame, cartoonized_frame))
        cv.imshow('Original | Cartoonized', merge)
        if cv.waitKey(1) & 0xFF == 27:
            break
    video_capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    input_video = 'data/test.mov'
    process_video(input_video, 2)
