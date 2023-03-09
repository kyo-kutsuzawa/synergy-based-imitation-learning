import datetime
import time
import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    camera_width = 640
    camera_height = 480
    scale = 2.5
    fps = 30

    n_markers = 32
    marker_size = 0.042  # [m]

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, camera_width, camera_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, camera_width, camera_height, rs.format.bgr8, fps)

    print("Start streaming")
    print("[s]tart recording  [f]inish recording  [q]uit")
    cfg = pipeline.start(config)
    profile = cfg.get_stream(rs.stream.depth)

    intrinsics = profile.as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                              [0, intrinsics.fy, intrinsics.ppy],
                              [0, 0, 1]])
    distortion_coeff = np.array(intrinsics.coeffs)

    cv2.namedWindow("RealsenseImage", cv2.WINDOW_AUTOSIZE)

    status = "waiting"

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        corners, ids, _ = aruco.detectMarkers(gray_image, dictionary)

        data = np.full((n_markers, 6), np.nan)

        if ids is not None:
            for i, id in enumerate(ids):
                if id > n_markers:
                    continue

                if id == 0:
                    marker_size = 0.0525
                else:
                    marker_size = 0.042

                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix, distortion_coeff)
                data[id] = np.concatenate([tvec[0][0], rvec[0][0]])

        aruco.drawDetectedMarkers(color_image, corners, ids, (0,255,0))

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))

        images = cv2.resize(images, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("RealsenseImage", images)

        key_command = cv2.waitKey(1) & 0xFF
        if key_command == ord("q"):
            break
        elif key_command == ord("s"):
            if status == "waiting":
                status = "recording"
                filename = "data_{}.csv".format(datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
                f = open("dataset/realsense/" + filename, "w")
                t_start = time.perf_counter()
        elif key_command == ord("f"):
            if status == "recording":
                status = "waiting"
                f.close()

        if status == "waiting":
            continue
        elif status == "recording":
            t = time.perf_counter() - t_start
            line = ",".join(map(str, [t] + data.flatten().tolist()))
            f.write(line + "\n")
            print("t={:.3f} IDs: {}".format(t, np.array(ids).flatten()))


if __name__ == "__main__":
    main()
