from Functions import *
from tqdm import tqdm

# -----------------------HYPERPARAMETERS

# intrinsic camera parameter
K = np.array([[1.66713324e+03, 0.00000000e+00, 9.51137528e+02],
              [0.00000000e+00, 1.66954293e+03, 5.29018205e+02],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
# distorsion parameter
distortion = np.array([[1.13913462e-01, -2.75143257e-02,  3.39404705e-04, -2.42018474e-03, -6.13191579e-01]], dtype=np.float32)
# which video
# N_V = 1
# directory
FOLDER = "C:/Users/zanna/JupyterAnaconda/Project CV/data/"
# video
# VIDEO = f"obj0{N_V}.mp4"
# n voxel
N_VOXEL = 100
# side cube
# CUBE_SIDE = 100.
# height cube start
# HEIGHT_START = 85.

VERBOSE = 0  # can be an integer between 0 and 3

if __name__ == '__main__':
    # for each video
    for N_V, CUBE_SIDE, HEIGHT_START, name_obj in [(1, 110, 85, "tucan"), (2, 120, 85, "tricer"), (3, 130, 75, "cracker"), (4, 90, 85, "hindu")]:
        # compute grid 3D in homogeneous coordinates
        grid_3d = compute_grid_3D(height_start=HEIGHT_START, cube_side=CUBE_SIDE, n_voxel=N_VOXEL)
        VIDEO = f"obj0{N_V}.mp4"
        # create object video
        video = VideoReader(FOLDER, VIDEO, K, distortion)
        # take frame (reduce dimension of frame)
        I_g = video.get_actual_frame([1150, 1590])
        # result = cv.VideoWriter(f'{N_V}_silouette.avi',
        #                         cv.VideoWriter_fourcc(*'MJPG'),
        #                         20, (int(video.get(3)), int(video.get(4))))
        # extract polygons from image
        polygon_list = find_polygons_LK(I_g)
        # for each polygon extract only 3 points (the 2 most distance form origin and the concave vertex)
        polygon_list = find_important_points_LK(polygon_list)
        # determine angles
        angle_list = find_angle_list_LK(I_g, polygon_list)
        # set old image and old polygons
        I_g_old = I_g
        polygon_list_old = polygon_list
        # go to next frame
        video.next_frame()
        pbar = tqdm(total=video.total_frames)
        while not video.is_finished():
            # take actual frame
            I_g = video.get_actual_frame([1150, 1590])
            # compute polygons (each 20 iterations I compute a full ispection to have a better result)
            if not (video.get_frame_id() % 20 == 0):
                # LK flow
                polygon_list, status, error = optical_flow(I_g_old, I_g, polygon_list_old)
                # remove polygons no longer in right domain for viewer
                polygon_list, l_del = check_right_polygons_LK(polygon_list, n_points=3)
                # remove angles of polygons no longer in right domain for viewer
                angle_list = np.delete(angle_list, l_del, axis=0)
                # increment precision of points for each polygon
                polygon_list = find_more_accurate_points_LK(I_g, polygon_list, n_points=3)
            # refresh
            else:
                # find polygons
                polygon_list = find_polygons_LK(I_g)
                # extact only important points for each polygon (concave vertex, the 2 most distance points from "origin")
                polygon_list = find_important_points_LK(polygon_list)
                # define angle list
                angle_list = find_angle_list_LK(I_g, polygon_list)
            # set old image
            I_g_old = I_g
            # set old polygons
            polygon_list_old = polygon_list
            # adjust x axes to bring all polygons in full image
            polygon_list_adj = adj_axis(polygon_list, adj_x=1150.)
            # determine silouette
            silouette = determine_silouette(video.get_actual_frame(color=True), apply_closing=N_V == 1)
            # compute rotation translation matrix
            RT_mat = compute_RT_matrix(polygon_list_adj, angle_list, K, distortion)
            # compute grid 2d in inhomogeneous coordinates
            grid_2d = compute_grid_2D(grid_3d, K, RT_mat)
            # filter grid_3d
            grid_3d = filter_grid_3D(grid_3d, grid_2d, silouette)
            if VERBOSE == 1:
                if verbose_cv_thresh(silouette):
                    break
            elif VERBOSE == 2:
                if verbose_cv(polygon_list_adj, angle_list, K, RT_mat, video.get_actual_frame(color=True), height_start=HEIGHT_START, cube_side=CUBE_SIDE, lenght_axes=30., save_video=result):
                    break
            elif VERBOSE == 3:
                verbose(polygon_list_adj, angle_list, video.get_actual_frame(color=True))
                verbose_2(angle_list, K, RT_mat, video.get_actual_frame(color=True), height_start=HEIGHT_START, cube_side=CUBE_SIDE, lenght_axes=30.)
            # next frame
            video.next_frame()
            # update bar
            pbar.update(1)
        # close video
        video.release()
        # close progress bar
        pbar.close()
        # result.release()
        create_ply_file(grid_3d, name_obj, name_obj + " mash")
