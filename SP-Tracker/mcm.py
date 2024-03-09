def speed_direction_batch_c(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = (dets[:,0] + dets[:,2])/2.0, (dets[:,1]+dets[:,3])/2.0
    CX2, CY2 = (tracks[:,0] + tracks[:,2]) /2.0, (tracks[:,1]+tracks[:,3])/2.0
    dx = CX1 - CX2 
    dy = CY1 - CY2 
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm 
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_lt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,1]
    CX2, CY2 = tracks[:,0], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_rt(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,0], dets[:,3]
    CX2, CY2 = tracks[:,0], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_lb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,1]
    CX2, CY2 = tracks[:,2], tracks[:,1]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def speed_direction_batch_rb(dets, tracks):
    tracks = tracks[..., np.newaxis]
    CX1, CY1 = dets[:,2], dets[:,3]
    CX2, CY2 = tracks[:,2], tracks[:,3]
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx**2 + dy**2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx # size: num_track x num_det

def cost_vel(Y, X, trackers, velocities, detections, previous_obs, vdc_weight):
    # Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    # iou_matrix = iou_batch(detections, trackers)
    scores = np.repeat(detections[:, -1][:, np.newaxis], trackers.shape[0], axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores
    return angle_diff_cost

def associate_5_points(detections, trackers, iou_threshold, lt, rt, lb, rb, C, previous_obs, vdc_weight, iou_type=None, args=None):
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    Y1, X1 = speed_direction_batch_lt(detections, previous_obs)
    Y2, X2 = speed_direction_batch_rt(detections, previous_obs)
    Y3, X3 = speed_direction_batch_lb(detections, previous_obs)
    Y4, X4 = speed_direction_batch_rb(detections, previous_obs)
    YC, XC = speed_direction_batch(detections, previous_obs)
    cost_lt = cost_vel(Y1, X1, trackers, lt, detections, previous_obs, vdc_weight)
    cost_rt = cost_vel(Y2, X2, trackers, rt, detections, previous_obs, vdc_weight)
    cost_lb = cost_vel(Y3, X3, trackers, lb, detections, previous_obs, vdc_weight)
    cost_rb = cost_vel(Y4, X4, trackers, rb, detections, previous_obs, vdc_weight)
    cost_C = cost_vel(YC, XC, trackers, C, detections, previous_obs, vdc_weight)

    iou_matrix = iou_type(detections, trackers)
    angle_diff_cost = cost_lt + cost_rt + cost_lb + cost_rb + cost_C

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-(iou_matrix + angle_diff_cost))
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def linear_assignment(cost_matrix, thresh=0.):
try:        # [hgx0411] goes here!
    import lap
    if thresh != 0:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    else:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])
except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))