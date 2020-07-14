from pathlib import Path
import json


PARTS_2D = ['pose_keypoints_2d',
            'face_keypoints_2d',
            'hand_left_keypoints_2d',
            'hand_right_keypoints_2d']

POSE_LINKS = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
              (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
              (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]

FACE_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
              (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13),
              (13, 14), (14, 15), (15, 16),
              (17, 18), (18, 19), (19, 20), (20, 21),
              (22, 23), (23, 24), (24, 25), (25, 26),
              (27, 28), (28, 29), (29, 30),
              (31, 32), (32, 33), (33, 34), (34, 35),
              (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
              (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
              (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
              (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
              (48, 60), (54, 64),
              (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66),
              (66, 67), (67, 60),
              (36, 68), (68, 39)]

HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4),
              (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20)]


def json_pack(snippets_dir, video_name, frame_width, frame_height, label='unknown',
              label_index=-1):
    sequence_info = []
    p = Path(snippets_dir)

    for path in p.glob(video_name+'*.json'):
        json_path = str(path)
        # print(path)
        frame_id = int(path.stem.split('_')[-2])
        frame_data = {'frame_index': frame_id}
        data = json.load(open(json_path))
        skeletons = []

        for person in data['people']:
            skeleton = {}
            skeleton['pose'] = []
            skeleton['score'] = []

            for part_name in PARTS_2D:
                score, coordinates = read_coordinates(
                    part_name, person, frame_width, frame_height)
                skeleton['pose'] += coordinates
                skeleton['score'] += score
            skeletons += [skeleton]

        frame_data['skeleton'] = skeletons
        sequence_info += [frame_data]

    video_info = dict()
    video_info['data'] = sequence_info
    video_info['label'] = label
    video_info['label_index'] = label_index
    return video_info

def read_coordinates(part_name, person, frame_width, frame_height):
    score, coordinates = [], []
    keypoints = person[part_name]
    for i in range(0, len(keypoints), 3):
        coordinates += [keypoints[i]/frame_width,
                        keypoints[i + 1]/frame_height]
        score += [keypoints[i + 2]]
    return score, coordinates