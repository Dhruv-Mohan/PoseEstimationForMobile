import numpy as np

def write_style_menpo(file_handle, pts):
    num_pts = pts.shape[0]  # assuming pts is an nparray
    file_handle.write('version: 1\nn_points: ' + str(num_pts) + '\n{ \n')
    for ptx, pty in pts:
        file_handle.write(str(ptx) + ' ' + str(pty) + '\n')
    file_handle.write('}')


def write_style_normal(file_handle, pts):
    for ptx, pty in pts:
        file_handle.write(str(ptx) + ' ' + str(pty) + '\n')


def write_pts(file_handle, pts, menpo=True):
    if menpo:
        write_style_menpo(file_handle, pts)
    else:
        write_style_normal(file_handle, pts)


def get_pts(file_name, patches):
    with open(file_name, 'r') as file_read:
        data = file_read.read()
        data = data.split()

    if data[-1] in '}':
        pts = get_type_menpo(data, patches)
    else:
        pts = get_type_normal(data, patches)

    return pts


def get_type_menpo(data, patches):
    data = data[5:-1]
    data = np.asarray(data, np.float32)
    data = np.reshape(data, (patches, 2))
    return data


def get_type_normal(data, patches = None):
    data = np.asarray(data, np.float32)
    data = np.reshape(data, (patches, 2))
    return data

