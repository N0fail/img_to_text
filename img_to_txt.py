import numpy as np
from PIL import Image
import requests as rq

def download_image(address, name = 'img'):
    """
    downloads image from specified URL and saves it, and returns a name of saved file

    :param address: URL of the picture to download
    :param name: name of the file to be saved
    :return: name of the file with the image if successful, and

    """
    try:
        extension = address[-5:]
        extension = extension[extension.find('.'):]
        pic = rq.get(address)
        if pic.status_code != 200:
            raise Exception("Error occured while downloading: " + str(pic.status_code) + ": " + pic.reason)
        img_name = "img" + extension
        out = open(img_name, 'wb')
        out.write(pic.content)
        out.close()
    except rq.RequestException as err:
        raise rq.RequestException(err)
    except Exception as err:
        raise Exception(err)

    return img_name

def resize_image_matrix(matrix, *, i_size = None, j_size = None):
    """
    resizes given matrix to given size

    :param matrix: matrix of image to be resized
    :param i_size: height of expected image, computes automatically if not given
    :param j_size: width of expected image, computes automatically if not given
    :return: resized matrix of given size
    """
    if i_size is None:
        if j_size is None:
            return matrix
        else:
            j_scale = matrix.shape[1] / j_size
            i_scale = j_scale * 2 #because letters are high, not squares
    else:
        if j_size is None:
            i_scale = matrix.shape[0] / i_size
            j_scale = i_scale * 0.5 #because letters are high, not squares
        else:
            i_scale = matrix.shape[0] / i_size
            j_scale = matrix.shape[1] / j_size

    i_indices = np.floor(np.arange(0,matrix.shape[0],i_scale)).astype(int)
    j_indices = np.floor(np.arange(0,matrix.shape[1],j_scale)).astype(int)

    result = matrix[i_indices]
    result = result[:,j_indices]

    return result

def get_cluster_matrix(matrix, c_num):
    """
    Creates matrix with cluster numbers from given int matrix

    Lower values from given matrix have lower cluster number in resulting matrix
    :param matrix: int matrix to split to clusters
    :param c_num: expected number of clusters
    :return: matrix with cluster numbers
    """

    variety = np.empty(256, dtype=bool)
    variety[matrix] = True
    variety_list = np.arange(256)[variety]
    if variety_list.shape[0] < c_num:
       raise Exception("Given matrix has less variety of elements then cluster number expected")

    clusters = variety_list[np.random.choice(variety_list.shape[0], c_num, replace=False)]
    clusters.sort()
    result = np.zeros(matrix.shape, dtype=int)
    eps = 0.01

    while True:
        result[:,:] = - 1
        ranges = (clusters[1:c_num] + clusters[:c_num-1])/2

        for i in range(c_num - 1):
            result[np.logical_and(matrix < ranges[i], result == -1)] = i
        result [result == -1] = c_num - 1

        old_clusters = clusters
        for i in range(c_num):
            clusters[i] = (matrix[result == i]).mean()
        if (np.abs(clusters - old_clusters)).max() < eps:
            break

    return result


def to_grey(matrix, coeffs=(0.299, 0.587, 0.114)):
    """
    creates a grey copy of given RGB matrix using given coefficients

    :param matrix: matrix to be transformed
    :param coeffs: coefficients for RGB to be multiplied, will be normalized to summary length of 1
    :return: returns a grey copy of given matrix
    """
    if len(matrix.shape) == 2:
        return matrix.copy()
    a_coeffs = np.zeros(matrix.shape[2],dtype=float)
    rgb = np.arange(3)
    a_coeffs[rgb] = coeffs
    a_coeffs /= sum(a_coeffs)
    result = ((matrix[:, :, :] * a_coeffs).sum(axis=2)).astype(dtype=matrix.dtype)
    return result

def to_text(matrix):
    """
    Creates char matrix from brightness matrix

    :param matrix: brightness matrix
    :return: char matrix
    """
    #symbols = np.array([' ', '\u2591', '\u2592' ,'\u2593', '\u2588'])#Block elements
    #symbols = np.array([' ', '\u2514', '\u2524', '\u253C', '\u256A', '\u256B', '\u256C', '\u2592', '@' ,'#'])#Box Drawing
    #symbols = np.array(['\u25A2', '\u25A4', '\u25A7', '\u25A6', '\u25A9'])#geometric shapes
    #symbols = np.array([' ', '\u2591', '\u2630', '\u2592' ,'\u2593', '\u2588'])
    #symbols = np.array(['\u2804', '\u2814', '\u2815', '\u281E','\u2897', '\u28F6', '\u287F', '\u28FF', '\u2AFC', '\u2A68', '\u2A69'])
    #symbols = np.array(['\u2804', '\u2814', '\u2815', '\u281E', '\u2897', '\u28F6', '\u287F', '\u28FF'])
    symbols = np.array(['\u2804', '\u2814', '\u2815', '\u2897', '\u28F6', '\u28FF'])
    #symbols = np.array(['\'', ',', '\"', ':', ';', '+', '@', '#'])
    #symbols = np.array(['\u2598', '\u259A', '\u2599', '\u2585', '\u2586','\u2587', '\u2588', '\u2589'])
    #symbols = np.array(['\u2A61', '\u2A66', '\u2A67', '\u2A86', '\u2A8A', '\u2A8C', '\u2A93'])
    #symbols = np.array(['\u2F13','\u2F14', '\u2F1C', '\u2F22', '\u2F25', '\u2F35', '\u2F3D', '\u2F46', '\u2F3F', '\u2F7A', '\u2F80', '\u2FA5', '\u2FAA', '\u2FC8', '\u2FC7', '\u2FC5', '\u2FCD', '\u2FD3', '\u2FDD'])
    #symbols = np.array(['.', '\"', '*', 'i', 'u', 'I', 's', 'w', 'P', 'H', '&', 'W', '@', '#'])
    #symbols = np.array(['\'', '*', 'u', 'w', 'U', 'W'])
    symbols = symbols[::-1]
    cluster_matrix = get_cluster_matrix(matrix, len(symbols))
    result = symbols[cluster_matrix]
    return result

def url_img_to_text(url, str_count = 150, col_count = None,file_name = "img.txt"):
    """
    Downloads picture from given url and creates text file with this image

    :param url: picture url
    :param str_count: string count in result file, (computes automaticly if given only col_count)
    :param col_count: column count in result file, (computes automaticly if not given)
    :param file_name: name of text file
    :return: None
    """
    img_name = download_image(url)
    pic = Image.open(img_name)
    pix = np.array(pic)
    new_pix = resize_image_matrix(pix, i_size=str_count, j_size=col_count)
    grey_pix = to_grey(new_pix)

    text = to_text(grey_pix)
    np.savetxt(file_name, text, fmt="%s", encoding='utf-8',delimiter='')

