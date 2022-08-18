import numpy as np
from math import sqrt
from imageio import imread
import outputByteMetrics as om
from HuffmanEncoder import huffmanCode
from matplotlib import pyplot as plt
import pickle

intType = "int8"
dimIntType = "int32"
dimIntBytes = 4
byte_order = "big"
nb_int_sig = int.from_bytes(b'\x80', byte_order, signed=True)


# Returns the image as an NumPy array
# If the image is RGB, returns 3 arrays for R, G, B.
# If the images is grayscale, returns one array.
def load_image_as_array(image_filename):
    image = imread(image_filename)
    if len(image.shape) > 2:
        return image[:,:,0].astype("int32"), \
               image[:,:,1].astype("int32"), \
               image[:,:,2].astype("int32")
    else:
        return image.astype("int32")


# Shifts the colors of the image
#   to the range of -128 to 127
def shift_colors(img):
    return img - 128


# Unshifts the colors by adding 128
# Shifts image colors to the range of 0 to 255
def unshift_colors(img):
    return img + 128


# Creates the transform matrices for the DCT.
# Parameters:
#   block_size : The size of the block that the DCT will be performed on.
# Returns:
#   T : Numpy array of the DCT matrix for the rows.
#   Tt : Numpy array that is the transpose of T. Performs DCT on columns.
def create_transform(block_size=8):
    T = np.zeros((block_size, block_size))
    one_root_n = 1/sqrt(block_size)
    
    for i in range(0, block_size):
        for j in range(0, block_size):
            if i == 0:
                T[i,j] = one_root_n
            else:
                T[i,j] = sqrt(2 / block_size) * np.cos(((2 * j + 1) * i * np.pi) / 2 / block_size)

    return T, T.transpose()


# Transforms matrix M using the provided transformation matrices.  Used for both
# the discrete cosine transform and the inverse DCT.
#       DCT(M) = T x M x Tt  IDCT(M) = Ti x M x Tti
#       For IDCT, Ti = inverse(T) and Tti = inverse(Tt).
# Parameters:
#   M : Square numpy array that is the target of the DCT
#   T : Numpy array representing transformation matrix to use
#   Tt : Numpy array representing transpose of T
# Returns:
#   dct_block : numpy array representing DCT(M) = TMTt
def transform(M, T, Tt):
    dct_block = np.matmul(np.matmul(T, M), Tt)
    return dct_block
    

# Creates Quantization matrix with optional parameter quality.
# Parameters:
#   quality : Default 50.  Determines how the quantization matrix will be
#             generated.  Quality can be an int from 1 to 99, with 99 resulting
#             in highest quality quantization matrix.
# Returns:
#   Q : quantization matrix generated according to quality.
def create_Q(quality=50):
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]])
    
    if quality == 50:
        return Q
    elif quality > 50:
       return Q * ((100 - quality)/50)
    else:
        return Q * (50 / quality)


# Quantizes a dct block using the given quality (default 50)
# Parameters:
#   dct_block : numpy array holding block of image after application of DCT.
#   quality : Default 50.  Quality of quantization from 1 - 99, with 99
#             resulting in highest quality.
# Returns:
#   q_block : numpy array holding quantized dct block.
def quantize(dct_block, quality=50):
    Q = create_Q(quality)
    q_block = np.round(np.divide(dct_block, Q)).astype("int32")
    return q_block


# Dequantizes a quantized block according to the original quantization matrix
# Parameters:
#   q_block : numpy array representing quantized DCT block
#   q_original : original numpy array used to quantize q_block
# Returns:
#   dequantized_block : numpy array representing q_block dequantized.
def dequantize(q_block, q_original):
    dequantized_block = np.multiply(q_block, q_original)
    return dequantized_block


# Reads the coefficients of a quantized DCT block into a list. Returned
# coefficient list is terminated by an nb_int_sig (NextBlock_int_signal).
# Parameters:
#   q_block : numpy array containing quantized dct block.
# Returns:
#   coefficient_list : list of all non-zero (and some zero) coefficients
#                      contained in q_block.
def read_coefficients(q_block):
    coefficient_list = zigZagMatrix(q_block)
    coefficient_list.append(nb_int_sig)
    return coefficient_list


# Encodes the compressed DCT coefficient output using a huffman encoding tree.
# Parameters:
#   compressed_out : List of DCT coefficients for each block. Coefficients for
#                    each block are immediately followed by a nb_int_sig.
# Returns:
#   encoded_output : Bits represented by a string. Bit data is encoded from
#                    compressed_out using a huffman tree.
#   code_map : Dictionary mapping binary sequence to value.  Keys are strings
#              representing binary sequences, values are integers.
def huffman(compressed_out):
    coefficient_frequency_map = om.getCoeffFrequencies(compressed_out)
    value_to_symbol, code_map = huffmanCode(coefficient_frequency_map)
    encoded_output = ""

    for coefficient in compressed_out:
        encoded_output += value_to_symbol[coefficient]

    return encoded_output, code_map


# M assumed to be square matrix
# Accepts a NumPy matrix and returns items as a list of bytes
# Each item is represented by intBytes number of bytes.
def zigZagMatrix(M):
    m,n = M.shape
    buffer = []
    output = []

    row = 0
    col = 0
    nextDiag = True

    # direction = True: go right-up
    # direction = false: go left-down
    direction = True

    for i in range(0, m*n):
        item = M[row, col]
        buffer.append(item)
        if item != 0:
            output.extend(buffer)
            buffer = []
        
        if nextDiag == True:
            if direction == True:
                if col == n-1:
                    row += 1
                else:
                    col += 1
            else:
                if row == n-1:
                    col += 1
                else:
                    row += 1
                    
            nextDiag = False
            direction = ~direction
        else:
            if direction == True:
                row -= 1
                col += 1
                if row == 0 or col == n-1:
                    nextDiag = True
            else:
                row += 1
                col -= 1
                if row == m-1 or col == 0:
                    nextDiag = True
    return output


# Compresses an image as numpy array to bytes that can be written to a file.
# Parameters:
#   image : numpy array representing a grayscale image.
#   blockSize : Default 8. Size of blocks of image to perform DCT on.
#   quality : Default 50. Quality of compressed image from 1 to 99, with
#             99 resulting in highest quality.
# Returns:
#   file_bytes : bytes object representing compressed image. Image data and
#                metadata formatted according to following diagram:
#                [pickled code map][4 byte m][4 byte n][2 byte Q][image data]
#   dct_image : numpy array that represents image after the DCT is applied to
#               each block.
#   q_image : numpy array with quantized results of the DCT
#
def compress(image, blockSize=8, quality=50):
    mi,ni = image.shape
    
    # Pad the image if not divisible by blockSize X blockSize blocks
    if mi % blockSize != 0:
        image = np.pad(image, ((0,blockSize - mi % blockSize), (0,0)), mode="edge")

    if ni % blockSize != 0:
        image = np.pad(image, ((0,0), (0,blockSize - ni % blockSize)), mode="edge")
        
    m,n = image.shape

    # Shift colors to -128 to 127
    image = shift_colors(image)

    # Create DCT transformation matrices
    T, Tt = create_transform(blockSize)

    compressed_out = []
    q_image = np.zeros((m, n))
    dct_image = np.zeros((m,n))

    # For each block of the image:
    #   1. Perform the DCT
    #   2. Quantize the DCT result
    #   3. Read the quantized coefficients and add to compressed_out list
    for i in range(0, m, blockSize):
        for j in range(0, n, blockSize):
            dct_block = transform(image[i:i + blockSize, j:j + blockSize], T, Tt)
            dct_image[i:i+blockSize, j:j+blockSize] = dct_block
            q_block = quantize(dct_block, quality)
            q_image[i:i + blockSize, j:j + blockSize] = q_block
            compressed_out.extend(read_coefficients(q_block))

    # Encode coefficients to bits and save code_map for decoding
    out_bits, code_map = huffman(compressed_out)

    # Save code map by pickling it and saving bytes
    code_map_bytes = pickle.dumps(code_map, protocol=pickle.HIGHEST_PROTOCOL)

    # Convert encoded bits to bytes
    data_bytes = bitstring_to_bytes(out_bits)

    # Convert image size to bytes
    size_bytes = mi.to_bytes(dimIntBytes, byteorder=byte_order, signed=False) + \
                 ni.to_bytes(dimIntBytes, byteorder=byte_order, signed=False)

    # Convert quality to bytes
    q_bytes = quality.to_bytes(2, byteorder=byte_order, signed=False)

    # Create file bytes by formatting previous bytes according to following:
    #   [pickled code map][4 byte m][4 byte n][2 byte Q][image data]
    file_bytes = code_map_bytes + size_bytes + q_bytes + data_bytes

    return file_bytes, dct_image, q_image


# Saves file_bytes to specified filename.
# Default filename="compressed_image.EE341"
def save_to_file(file_bytes, filename="compressed_image.EE341"):
    file = open(filename, "wb")
    file.write(file_bytes)
    file.close()
    return


# Converts string representation of bits to bytes
def bitstring_to_bytes(s):
    if len(s) % 8 != 0:
        for i in range(0, 8 - len(s) % 8):
            s += '0'

    out_bytes = b''
    for i in range(0, len(s), 8):
        if i + 8 > len(s):
            next_byte = int(s[i:], 2).to_bytes(1, byteorder=byte_order)
        else:
            next_byte = int(s[i:i+8], 2).to_bytes(1, byteorder=byte_order)

        out_bytes += next_byte

    return out_bytes


# Converts bytes to string representation of bits
def bytes_to_bitstring(data_bytes):
    out_bits = ""

    data_bytes = list(data_bytes)
    end_i = len(data_bytes) - 1

    for i in range(0, end_i):
        out_bits += bin(data_bytes[i])[2:].zfill(8)

    out_bits += bin(data_bytes[end_i])[2:].zfill(8)
    return out_bits


# Decompresses a compressed image and returns a numpy array of the image.
# Parameters:
#   filename : String, file name of the compressed image.
#   blockSize : block sized used in original image. Default is 8
# Returns:
#   decomp_image : numpy array containing decompressed image.
#   dct_decomp : numpy array containing the decompressed dct_block prior to
#                performing the IDCT.
#   q_image : numpy array containing quantized image created from compressed
#             image file data.
def decompress(filename, blockSize=8):
    # Open file
    file = open(filename, "rb")

    # Get code map
    code_map = pickle.load(file)

    # read the image data
    data = file.read()

    ni_end_index = 2 * dimIntBytes
    quality_end_index = ni_end_index + 2

    # Extract image dimensions and data
    mi = int.from_bytes(data[0:dimIntBytes], byte_order, signed=False)
    ni = int.from_bytes(data[dimIntBytes:ni_end_index], byte_order, signed=False)
    quality = int.from_bytes(data[ni_end_index:quality_end_index], byte_order, signed=False)
    data = data[10:]

    Q = create_Q(quality)

    # check if original image is divisible by 8x8 blocks
    if mi % blockSize != 0:
        m = mi + blockSize - mi % blockSize
    else:
        m = mi
        
    if ni % blockSize != 0:
        n = ni + blockSize - ni % blockSize
    else:
        n = ni

    # get bits
    data_bits = bytes_to_bitstring(data)

    # Decode file values using codeMap
    sequence = ""
    q_data = []
    for i in range(0, len(data_bits)):
        sequence += data_bits[i]
        if sequence in code_map:
            q_data.append(code_map[sequence])
            sequence = ""

    q_image = np.zeros((m,n))
    dct_decomp = np.zeros((m,n))
    decomp_image = np.zeros((m,n))
    cdi = 0

    # Create inverse transform matrices
    T, Tt = create_transform(blockSize)
    Ti = np.linalg.inv(T)
    Tti = np.linalg.inv(Tt)

    # For q_data:
    #   1. Find the next index of nb_int_sig, cdi
    #   2. De zig zag all coefficients up to cdi into a new q_block
    #   3. Dequantize the new q_block
    #   4. Perform IDCT on dequantized block to get image_block
    #   5. Add image_block to decompressed image
    for i in range(0, m, blockSize):
        for j in range(0, n, blockSize):
            q_data = q_data[cdi:]
            q_block, cdi = dezag(q_data, blockSize)
            q_image[i:i+blockSize, j:j+blockSize] = q_block
            dct_block = dequantize(q_block, Q)
            dct_decomp[i:i+blockSize, j:j+blockSize] = dct_block
            img_block = transform(dct_block, Ti, Tti)
            decomp_image[i:i+blockSize, j:j+blockSize] = img_block

    # Shift colors back to 0 to 255 and trim image to original dimensions
    decomp_image = unshift_colors(decomp_image[0:mi,0:ni])

    return decomp_image, dct_decomp, q_image


def dezag(q_data, blockSize):
    next_nb_int_sig = q_data.index(nb_int_sig)
    q_block = np.zeros((blockSize,blockSize))

    row = 0
    col = 0
    nextDiag = True

    # direction = True: go right-up
    # direction = false: go left-down
    direction = True

    for i in range(0, next_nb_int_sig):
        q_block[row, col] = q_data[i]
        
        if nextDiag == True:
            if direction == True:
                if col == blockSize-1:
                    row += 1
                else:
                    col += 1
            else:
                if row == blockSize-1:
                    col += 1
                else:
                    row += 1
                    
            nextDiag = False
            direction = ~direction
        else:
            if direction == True:
                row -= 1
                col += 1
                if row == 0 or col == blockSize-1:
                    nextDiag = True
            else:
                row += 1
                col -= 1
                if row == blockSize-1 or col == 0:
                    nextDiag = True

    return q_block, next_nb_int_sig + 1


# Compresses and decompresses the provided numpy image and displays the original
# image, the DCT, the quantized DCT, the quantized DCT after decompressing from
# a file, the decompressed DCT, and the decompressed image. Prints the original
# image size, the compressed image size, and the compression ratio.
# Parameters:
#   img : numpy array containing a grayscale image
# No returned values.
# Plots img, decompressed image, and the image during intermediate steps of the
# compression and decompression. Prints original image size, compressed image
# size, and the final compression ratio.
def compress_decompress_test(img, figsize=(6, 6)):
    print("Beginning Compression...")
    file_bytes, dct_image, q_image = compress(img)
    save_to_file(file_bytes, "comp_img.EE341")

    print("Beginning Decompression...")
    img2, dct_decomp, q_decomp = decompress("comp_img.EE341")

    print("-----------------------")
    m, n = img.shape
    print("  Original Image Size in Bytes: " + str(m * n))
    print("Compressed Image Size in Bytes: " + str(len(file_bytes)))
    print("       Final Compression Ratio: " + str(round(len(file_bytes) / (m * n), 2)))

    np.set_printoptions(precision=2, suppress=True)
    #print(c_bytes)
    #print(d_bytes)

    plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.subplot(3, 2, 1)
    plt.axis("off")
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.subplot(3, 2, 2)
    plt.axis("off")
    plt.title("Decompressed Image")
    plt.imshow(img2, cmap='gray')
    plt.subplot(3, 2, 5)
    plt.axis("off")
    plt.title("DCT of Image")
    plt.imshow(q_image, cmap='gray')
    plt.subplot(3, 2, 6)
    plt.axis("off")
    plt.title("Dequantized DCT from File")
    plt.imshow(q_decomp, cmap='gray')
    plt.subplot(3, 2, 3)
    plt.axis("off")
    plt.title("Quantized DCT")
    plt.imshow(dct_image, cmap='gray')
    plt.subplot(3, 2, 4)
    plt.axis("off")
    plt.title("Quantized DCT from file")
    plt.imshow(dct_decomp, cmap='gray')
    plt.show()


# Compresses an image to a specified file name with default quality 50.
# Parameters:
#   img_filename : filename of the original grayscale image.
#   compressed_filename : filename to save compressed image as
#   quality : Default 50.  Quality of the compression from 1 - 99. Quality of 99
#             corresponds to higher image quality
# No Return Values. Saves compressed image under compressed_filename.
def compress_image(img_filename, compressed_filename, quality=50):
    image = load_image_as_array(img_filename)
    file_bytes, q_image, dct_image = compress(image, quality=quality)
    save_to_file(file_bytes, compressed_filename)
    print('Finished compressing "' + img_filename + '" to "' + compressed_filename + '"')


# Decompresses the given compressed image file and returns a numpy array with
# the decompressed image.
# Parameters:
#   filename : file name of the compressed image.
# Returns:
#   decompressed_image : A numpy array containing the decompressed image.
def decompress_image(filename):
    decompressed_image, q_decomp, dct_decomp = decompress(filename)
    return decompressed_image
